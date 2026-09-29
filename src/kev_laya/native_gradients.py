"""Bounded, shape-stable CUDA BF16 training derivatives.

The fused SDPA forward remains the native implementation. Its long backward is
recomputed in double query tiles so cached and full-row requests use the same
causal derivative and shared-prefix contributions accumulate before rounding.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


DERIVATIVE_VERSION = "cuda-bf16-canonical-v2"
ATTENTION_QUERY_TILE = 64


def causal_vjp(q: Tensor, k: Tensor, v: Tensor, upstream: Tensor,
               *, split_at: int | None = None, tile: int = ATTENTION_QUERY_TILE
               ) -> tuple[Tensor, Tensor, Tensor]:
    """Lower-right causal Q/K/V VJP with at most one score tile resident."""
    if q.shape[:-2] != k.shape[:-2] or k.shape != v.shape or q.shape != upstream.shape:
        raise ValueError("attention VJP shapes differ")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] < q.shape[-2] or tile < 1:
        raise ValueError("invalid causal attention dimensions")
    q, k, v, upstream = (item.double() for item in (q, k, v, upstream))
    length, key_length = q.shape[-2], k.shape[-2]
    prefix = key_length - length
    scale = 1 / math.sqrt(q.shape[-1])
    dq, dk, dv = torch.empty_like(q), torch.zeros_like(k), torch.zeros_like(v)
    start = 0
    while start < length:
        stop = min(start + tile, length)
        if split_at is not None and prefix + start < split_at < prefix + stop:
            stop = split_at - prefix
        last_key = prefix + stop
        qc, gc = q[..., start:stop, :], upstream[..., start:stop, :]
        kc, vc = k[..., :last_key, :], v[..., :last_key, :]
        scores = (qc @ kc.transpose(-2, -1)) * scale
        allowed = (torch.arange(last_key, device=q.device)[None, :]
                   <= prefix + torch.arange(start, stop, device=q.device)[:, None])
        probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        alpha = gc @ vc.transpose(-2, -1)
        ds = probabilities * (alpha - (probabilities * alpha).sum(-1, keepdim=True))
        dq[..., start:stop, :] = (ds @ kc) * scale
        dk[..., :last_key, :] += (ds.transpose(-2, -1) @ qc) * scale
        dv[..., :last_key, :] += probabilities.transpose(-2, -1) @ gc
        start = stop
    return dq, dk, dv


class CanonicalLongSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor,
                attn_mask, is_causal: bool, split_at: int | None):
        output = F.scaled_dot_product_attention(
            query, key.to(query.dtype), value.to(query.dtype),
            attn_mask=attn_mask, is_causal=is_causal, dropout_p=0.0)
        ctx.save_for_backward(query, key, value)
        ctx.split_at = split_at
        return output

    @staticmethod
    def backward(ctx, upstream: Tensor):
        query, key, value = ctx.saved_tensors
        dq, dk, dv = causal_vjp(query, key, value, upstream,
                               split_at=ctx.split_at)
        return dq.to(query.dtype), dk.to(key.dtype), dv.to(value.dtype), None, None, None


class CanonicalShortAttention(torch.autograd.Function):
    """Preserve short eager BF16 forward while accumulating its VJP in double."""

    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, offset: int):
        rounded_key, rounded_value = key.to(query.dtype), value.to(query.dtype)
        length, total = query.shape[-2], rounded_key.shape[-2]
        scale = query.shape[-1] ** -0.5
        scores = (query.double() @ rounded_key.transpose(-2, -1).double()).to(query.dtype) * scale
        allowed = torch.ones((length, total), device=query.device, dtype=torch.bool).tril(
            diagonal=total - length)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = (probabilities.double() @ rounded_value.double()).to(query.dtype)
        ctx.save_for_backward(query, rounded_key, rounded_value, probabilities)
        ctx.offset = offset
        return output

    @staticmethod
    def backward(ctx, upstream: Tensor):
        query, key, value, probabilities = ctx.saved_tensors
        q, k, v, p, g = (item.double() for item in
                         (query, key, value, probabilities, upstream))
        scale = query.shape[-1] ** -0.5
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        for index in range(q.shape[-2]):
            valid = ctx.offset + index + 1
            qi, gi = q[..., index, :], g[..., index, :]
            pi, ki, vi = p[..., index, :valid], k[..., :valid, :], v[..., :valid, :]
            dp = (gi[..., None, :] * vi).sum(-1)
            ds = pi * (dp - (pi * dp).sum(-1, keepdim=True)) * scale
            dq[..., index, :] = (ds[..., None] * ki).sum(-2)
            dk[..., :valid, :] += ds[..., None] * qi[..., None, :]
            dv[..., :valid, :] += pi[..., None] * gi[..., None, :]
        return dq.to(query.dtype), dk, dv, None


class CanonicalLoRAProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None,
                a: Tensor, b: Tensor, scale: float):
        rows = x.reshape(-1, x.shape[-1])
        outputs, inners = [], []
        for start in range(0, rows.shape[0], 1024):
            tile = rows[start:start + 1024]
            count = tile.shape[0]
            if count < 1024:
                tile = F.pad(tile, (0, 0, 0, 1024 - count))
            inner = F.linear(tile, a)
            projected = F.linear(tile, weight, bias) + F.linear(inner, b) * scale
            outputs.append(projected[:count])
            inners.append(inner[:count])
        ctx.save_for_backward(x, weight, a, b, torch.cat(inners, dim=0))
        ctx.scale = scale
        return torch.cat(outputs, dim=0).reshape(*x.shape[:-1], outputs[0].shape[-1])

    @staticmethod
    def backward(ctx, upstream: Tensor):
        x, weight, a, b, inner = ctx.saved_tensors
        g = upstream.reshape(-1, upstream.shape[-1]).double()
        x_eff = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).double()
        w_eff, a_eff, b_eff = (item.to(torch.bfloat16).double()
                              for item in (weight, a, b))
        grad_inner = (g @ b_eff) * ctx.scale
        grad_x = g @ w_eff + grad_inner @ a_eff
        grad_a = grad_inner.T @ x_eff
        grad_b = (g * ctx.scale).T @ inner.double()
        grad_bias = g.sum(0) if ctx.needs_input_grad[2] else None
        grad_weight = g.T @ x_eff if ctx.needs_input_grad[1] else None
        return (grad_x.reshape_as(x).to(x.dtype),
                None if grad_weight is None else grad_weight.to(weight.dtype),
                None if grad_bias is None else grad_bias.to(upstream.dtype),
                grad_a.to(a.dtype), grad_b.to(b.dtype), None)
