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


DERIVATIVE_VERSION = "cuda-bf16-canonical-v3"
FP32_DERIVATIVE_VERSION = "cuda-fp32-canonical-v10"
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


class CanonicalFP32ShortAttention(torch.autograd.Function):
    """Preserve eager FP32 forward with double shared K/V accumulation.

    Double K/V cache remains the shared accumulation boundary. The local
    query derivative retains the rounded reference operation order.
    """

    @staticmethod
    def _eager(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        length, total = query.shape[-2], key.shape[-2]
        if not (1 <= length <= total <= 256) or key.shape != value.shape:
            raise ValueError("short FP32 attention requires 1..256 causal positions")
        # Place every real query at its absolute position in one fixed 256x256
        # GEMM. Cached q[255] and full-row q[255] then use the same row/shape.
        start = total - length
        padded_query = F.pad(query, (0, 0, start, 256 - total))
        padded_key = F.pad(key, (0, 0, 0, 256 - total))
        padded_value = F.pad(value, (0, 0, 0, 256 - total))
        scale = query.shape[-1] ** -0.5
        scores = torch.matmul(padded_query, padded_key.transpose(-2, -1)) * scale
        allowed = torch.ones((256, 256), device=query.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, padded_value)
        return output[..., start:start + length, :]

    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, ratio: int):
        # Round each grouped KV head before repeating it, as in the pinned
        # eager FP32 graph. `key` and `value` remain double cache parents.
        if ratio < 1 or query.shape[1] != key.shape[1] * ratio or key.shape != value.shape:
            raise ValueError("invalid grouped KV short attention shape")
        ctx.save_for_backward(query, key, value)
        ctx.ratio = ratio
        return CanonicalFP32ShortAttention._eager(
            query, key.to(query.dtype).repeat_interleave(ratio, dim=1),
            value.to(query.dtype).repeat_interleave(ratio, dim=1))

    @staticmethod
    def backward(ctx, upstream: Tensor):
        query, key, value = ctx.saved_tensors
        length, total = query.shape[-2], key.shape[-2]
        start = total - length
        ratio = ctx.ratio
        # Recompute the exact rounded FP32 score/softmax graph. Its local
        # query and score VJPs retain the pinned eager operator's cast points.
        with torch.enable_grad():
            q = query.detach().requires_grad_(True)
            k = key.detach().to(query.dtype).repeat_interleave(ratio, dim=1)
            v = value.detach().to(query.dtype).repeat_interleave(ratio, dim=1)
            padded_q = F.pad(q, (0, 0, start, 256 - total))
            padded_k = F.pad(k, (0, 0, 0, 256 - total))
            padded_v = F.pad(v, (0, 0, 0, 256 - total))
            scores = torch.matmul(padded_q, padded_k.transpose(-2, -1)) * (
                query.shape[-1] ** -0.5)
            allowed = torch.ones((256, 256), device=query.device, dtype=torch.bool).tril()
            probabilities = F.softmax(
                scores.masked_fill(~allowed, torch.finfo(scores.dtype).min),
                dim=-1, dtype=torch.float32).to(query.dtype)
            output = torch.matmul(probabilities, padded_v)[..., start:start + length, :]
            dq, ds = torch.autograd.grad(output, (q, scores), upstream)

        # The rounded FP32 score VJP is the derivative of the actual forward.
        # Products and query-row sums into shared K/V use double so that
        # prefix/suffix and full-row gradients have one accumulation precision.
        padded_upstream = F.pad(upstream, (0, 0, start, 256 - total))
        per_head_k = (ds.double().transpose(-2, -1) @ padded_q.detach().double()) * (
            query.shape[-1] ** -0.5)
        per_head_v = probabilities.detach().double().transpose(-2, -1) @ (
            padded_upstream.double())
        batch, heads, _, width = per_head_k.shape
        groups = heads // ratio
        # Group repeated heads and accumulate their shared K/V contribution
        # in double before the single cache-parent cast. Full and cached
        # requests therefore have the same summation precision.
        dk = per_head_k.reshape(batch, groups, ratio, 256, width).sum(dim=2)[..., :total, :]
        dv = per_head_v.reshape(batch, groups, ratio, 256, width).sum(dim=2)[..., :total, :]
        return dq.to(query.dtype), dk.to(key.dtype), dv.to(value.dtype), None


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


class CanonicalMediumAttention(torch.autograd.Function):
    """Tiled eager-arithmetic BF16 attention with a shape-stable double VJP."""

    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, offset: int):
        length, total = query.shape[-2], key.shape[-2]
        rounded_key, rounded_value = key.to(query.dtype), value.to(query.dtype)
        scale = query.shape[-1] ** -0.5
        pieces = []
        for start in range(0, length, ATTENTION_QUERY_TILE):
            stop = min(start + ATTENTION_QUERY_TILE, length)
            key_stop = offset + stop
            scores = (query[..., start:stop, :].double()
                      @ rounded_key[..., :key_stop, :].double().transpose(-2, -1)).to(query.dtype) * scale
            allowed = (torch.arange(key_stop, device=query.device)[None, :]
                       <= offset + torch.arange(start, stop, device=query.device)[:, None])
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            pieces.append((probabilities.double() @ rounded_value[..., :key_stop, :].double()).to(query.dtype))
        ctx.save_for_backward(query, rounded_key, rounded_value)
        ctx.offset = offset
        ctx.key_dtype, ctx.value_dtype = key.dtype, value.dtype
        return torch.cat(pieces, dim=-2)

    @staticmethod
    def backward(ctx, upstream: Tensor):
        query, key, value = ctx.saved_tensors
        q, k, v, g = (item.double() for item in (query, key, value, upstream))
        length = q.shape[-2]
        scale = query.shape[-1] ** -0.5
        dq, dk, dv = torch.empty_like(q), torch.zeros_like(k), torch.zeros_like(v)
        for start in range(0, length, ATTENTION_QUERY_TILE):
            stop = min(start + ATTENTION_QUERY_TILE, length)
            key_stop = ctx.offset + stop
            qc, gc = q[..., start:stop, :], g[..., start:stop, :]
            kc, vc = k[..., :key_stop, :], v[..., :key_stop, :]
            scores = (qc @ kc.transpose(-2, -1)).to(query.dtype) * scale
            allowed = (torch.arange(key_stop, device=query.device)[None, :]
                       <= ctx.offset + torch.arange(start, stop, device=query.device)[:, None])
            probabilities = F.softmax(scores.masked_fill(~allowed, torch.finfo(query.dtype).min),
                                      dim=-1, dtype=torch.float32).to(query.dtype).double()
            alpha = gc @ vc.transpose(-2, -1)
            ds = probabilities * (alpha - (probabilities * alpha).sum(-1, keepdim=True)) * scale
            dq[..., start:stop, :] = ds @ kc
            dk[..., :key_stop, :] += ds.transpose(-2, -1) @ qc
            dv[..., :key_stop, :] += probabilities.transpose(-2, -1) @ gc
        return dq.to(query.dtype), dk.to(ctx.key_dtype), dv.to(ctx.value_dtype), None


class CanonicalLoRAProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None,
                a: Tensor, b: Tensor, scale: float):
        rows = x.reshape(-1, x.shape[-1])
        outputs, inners = [], []
        tile_rows = 1024 if torch.is_autocast_enabled("cuda") else 64
        for start in range(0, rows.shape[0], tile_rows):
            tile = rows[start:start + tile_rows]
            count = tile.shape[0]
            if count < tile_rows:
                tile = F.pad(tile, (0, 0, 0, tile_rows - count))
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
        calculation_dtype = inner.dtype
        x_eff = x.reshape(-1, x.shape[-1]).to(calculation_dtype).double()
        w_eff, a_eff, b_eff = (item.to(calculation_dtype).double()
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
