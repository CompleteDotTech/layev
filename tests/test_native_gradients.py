"""Independent small-tensor checks for the bounded causal training VJP."""
import math

import pytest
import torch

from kev_laya.native_gradients import CanonicalFP32ShortAttention, causal_vjp


@pytest.mark.parametrize("prefix", [0, 8])
def test_causal_vjp_matches_independent_double_autograd(prefix):
    generator = torch.Generator().manual_seed(2618 + prefix)
    q = torch.randn(1, 2, 5, 8, generator=generator, dtype=torch.float64,
                    requires_grad=True)
    k = torch.randn(1, 2, prefix + 5, 8, generator=generator,
                    dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, prefix + 5, 8, generator=generator,
                    dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(q.shape, generator=generator, dtype=torch.float64)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    allowed = (torch.arange(k.shape[-2])[None, :]
               <= prefix + torch.arange(q.shape[-2])[:, None])
    output = scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v
    expected = torch.autograd.grad((output * upstream).sum(), (q, k, v))
    actual = causal_vjp(q.detach(), k.detach(), v.detach(), upstream,
                        split_at=prefix + 2, tile=3)
    for observed, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, oracle, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("prefix", [0, 11])
def test_fp32_short_grouped_double_cache_matches_explicit_eager_gqa(prefix):
    generator = torch.Generator(device="cuda").manual_seed(4521 + prefix)
    q0 = torch.randn(1, 4, 7, 16, generator=generator, device="cuda", dtype=torch.float32)
    k0 = torch.randn(1, 2, prefix + 7, 16, generator=generator,
                     device="cuda", dtype=torch.float32).double()
    v0 = torch.randn(k0.shape, generator=generator, device="cuda", dtype=torch.float32).double()
    upstream = torch.randn(q0.shape, generator=generator, device="cuda", dtype=torch.float32)
    q, k, v = [tensor.detach().clone().requires_grad_(True) for tensor in (q0, k0, v0)]
    actual = CanonicalFP32ShortAttention.apply(q, k, v, 2)
    actual_vjp = torch.autograd.grad(actual, (q, k, v), upstream)

    rq, rk, rv = [tensor.detach().clone().requires_grad_(True) for tensor in (q0, k0, v0)]
    repeated_k = rk.float().repeat_interleave(2, dim=1)
    repeated_v = rv.float().repeat_interleave(2, dim=1)
    total = prefix + 7
    padded_q = torch.nn.functional.pad(rq, (0, 0, prefix, 256 - total))
    padded_k = torch.nn.functional.pad(repeated_k, (0, 0, 0, 256 - total))
    padded_v = torch.nn.functional.pad(repeated_v, (0, 0, 0, 256 - total))
    scores = torch.matmul(padded_q, padded_k.transpose(-2, -1)) * (16 ** -0.5)
    allowed = torch.ones((256, 256), device="cuda", dtype=torch.bool).tril()
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    probabilities = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(rq.dtype)
    expected = torch.matmul(probabilities, padded_v)[..., prefix:prefix + 7, :]
    eager_dq, eager_dk, eager_dv, score_vjp = torch.autograd.grad(
        expected, (rq, rk, rv, scores), upstream)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Query VJP preserves the exact rounded FP32 eager graph. Shared K/V cache
    # VJPs accumulate query rows and repeated heads in double before returning
    # to the double cache parent.
    torch.testing.assert_close(actual_vjp[0], eager_dq, atol=0, rtol=0)
    padded_upstream = torch.nn.functional.pad(upstream, (0, 0, prefix, 256 - total))
    per_head_k = torch.einsum("bhqk,bhqd->bhkd", score_vjp.double(),
                              padded_q.detach().double()) * (16 ** -0.5)
    per_head_v = torch.einsum("bhqk,bhqd->bhkd", probabilities.detach().double(),
                              padded_upstream.double())
    analytical_dk = per_head_k.reshape(1, 2, 2, 256, 16).sum(2)[..., :total, :]
    analytical_dv = per_head_v.reshape(1, 2, 2, 256, 16).sum(2)[..., :total, :]
    for actual_cache_vjp, analytical_vjp, eager_vjp in (
            (actual_vjp[1], analytical_dk, eager_dk),
            (actual_vjp[2], analytical_dv, eager_dv)):
        assert bool(torch.isfinite(actual_cache_vjp).all())
        torch.testing.assert_close(actual_cache_vjp, analytical_vjp, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_cache_vjp, eager_vjp,
                                   atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("prefix_length", [1, 11, 62, 127, 200, 254, 255])
def test_fp32_short_fixed256_matches_any_cached_prefix(prefix_length):
    generator = torch.Generator(device="cuda").manual_seed(6256 + prefix_length)
    query = torch.randn(1, 4, 256, 16, generator=generator, device="cuda")
    key = torch.randn(1, 2, 256, 16, generator=generator, device="cuda").double()
    value = torch.randn(1, 2, 256, 16, generator=generator, device="cuda").double()
    upstream = torch.randn(query.shape, generator=generator, device="cuda")
    full_inputs = [tensor.detach().clone().requires_grad_(True)
                   for tensor in (query, key, value)]
    split_inputs = [tensor.detach().clone().requires_grad_(True)
                    for tensor in (query, key, value)]
    whole = CanonicalFP32ShortAttention.apply(*full_inputs, 2)
    fq, fk, fv = split_inputs
    prefix = CanonicalFP32ShortAttention.apply(
        fq[..., :prefix_length, :], fk[..., :prefix_length, :],
        fv[..., :prefix_length, :], 2)
    suffix = CanonicalFP32ShortAttention.apply(
        fq[..., prefix_length:, :], fk, fv, 2)
    split = torch.cat((prefix, suffix), dim=-2)
    torch.testing.assert_close(split, whole, atol=0, rtol=0)
    full_grads = torch.autograd.grad(whole, full_inputs, upstream)
    split_grads = torch.autograd.grad(split, split_inputs, upstream)
    for full_grad, split_grad in zip(full_grads, split_grads, strict=True):
        assert bool(torch.isfinite(full_grad).all())
        assert bool(torch.isfinite(split_grad).all())
        torch.testing.assert_close(split_grad, full_grad, atol=2e-5, rtol=2e-5)
