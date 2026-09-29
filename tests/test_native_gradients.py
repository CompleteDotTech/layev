"""Independent small-tensor checks for the bounded causal training VJP."""
import math

import pytest
import torch

from kev_laya.native_gradients import causal_vjp


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
