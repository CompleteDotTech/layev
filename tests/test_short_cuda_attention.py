"""GPU regression for the real-weight oracle's short eager attention path."""
import pytest
import torch
from torch.nn import functional as F

from kev_laya.model import Attention, BackboneConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("prefix_length,suffix_length", [(0, 17), (11, 7)])
def test_short_cuda_uses_eager_arithmetic_and_right_aligned_mask(monkeypatch, prefix_length, suffix_length):
    torch.manual_seed(17)
    cfg = BackboneConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=128, max_position_embeddings=512)
    attention = Attention(cfg).cuda().eval()
    x = torch.randn(1, suffix_length, cfg.hidden_size, device="cuda")
    parent = None
    if prefix_length:
        prefix = torch.randn(1, prefix_length, cfg.hidden_size, device="cuda")
        _, parent = attention(prefix, 0)

    def fail_sdpa(*args, **kwargs):
        raise AssertionError("short CUDA path used SDPA instead of pinned eager arithmetic")

    monkeypatch.setattr(F, "scaled_dot_product_attention", fail_sdpa)
    result, _ = attention(x, prefix_length, parent)
    q = attention.q_proj(x).view(1, suffix_length, attention.h, attention.hd).transpose(1, 2)
    k = attention.k_proj(x).view(1, suffix_length, attention.kh, attention.hd).transpose(1, 2)
    v = attention.v_proj(x).view(1, suffix_length, attention.kh, attention.hd).transpose(1, 2)
    pos = torch.arange(prefix_length, prefix_length + suffix_length, device="cuda")
    q, k = attention.rotate(q, pos), attention.rotate(k, pos)
    if parent is not None:
        k, v = torch.cat((parent[0], k), -2), torch.cat((parent[1], v), -2)
    k = k.repeat_interleave(attention.h // attention.kh, dim=1)
    v = v.repeat_interleave(attention.h // attention.kh, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (attention.hd ** -0.5)
    allowed = torch.ones((suffix_length, k.shape[-2]), dtype=torch.bool, device="cuda").tril(diagonal=prefix_length)
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    expected = torch.matmul(F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype), v)
    expected = attention.o_proj(expected.transpose(1, 2).reshape(1, suffix_length, -1))
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_longer_cuda_sequence_keeps_sdpa(monkeypatch):
    cfg = BackboneConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=128, max_position_embeddings=512)
    attention = Attention(cfg).cuda().eval()
    original = F.scaled_dot_product_attention
    calls = []

    def record(*args, **kwargs):
        calls.append((args[0].shape[-2], args[1].shape[-2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    output, _ = attention(torch.randn(1, 257, cfg.hidden_size, device="cuda"), 0)
    assert output.shape == (1, 257, cfg.hidden_size)
    assert calls == [(257, 257)]
