"""GPU regression for the real-weight oracle's short eager attention path."""
import pytest
import torch
from torch.nn import functional as F

from kev_laya.model import Attention, BackboneConfig, backbone_project


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
    q = backbone_project(attention.q_proj, x).view(1, suffix_length, attention.h, attention.hd).transpose(1, 2)
    k = backbone_project(attention.k_proj, x).view(1, suffix_length, attention.kh, attention.hd).transpose(1, 2)
    v = backbone_project(attention.v_proj, x).view(1, suffix_length, attention.kh, attention.hd).transpose(1, 2)
    pos = torch.arange(prefix_length, prefix_length + suffix_length, device="cuda")
    q, k = attention.rotate(q, pos), attention.rotate(k, pos)
    if parent is not None:
        k, v = torch.cat((parent[0], k), -2), torch.cat((parent[1], v), -2)
    k = k.repeat_interleave(attention.h // attention.kh, dim=1)
    v = v.repeat_interleave(attention.h // attention.kh, dim=1)
    # Place each suffix query at its absolute row in the fixed 256 matrix.
    total = k.shape[-2]
    padded_q = F.pad(q, (0, 0, prefix_length, 256 - total))
    padded_k = F.pad(k, (0, 0, 0, 256 - total))
    padded_v = F.pad(v, (0, 0, 0, 256 - total))
    scores = torch.matmul(padded_q, padded_k.transpose(-2, -1)) * (attention.hd ** -0.5)
    allowed = torch.ones((256, 256), dtype=torch.bool, device="cuda").tril()
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    expected = torch.matmul(F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype),
                            padded_v)[..., prefix_length:prefix_length + suffix_length, :]
    expected = backbone_project(attention.o_proj,
                                expected.transpose(1, 2).reshape(1, suffix_length, -1))
    torch.testing.assert_close(result, expected, atol=0, rtol=0)

    # Retain the independently unpadded eager reference at the logits gate.
    eager_scores = torch.matmul(q, k.transpose(-2, -1)) * (attention.hd ** -0.5)
    eager_allowed = torch.ones((suffix_length, total), dtype=torch.bool,
                               device="cuda").tril(diagonal=prefix_length)
    eager_scores = eager_scores.masked_fill(~eager_allowed,
                                            torch.finfo(eager_scores.dtype).min)
    eager = torch.matmul(F.softmax(eager_scores, dim=-1, dtype=torch.float32).to(q.dtype), v)
    eager = backbone_project(attention.o_proj,
                             eager.transpose(1, 2).reshape(1, suffix_length, -1))
    torch.testing.assert_close(result, eager, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_longer_cuda_sequence_keeps_sdpa(monkeypatch):
    from torch.nn.attention.bias import CausalBias
    cfg = BackboneConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=128, max_position_embeddings=512)
    attention = Attention(cfg).cuda().eval()
    # CausalBias dispatch compares the public SDPA function by identity.
    # Observe its dispatcher without replacing that function.
    original = CausalBias._dispatch
    calls = []

    def record(*args, **kwargs):
        calls.append((args[0].shape[-2], args[1].shape[-2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(CausalBias, "_dispatch", staticmethod(record))
    output, _ = attention(torch.randn(1, 257, cfg.hidden_size, device="cuda"), 0)
    assert output.shape == (1, 257, cfg.hidden_size)
    assert calls == [(1, 257)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_short_bf16_training_preserves_eager_forward_and_double_parent_cache():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    torch.manual_seed(1807)
    cfg = BackboneConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=128, max_position_embeddings=512)
    attention = Attention(cfg).cuda()
    prefix = torch.randn(1, 11, cfg.hidden_size, device="cuda")
    suffix = torch.randn(1, 7, cfg.hidden_size, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        attention.eval()
        _, eval_cache = attention(prefix, 0)
        expected, _ = attention(suffix, 11, eval_cache)
        attention.train()
        _, train_cache = attention(prefix, 0)
        actual, child_cache = attention(suffix, 11, train_cache)
    assert all(t.dtype == torch.float64 for t in (*train_cache, *child_cache))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("prefix_length", [255, 256])
@pytest.mark.parametrize("training", [False, True])
def test_fp32_absolute_query_dispatch_matches_independent_full_row(prefix_length, training):
    torch.manual_seed(3256 + prefix_length)
    cfg = BackboneConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=128, max_position_embeddings=512)
    attention = Attention(cfg).cuda().train(training)
    row = torch.randn(1, prefix_length + 56, cfg.hidden_size, device="cuda")
    full, _ = attention(row, 0)
    first, parent = attention(row[:, :prefix_length], 0)
    suffix, _ = attention(row[:, prefix_length:], prefix_length, parent)
    assert torch.isfinite(full).all() and torch.isfinite(first).all() and torch.isfinite(suffix).all()
    torch.testing.assert_close(torch.cat((first, suffix), dim=1), full, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_fp32_rope_uses_cpu_initialized_frequencies_after_dtype_moves_and_reload():
    cfg = BackboneConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=128, max_position_embeddings=32768)
    attention = Attention(cfg).cuda().float().eval()
    positions = torch.tensor([0, 1, 63, 255, 256, 8191, 8192, 32767], device="cuda")
    values = torch.randn(1, attention.h, len(positions), attention.hd, device="cuda")
    cpu_indices = torch.arange(0, attention.hd, 2, dtype=torch.int64).to(torch.float32)
    cpu_inv = 1.0 / (cfg.rope_theta ** (cpu_indices / attention.hd))
    angles = positions.float()[:, None] * cpu_inv.cuda()[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    rotated_half = torch.cat((-values[..., attention.hd // 2:],
                              values[..., :attention.hd // 2]), dim=-1)
    expected = values * angles.cos()[None, None] + rotated_half * angles.sin()[None, None]
    state = attention.state_dict()
    assert "_fp32_rope_inv_freq_bits" not in state
    torch.testing.assert_close(attention.rotate(values, positions), expected, atol=0, rtol=0)
    split = torch.cat((attention.rotate(values[..., :4, :], positions[:4]),
                       attention.rotate(values[..., 4:, :], positions[4:])), dim=-2)
    torch.testing.assert_close(split, expected, atol=0, rtol=0)
    attention.to(dtype=torch.bfloat16).to(dtype=torch.float32)
    torch.testing.assert_close(attention.rotate(values, positions), expected, atol=0, rtol=0)
    restored = Attention(cfg).cuda().float().eval()
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored.rotate(values, positions), expected, atol=0, rtol=0)
