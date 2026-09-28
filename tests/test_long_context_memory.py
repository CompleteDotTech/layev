"""CUDA checks for bounded long-training activations and grouped attention."""

import pytest
import torch

from kev_laya.model import Attention, BackboneConfig, MLP


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _forward_gradients(module, inputs, *, training, autocast):
    module.train(training)
    module.zero_grad(set_to_none=True)
    x = inputs.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        output = module(x, 0) if isinstance(module, Attention) else module(x)
        if isinstance(output, tuple):
            output = output[0]
        output.float().square().mean().backward()
    return output.detach(), x.grad.detach(), {
        name: parameter.grad.detach().clone() for name, parameter in module.named_parameters()
    }


def test_chunked_mlp_matches_full_row_fp32_gradients():
    torch.manual_seed(227)
    mlp = MLP(BackboneConfig(hidden_size=64, intermediate_size=128)).cuda()
    inputs = torch.randn(1, 2048, 64, device="cuda") * 0.1
    baseline = _forward_gradients(mlp, inputs, training=False, autocast=False)
    candidate = _forward_gradients(mlp, inputs, training=True, autocast=False)
    torch.testing.assert_close(candidate[0], baseline[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(candidate[1], baseline[1], atol=2e-5, rtol=2e-5)
    for name in baseline[2]:
        torch.testing.assert_close(candidate[2][name], baseline[2][name], atol=2e-5, rtol=2e-5)


def test_long_grouped_attention_matches_materialized_heads_bf16():
    torch.manual_seed(229)
    attention = Attention(BackboneConfig(hidden_size=112, num_attention_heads=14,
                                         num_key_value_heads=2,
                                         max_position_embeddings=16384)).cuda()
    inputs = torch.randn(1, 8192, 112, device="cuda") * 0.1
    baseline = _forward_gradients(attention, inputs, training=False, autocast=True)
    candidate = _forward_gradients(attention, inputs, training=True, autocast=True)
    torch.testing.assert_close(candidate[0], baseline[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(candidate[1], baseline[1], atol=2e-5, rtol=2e-5)
    for name in baseline[2]:
        torch.testing.assert_close(candidate[2][name], baseline[2][name], atol=2e-5, rtol=2e-5)
