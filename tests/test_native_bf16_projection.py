"""CUDA BF16/FP32 projection shape and gradient gates for cached branches."""
import pytest
import torch
from torch import nn

from kev_laya.model import LoRALinear, backbone_project


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA native projection gate requires a GPU")
@pytest.mark.parametrize("lora", [False, True])
def test_bf16_projection_is_row_count_stable_with_backward(lora):
    torch.manual_seed(1763)
    layer = nn.Linear(64, 96, bias=True).cuda()
    if lora:
        layer = LoRALinear(layer, rank=4, alpha=8).cuda()
        with torch.no_grad():
            layer.b.normal_(std=0.01)
    token = torch.randn(3, 64, device="cuda")
    left = token.clone().requires_grad_()
    right = torch.cat((torch.randn(1028, 64, device="cuda"), token), dim=0).requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        short = backbone_project(layer, left)
        long = backbone_project(layer, right)[-3:]
        assert torch.equal(short, long)
        short.float().square().mean().backward()
        long.float().square().mean().backward()
    assert torch.isfinite(left.grad).all()
    assert torch.isfinite(right.grad).all()
    assert torch.equal(left.grad, right.grad[-3:])
    assert all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters()
               if parameter.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA native projection gate requires a GPU")
@pytest.mark.parametrize("lora", [False, True])
def test_fp32_projection_is_row_count_stable_with_backward(lora):
    torch.manual_seed(1764)
    layer = nn.Linear(64, 96, bias=True).cuda()
    if lora:
        layer = LoRALinear(layer, rank=4, alpha=8).cuda()
        with torch.no_grad():
            layer.b.normal_(std=0.01)
    token = torch.randn(3, 64, device="cuda")
    left = token.clone().requires_grad_()
    right = torch.cat((torch.randn(1028, 64, device="cuda"), token), dim=0).requires_grad_()
    short = backbone_project(layer, left)
    long = backbone_project(layer, right)[-3:]
    assert torch.equal(short, long)
    short.square().mean().backward()
    long.square().mean().backward()
    assert torch.isfinite(left.grad).all()
    assert torch.isfinite(right.grad).all()
    assert torch.equal(left.grad, right.grad[-3:])
