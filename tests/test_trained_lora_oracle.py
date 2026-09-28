"""Opt-in trained native CUDA regression; set a real pinned checkpoint path."""
import os
from pathlib import Path

import pytest
import torch

from kev_laya.checkpoint import load_checkpoint
from kev_laya.encoding import Limits, encode_request
from kev_laya.model import LoRALinear
from kev_laya.native_validation import PROTOCOL, hf_parity
from kev_laya.schema import SystemOneRequest


@pytest.mark.native
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_trained_lora_checkpoint_matches_independent_pinned_oracle():
    checkpoint = os.environ.get("KEV_LAYA_TRAINED_NATIVE_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set KEV_LAYA_TRAINED_NATIVE_CHECKPOINT to a verified real-weight checkpoint")
    model, tokenizer, point = load_checkpoint(Path(checkpoint), "cuda:0")
    assert model.native_weights_loaded
    assert model.training_steps >= 1
    assert point["training_state"]["step"] >= 1
    assert model.cfg.lora_rank > 0
    adapters = [getattr(layer.self_attn, name) for layer in model.backbone.layers
                for name in ("q_proj", "v_proj")]
    assert all(isinstance(adapter, LoRALinear) for adapter in adapters)
    assert any(torch.count_nonzero(adapter.b).item() for adapter in adapters)
    assert PROTOCOL["hf_hidden_atol"] == PROTOCOL["hf_hidden_rtol"] == 1e-4
    request = SystemOneRequest(
        state="A short real-weight reference state.",
        questions={"q": {"type": "noul", "instructions": "Is the state short?"}},
        model="kev-laya-preview",
    )
    encoded = encode_request(request, tokenizer, Limits())
    result = hf_parity(model.eval(), encoded.state + encoded.branches[0].ids)
    assert result["passed"], result
    assert result["oracle"].endswith("/eager+two-linear-LoRA")
