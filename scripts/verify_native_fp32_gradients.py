"""Independent, opt-in short FP32 actual-weight HF gradient comparison.

Preparation only: this script runs CUDA when explicitly invoked. It does not
change the checkpoint, source tree, or the frozen acceptance tolerances.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SOURCE = Path(__file__).resolve().parents[1]
ALLOCATOR_FRACTION = 0.70
CHECKPOINT = OUTPUT = None
EXPECTED_CHECKPOINT_SHA256 = None
sys.path.insert(0, str(SOURCE / "src"))

from kev_laya.checkpoint import load_checkpoint  # noqa: E402
from kev_laya.encoding import Limits, encode_request  # noqa: E402
from kev_laya.model import LoRALinear  # noqa: E402
from kev_laya.native_validation import (PROTOCOL, merged_backbone,
                                        require_reference_version)  # noqa: E402
from kev_laya.schema import SystemOneRequest  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrainableReferenceLoRA(nn.Module):
    """HF Qwen projection with independent parameters and native operation order."""

    def __init__(self, base: nn.Linear, native: LoRALinear):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(native.a.detach().clone())
        self.b = nn.Parameter(native.b.detach().clone())
        self.scale = native.scale

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def compare(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict:
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("oracle comparison shape, dtype, or device mismatch")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("oracle comparison contains nonfinite values")
    absolute = (a - b).abs()
    denominator = atol + rtol * b.abs()
    ratio = absolute / denominator
    return {"passed": bool(torch.all(absolute <= denominator)),
            "max_absolute": float(absolute.detach().max()),
            "max_tolerance_ratio": float(ratio.detach().max())}


def source_fingerprint():
    import subprocess
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip(),
            "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True).strip()),
            "files": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                      for name in ("src/kev_laya/model.py", "src/kev_laya/native_gradients.py",
                                   "src/kev_laya/training.py", "src/kev_laya/execution.py")}}


def main() -> None:
    global CHECKPOINT, OUTPUT, EXPECTED_CHECKPOINT_SHA256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    CHECKPOINT, OUTPUT = args.checkpoint, args.out
    EXPECTED_CHECKPOINT_SHA256 = args.expected_checkpoint_sha256
    if OUTPUT.exists() or OUTPUT.with_suffix(OUTPUT.suffix + ".failure.json").exists():
        raise FileExistsError("output or failure receipt already exists")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash differs from frozen pin")
    source_identity = source_fingerprint()
    source_commit = source_identity["commit"]
    version = require_reference_version()
    import transformers
    if transformers.__version__ != version:
        raise ValueError("imported Transformers differs from pinned distribution")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this native probe")
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_FRACTION, device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or
            torch.get_float32_matmul_precision() != "highest"):
        raise RuntimeError("FP32 computation settings were not applied")
    free_before, total_memory = torch.cuda.mem_get_info(0)
    torch.cuda.reset_peak_memory_stats(0)
    from transformers import Qwen2Config, Qwen2Model

    model, tokenizer, point = load_checkpoint(CHECKPOINT, "cuda:0")
    if not model.native_weights_loaded or model.training_steps < 1 or model.cfg.lora_rank < 1:
        raise ValueError("checkpoint is not a trained native LoRA checkpoint")
    if point["training_state"]["step"] < 1:
        raise ValueError("checkpoint has no training exposure")
    request = SystemOneRequest(
        state="A short real-weight reference state.",
        questions={"q": {"type": "noul", "instructions": "Is the state short?"}},
        model="kev-laya-preview")
    encoded = encode_request(request, tokenizer, Limits())
    ids = encoded.state + encoded.branches[0].ids
    if not 1 <= len(ids) <= 256:
        raise ValueError("probe must stay on the short attention path")

    fields = asdict(model.cfg)
    keep = {key: value for key, value in fields.items() if key in {
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
        "rope_theta", "rms_norm_eps"}}
    config = Qwen2Config(**keep, hidden_act="silu", use_sliding_window=False,
                         attention_dropout=0.0)
    config._attn_implementation = "eager"
    oracle = Qwen2Model(config).to("cuda:0")
    oracle.load_state_dict(merged_backbone(model, merge_lora=False), strict=True)
    oracle.requires_grad_(False)
    native_adapters = {}
    reference_adapters = {}
    for index, (native_layer, reference_layer) in enumerate(
            zip(model.backbone.layers, oracle.layers, strict=True)):
        for projection in ("q_proj", "v_proj"):
            native = getattr(native_layer.self_attn, projection)
            if not isinstance(native, LoRALinear):
                raise ValueError("expected native LoRA at every Q/V projection")
            name = f"layers.{index}.self_attn.{projection}"
            reference = TrainableReferenceLoRA(
                getattr(reference_layer.self_attn, projection), native)
            setattr(reference_layer.self_attn, projection, reference)
            native_adapters[name] = native
            reference_adapters[name] = reference
    if not native_adapters or not any(bool(torch.count_nonzero(a.b)) for a in native_adapters.values()):
        raise ValueError("no trained adapter B weights found")

    model.train()
    oracle.eval()
    model.zero_grad(set_to_none=True)
    oracle.zero_grad(set_to_none=True)
    native_hidden, _ = model.backbone(ids)
    reference_hidden = oracle(
        input_ids=torch.tensor(ids, device="cuda:0")[None],
        use_cache=False).last_hidden_state[0]
    expected_shape = (len(ids), model.cfg.hidden_size)
    if native_hidden.shape != expected_shape or reference_hidden.shape != expected_shape:
        raise ValueError("incomplete hidden-state tensors")
    hidden_result = compare(native_hidden, reference_hidden,
                            PROTOCOL["hf_hidden_atol"], PROTOCOL["hf_hidden_rtol"])
    # A nonconstant deterministic VJP excites positions and all hidden channels.
    row = torch.arange(len(ids), device="cuda:0", dtype=torch.float32)[:, None]
    column = torch.arange(model.cfg.hidden_size, device="cuda:0", dtype=torch.float32)[None, :]
    upstream = (torch.sin(row * 0.017 + column * 0.013) +
                torch.cos(row * 0.031 - column * 0.007)) / math.sqrt(len(ids) * model.cfg.hidden_size)
    if not bool(torch.isfinite(upstream).all()):
        raise ValueError("nonfinite upstream")
    torch.autograd.backward(native_hidden, upstream)
    torch.autograd.backward(reference_hidden, upstream)
    gradient_results = {}
    for name in sorted(native_adapters):
        native, reference = native_adapters[name], reference_adapters[name]
        for suffix in ("a", "b"):
            native_gradient = getattr(native, suffix).grad
            reference_gradient = getattr(reference, suffix).grad
            if native_gradient is None or reference_gradient is None:
                raise ValueError(f"missing adapter gradient: {name}.{suffix}")
            gradient_results[f"{name}.{suffix}"] = compare(
                native_gradient, reference_gradient,
                PROTOCOL["gradient_atol"], PROTOCOL["gradient_rtol"])
    expected_count = 4 * model.cfg.num_hidden_layers
    if len(gradient_results) != expected_count:
        raise ValueError("incomplete adapter gradient set")
    free_after, _ = torch.cuda.mem_get_info(0)
    receipt = {
        "evidence_class": "short-actual-weight-fp32-hf-backbone-adapter-gradient-oracle",
        "source_commit": source_commit,
        "source_worktree_dirty": bool(__import__("subprocess").check_output(
            ["git", "status", "--porcelain"], cwd=SOURCE, text=True).strip()),
        "model_sha256": sha256(SOURCE / "src/kev_laya/model.py"),
        "native_gradients_sha256": sha256(SOURCE / "src/kev_laya/native_gradients.py"),
        "script_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "transformers_version": version,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "cuda_allocator_fraction": ALLOCATOR_FRACTION,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_total_memory_bytes": total_memory,
        "cuda_free_before_bytes": free_before,
        "cuda_free_after_bytes": free_after,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "precision": "fp32",
        "attention_implementation": "transformers-Qwen2Model-eager",
        "adapter_mapping": "unmerged-Q/V-two-linear-LoRA",
        "training_steps": model.training_steps,
        "tokens": len(ids),
        "hidden": hidden_result,
        "gradient_count": len(gradient_results),
        "gradients": gradient_results,
        "worst_gradient_name": max(gradient_results, key=lambda key: gradient_results[key]["max_tolerance_ratio"]),
        "passed": hidden_result["passed"] and all(item["passed"] for item in gradient_results.values()),
        "interpretation": "one short actual-weight FP32 backbone VJP; not long-context, exact-limit, or task-quality evidence",
    }
    if source_fingerprint() != source_identity:
        raise ValueError("source changed during oracle")
    receipt["source"] = source_identity
    with OUTPUT.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(OUTPUT), "passed": receipt["passed"],
                      "gradient_count": receipt["gradient_count"],
                      "worst_gradient_name": receipt["worst_gradient_name"]}, allow_nan=False))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if OUTPUT is not None and not OUTPUT.exists():
            failure = OUTPUT.with_suffix(OUTPUT.suffix + ".failure.json")
            if not failure.exists():
                with failure.open("x", encoding="utf-8") as stream:
                    json.dump({"passed": False, "stage": "oracle", "error_type": type(exc).__name__,
                               "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256}, stream)
                    stream.write("\n")
        raise
