"""Opt-in #2/#3 real-weight CUDA smoke; not trained-context acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from kev_laya import model as model_module
from kev_laya import native_validation as native_validation_module
from kev_laya.backbone_provenance import verify_source
from kev_laya.checkpoint import load_checkpoint
from kev_laya.encoding import Limits, encode_request, preprocessing_identity
from kev_laya.native_validation import hf_parity, measurement_device, require_reference_version
from kev_laya.schema import SystemOneRequest


def run(source: Path, checkpoint: Path, device_name: str) -> dict:
    device = measurement_device(device_name, "fp32")
    require_reference_version()
    source_receipt = verify_source(source)
    if source_receipt["status"] != "verified":
        raise ValueError("pinned backbone source gate did not verify")
    model, tokenizer, point = load_checkpoint(checkpoint, device)
    if not model.native_weights_loaded or model.loaded_backbone_sha256 != source_receipt["weight_anchor"]["sha256"]:
        raise ValueError("checkpoint does not contain the pinned loaded backbone")
    if point["preprocessing"] != preprocessing_identity(tokenizer):
        raise ValueError("checkpoint preprocessing identity mismatch")
    request = SystemOneRequest(
        state="A short real-weight reference state.",
        questions={"q": {"type": "noul", "instructions": "Is the state short?"}},
        model="kev-laya-preview",
    )
    encoded = encode_request(request, tokenizer, Limits())
    ids = encoded.state + encoded.branches[0].ids
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    parity = hf_parity(model, ids)
    torch.cuda.synchronize(device)
    parity_seconds = time.perf_counter() - start
    with torch.inference_mode():
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits, usage = model(encoded)
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - start
    finite = all(torch.isfinite(z).all().item() for z in logits)
    return {
        "gate": "short-native-real-weight-v1",
        "passed": bool(parity["passed"] and finite and len(logits) == 1 and tuple(logits[0].shape) == (2,)),
        "model_source_sha256": hashlib.sha256(Path(model_module.__file__).read_bytes()).hexdigest(),
        "oracle_source_sha256": hashlib.sha256(Path(native_validation_module.__file__).read_bytes()).hexdigest(),
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_sha256": source_receipt["source_manifest_sha256"],
        "checkpoint_sha256": point["checkpoint_sha256"],
        "weight_sha256": model.loaded_backbone_sha256,
        "tokenizer_sha256": tokenizer.identity.removeprefix("qwen-tokenizer-sha256:"),
        "preprocessing": point["preprocessing"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "transformers": require_reference_version(),
        "tokens": len(ids),
        "hf_parity": parity,
        "hf_parity_seconds": parity_seconds,
        "forward_seconds": forward_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "logit_shapes": [list(z.shape) for z in logits],
        "logits_finite": finite,
        "usage": usage,
        "training_steps": model.training_steps,
        "calibration_status": model.calibration_provenance.get("status"),
        "training_modes": "not_tested",
        "serving": "not_tested",
        "trained_context": "not_tested",
        "representative_quality": "unmeasured",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("output file already exists")
    if not args.out.parent.is_dir():
        parser.error("output parent directory must exist")
    report = run(args.source, args.checkpoint, args.device)
    # Exclusive creation preserves the first result, including a failed parity result.
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.out)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
