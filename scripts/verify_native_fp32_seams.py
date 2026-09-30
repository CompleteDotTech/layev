"""Opt-in actual-weight FP32 short-attention dispatch seam diagnostic."""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[1]
ARTIFACTS = CHECKPOINT = None
CHECKPOINT_SHA = None
OWNS_OUTPUT_DIRECTORY = False
OUTPUT_STEM = "fp32-seam"
sys.path.insert(0, str(SOURCE / "src"))

from kev_laya.checkpoint import load_checkpoint  # noqa: E402
from kev_laya.encoding import Limits, encode_request  # noqa: E402
from kev_laya.io import atomic_json, sha256_file  # noqa: E402
from kev_laya.native_validation import PROTOCOL, require_reference_version  # noqa: E402
from kev_laya.objectives import ObjectiveConfig, objective  # noqa: E402
from kev_laya.schema import SystemOneRequest  # noqa: E402


def compare(left, right, *, atol, rtol):
    if len(left) != len(right) or not left:
        raise ValueError("incomplete comparison")
    maximum = ratio = 0.0
    for a, b in zip(left, right, strict=True):
        if a.shape != b.shape or a.dtype != b.dtype or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("nonfinite, dtype, or shape-mismatched comparison")
        delta = (a - b).abs()
        threshold = atol + rtol * b.abs()
        maximum = max(maximum, float(delta.max()))
        ratio = max(ratio, float((delta / threshold).max()))
    return {"max_absolute": maximum, "max_tolerance_ratio": ratio, "passed": ratio <= 1}


def request_with_filler(n, suffix_filler=0):
    return SystemOneRequest(
        state={"route": "alpha", "urgency": 2, "active": True, "context": "x " * n},
        questions={
            "choice": {"type": "choice", "instructions": "Choose the listed route using the state." + " x" * suffix_filler,
                       "criteria": {"alpha": "A route", "beta": "B route", "gamma": "C route"}},
            "noul": {"type": "noul", "instructions": "Is the state explicitly marked active?" + " x" * suffix_filler},
        }, model="kev-laya-preview")


def exact_encoding(tokenizer, desired, suffix_filler=0):
    limits = Limits(branch=32768, aggregate=65536)
    low, high = 0, 20000
    while low < high:
        middle = (low + high) // 2
        size = len(encode_request(request_with_filler(middle, suffix_filler), tokenizer, limits).state)
        if size < desired:
            low = middle + 1
        else:
            high = middle
    for n in range(max(0, low - 32), min(20000, low + 32) + 1):
        encoded = encode_request(request_with_filler(n, suffix_filler), tokenizer, limits)
        if len(encoded.state) == desired:
            if len(encoded.branches) != 2:
                raise ValueError("two complete branches required")
            return encoded, n
    raise ValueError(f"exact {desired}-token state unavailable with frozen filler")


def largest_short_full_case(tokenizer):
    # The serialized question suffix is not empty. A 254/255-token state
    # cannot yield a <=256 full row unless both suffixes fit in 1-2 tokens.
    # Choose the highest representable state whose two actual rows are short.
    sample, _ = exact_encoding(tokenizer, 200)
    upper = min(255, 256 - max(len(branch.ids) for branch in sample.branches))
    for desired in range(upper, 0, -1):
        try:
            encoded, count = exact_encoding(tokenizer, desired)
        except ValueError:
            continue
        lengths = [desired + len(branch.ids) for branch in encoded.branches]
        if max(lengths) <= 256:
            return encoded, count
    raise ValueError("no representable two-branch short full-row case")


def long_suffix_case(tokenizer, desired=255):
    low, high = 0, 12000
    while low < high:
        middle = (low + high) // 2
        encoded, _ = exact_encoding(tokenizer, desired, middle)
        if min(desired + len(branch.ids) for branch in encoded.branches) <= 8192:
            low = middle + 1
        else:
            high = middle
    for count in range(max(0, low - 8), min(12000, low + 8) + 1):
        encoded, state_filler = exact_encoding(tokenizer, desired, count)
        lengths = [desired + len(branch.ids) for branch in encoded.branches]
        if min(lengths) > 8192:
            return encoded, state_filler, count
    raise ValueError("cannot construct bounded >8192 full rows")


def run_case(model, encoded, filler_count, source_identity, free_before, total_memory,
             *, case_name, suffix_filler):
    state_tokens = len(encoded.state)
    receipt_path = ARTIFACTS / f"{OUTPUT_STEM}-{case_name}.json"
    progress_path = ARTIFACTS / f"{OUTPUT_STEM}-{case_name}.progress.json"
    if receipt_path.exists() or progress_path.exists():
        raise FileExistsError(f"existing seam artifacts for {case_name}")
    def progress(stage, **extra):
        atomic_json(progress_path, {"case_name": case_name, "state_tokens": state_tokens, "stage": stage,
                                    "source": source_identity, **extra})
        print({"case_name": case_name, "state_tokens": state_tokens, "stage": stage, **extra}, flush=True)
    kinds = [branch.question.type for branch in encoded.branches]
    targets = []
    for branch in encoded.branches:
        n = len(branch.option_ends)
        target = [1 / n] * n
        target[0] += 0.1
        target[-1] -= 0.1
        targets.append(target)
    results = {}
    progress("encoded", filler_count=filler_count, suffix_filler=suffix_filler,
             branch_tokens=[state_tokens + len(b.ids) for b in encoded.branches])
    try:
        for mode, kwargs in (("batched", {}), ("serial", {"serial_reference": True}),
                             ("full", {"reference": True})):
            progress("mode_running", mode=mode)
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            logits, usage = model(encoded, **kwargs)
            loss = objective(logits, targets, kinds, ObjectiveConfig(ordinal=0.1))[0]
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"nonfinite {mode} loss")
            logit_gradients = torch.autograd.grad(loss, logits)
            for index, (branch_logits, branch_gradient) in enumerate(
                    zip(logits, logit_gradients, strict=True)):
                branch_logits.backward(branch_gradient, retain_graph=index < len(logits) - 1)
            torch.cuda.synchronize(0)
            expected = {name for name, p in model.named_parameters() if p.requires_grad}
            gradients = {name: p.grad.detach().float().cpu().clone()
                         for name, p in model.named_parameters() if p.requires_grad and p.grad is not None}
            if (set(gradients) != expected or len(gradients) != 100 or
                    not all(bool(torch.isfinite(g).all()) for g in gradients.values())):
                raise ValueError(f"missing/nonfinite {mode} gradients")
            results[mode] = {"logits": [x.detach().float().cpu() for x in logits],
                             "gradients": gradients, "loss": float(loss.detach()),
                             "usage": usage, "elapsed_seconds": time.perf_counter() - start,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
            progress("mode_complete", mode=mode,
                     elapsed_seconds=results[mode]["elapsed_seconds"],
                     peak_allocated_bytes=results[mode]["peak_allocated_bytes"])
        comparisons = {}
        for first, second in (("batched", "serial"), ("serial", "full"), ("batched", "full")):
            a, b = results[first], results[second]
            gradients = {name: compare([a["gradients"][name]], [b["gradients"][name]],
                                       atol=PROTOCOL["gradient_atol"], rtol=PROTOCOL["gradient_rtol"])
                         for name in sorted(a["gradients"])}
            worst = max(gradients, key=lambda name: gradients[name]["max_tolerance_ratio"])
            logits = compare(a["logits"], b["logits"],
                             atol=PROTOCOL["fp32_logit_atol"], rtol=PROTOCOL["fp32_logit_rtol"])
            probabilities = compare([x.softmax(-1) for x in a["logits"]],
                                    [x.softmax(-1) for x in b["logits"]],
                                    atol=PROTOCOL["probability_atol"], rtol=PROTOCOL["probability_rtol"])
            comparisons[f"{first}_{second}"] = {
                "logits": logits, "probabilities": probabilities,
                "gradient_count": len(gradients),
                "failed_gradient_count": sum(not row["passed"] for row in gradients.values()),
                "failed_gradients": {name: row for name, row in gradients.items() if not row["passed"]},
                "worst_gradient_name": worst, "worst_gradient": gradients[worst],
                "passed": logits["passed"] and probabilities["passed"] and
                          all(row["passed"] for row in gradients.values())}
        current_identity = source_fingerprint()
        if current_identity != source_identity:
            raise ValueError("source changed during boundary case")
        free_after, _ = torch.cuda.mem_get_info(0)
        receipt = {"evidence_class": "actual-weight-fp32-short-attention-dispatch-seam-parity",
                   "source": source_identity, "script_sha256": sha256_file(Path(__file__)),
                   "checkpoint_sha256": CHECKPOINT_SHA, "torch_version": torch.__version__,
                   "device": torch.cuda.get_device_name(0), "precision": "fp32",
                   "cuda_allocator_fraction": 0.70,
                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "cuda_total_memory_bytes": total_memory, "cuda_free_before_bytes": free_before,
                   "cuda_free_after_bytes": free_after,
                   "case_name": case_name, "state_tokens": state_tokens,
                   "filler_count": filler_count, "suffix_filler": suffix_filler,
                   "logical_tokens": encoded.logical_tokens,
                   "branch_tokens": [state_tokens + len(b.ids) for b in encoded.branches],
                   "training_steps": model.training_steps,
                   "runs": {name: {key: value for key, value in row.items()
                                   if key not in ("logits", "gradients")}
                            for name, row in results.items()},
                   "comparisons": comparisons,
                   "passed": all(row["passed"] for row in comparisons.values()),
                   "interpretation": "one synthetic attention-dispatch seam case; not exact-limit or quality evidence"}
        json.dumps(receipt, allow_nan=False)
        atomic_json(receipt_path, receipt)
        progress("complete", receipt=str(receipt_path), receipt_sha256=sha256_file(receipt_path),
                 passed=receipt["passed"])
        return receipt
    except BaseException as exc:
        progress("failed", error_type=type(exc).__name__, error=str(exc))
        failure_path = ARTIFACTS / f"{OUTPUT_STEM}-{case_name}.failure.json"
        if failure_path.exists():
            raise FileExistsError(failure_path) from exc
        atomic_json(failure_path, {
            "evidence_class": "actual-weight-fp32-short-attention-seam-probe-failure",
            "source": source_identity, "script_sha256": sha256_file(Path(__file__)),
            "checkpoint_sha256": CHECKPOINT_SHA, "case_name": case_name,
            "state_tokens": state_tokens,
            "stage": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "completed_modes": sorted(results), "passed": False})
        return {"state_tokens": state_tokens, "passed": False,
                "failure_receipt": str(failure_path)}


def source_fingerprint():
    import subprocess
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip(),
            "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True).strip()),
            "files": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                      for name in ("src/kev_laya/model.py", "src/kev_laya/native_gradients.py",
                                   "src/kev_laya/training.py", "src/kev_laya/execution.py")}}


def main():
    global CHECKPOINT, CHECKPOINT_SHA, ARTIFACTS, OWNS_OUTPUT_DIRECTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=(
        "short_full", "short_medium_255", "short_medium_256", "medium_257", "short_long_255", "boundary_8191", "boundary_8192", "long_10520"),
        default=["short_full", "short_medium_255", "short_medium_256", "medium_257"])
    args = parser.parse_args()
    CHECKPOINT, CHECKPOINT_SHA, ARTIFACTS = args.checkpoint, args.expected_checkpoint_sha256, args.out_dir
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    OWNS_OUTPUT_DIRECTORY = True
    require_reference_version()
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("checkpoint identity changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    source_identity = source_fingerprint()
    torch.cuda.set_per_process_memory_fraction(0.70, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or
            torch.get_float32_matmul_precision() != "highest"):
        raise RuntimeError("TF32 settings not applied")
    free_before, total_memory = torch.cuda.mem_get_info(0)
    model, tokenizer, point = load_checkpoint(CHECKPOINT, "cuda:0")
    if (not model.native_weights_loaded or model.training_steps < 1 or model.cfg.lora_rank < 1
            or point["training_state"]["step"] < 1):
        raise ValueError("checkpoint is not a trained native model")
    model.train()
    outcomes = []
    for case_name in args.cases:
        state_tokens = {"short_full": None, "short_medium_255": 255,
                        "short_medium_256": 256, "medium_257": 257,
                        "short_long_255": 255, "boundary_8191": 8191, "boundary_8192": 8192,
                        "long_10520": 10520}[case_name]
        try:
            if case_name == "short_full":
                encoded, filler_count = largest_short_full_case(tokenizer)
                suffix_filler = 0
            elif case_name == "short_long_255":
                encoded, filler_count, suffix_filler = long_suffix_case(tokenizer)
            else:
                encoded, filler_count = exact_encoding(tokenizer, state_tokens)
                suffix_filler = 0
            actual_state_tokens = len(encoded.state)
            full_lengths = [actual_state_tokens + len(branch.ids) for branch in encoded.branches]
            if case_name == "short_full" and max(full_lengths) > 256:
                raise ValueError("short full-row dispatch not reached")
            if case_name.startswith("short_medium") and (actual_state_tokens > 256 or
                                                           min(full_lengths) <= 256 or
                                                           max(full_lengths) >= 8192):
                raise ValueError("short-to-medium dispatch not reached")
            if case_name == "medium_257" and actual_state_tokens != 257:
                raise ValueError("medium prefix dispatch not reached")
            if case_name in ("boundary_8191", "boundary_8192", "long_10520") and min(full_lengths) < 8192:
                raise ValueError("long/boundary dispatch not reached")
            if case_name == "short_long_255" and min(full_lengths) <= 8192:
                raise ValueError("short-to-long dispatch not reached")
        except BaseException as exc:
            failure_path = ARTIFACTS / f"{OUTPUT_STEM}-{case_name}.failure.json"
            if failure_path.exists():
                raise FileExistsError(failure_path) from exc
            atomic_json(failure_path, {
                "evidence_class": "actual-weight-fp32-short-attention-seam-probe-failure",
                "source": source_identity, "script_sha256": sha256_file(Path(__file__)),
                "checkpoint_sha256": CHECKPOINT_SHA, "case_name": case_name,
                "requested_state_tokens": state_tokens,
                "stage": "encoding", "error_type": type(exc).__name__,
                "error": str(exc), "passed": False})
            outcomes.append({"case_name": case_name, "passed": False,
                             "failure_receipt": str(failure_path)})
            continue
        if source_fingerprint() != source_identity:
            raise ValueError("source changed before seam case")
        outcomes.append(run_case(model, encoded, filler_count, source_identity,
                                 free_before, total_memory, case_name=case_name,
                                 suffix_filler=suffix_filler))
    print({"seam_summary": [(row.get("case_name"), row["passed"]) for row in outcomes]},
          flush=True)
    if not all(row["passed"] for row in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if OWNS_OUTPUT_DIRECTORY:
            failure = ARTIFACTS / "preflight.failure.json"
            if not failure.exists():
                with failure.open("x", encoding="utf-8") as stream:
                    json.dump({"passed": False, "stage": "preflight", "error_type": type(exc).__name__,
                               "checkpoint_sha256": CHECKPOINT_SHA}, stream)
                    stream.write("\n")
        raise
