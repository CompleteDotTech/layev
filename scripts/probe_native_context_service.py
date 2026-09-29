"""Probe fixed native context limits through the local authenticated service.

This generated diagnostic measures request handling, not decision quality.
It never sends a request to a remote service or writes model weights.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from kev_laya.checkpoint import load_checkpoint
from kev_laya.encoding import Limits, encode_request
from kev_laya.io import atomic_json, sha256_file
from kev_laya.native_validation import make_case
from kev_laya.schema import SystemOneRequest, SystemOneResponse
from kev_laya.service import InferenceRuntime, ServeSettings, create_app


def measured_shape(data: dict, tokenizer) -> dict:
    encoded = encode_request(SystemOneRequest.model_validate(data), tokenizer,
                             Limits(262144, 524288, 1024))
    return {"state_tokens": len(encoded.state),
            "max_branch_tokens": max(len(encoded.state) + len(branch.ids) for branch in encoded.branches),
            "logical_tokens": encoded.logical_tokens,
            "questions": len(encoded.branches)}


def one_token_over(exact: dict, tokenizer, question_id: str) -> tuple[dict, dict]:
    for suffix in ("a", "x", "0", ".", "!", "?", " x", " a", " z", "\n", " q", " zz"):
        candidate = copy.deepcopy(exact)
        candidate["questions"][question_id]["instructions"] += suffix
        shape = measured_shape(candidate, tokenizer)
        if shape["logical_tokens"] == 65537:
            return candidate, shape
    raise ValueError(f"no measured one-token suffix for {question_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-process-fraction", type=float, default=0.70)
    parser.add_argument("--position", choices=("beginning", "middle", "end"), default="beginning")
    parser.add_argument("--language", choices=("en", "es"), default="en")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.gpu_process_fraction <= 1:
        raise ValueError("GPU process fraction must be in (0, 1]")
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint hash mismatch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("native context probe requires CUDA")
    torch.cuda.set_per_process_memory_fraction(args.gpu_process_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    exact = make_case(tokenizer, args.position, args.language).model_dump()
    exact_shape = measured_shape(exact, tokenizer)
    if exact_shape["max_branch_tokens"] != 32768 or exact_shape["logical_tokens"] != 65536:
        raise ValueError("generated request no longer sits on both boundaries")
    aggregate_over, aggregate_shape = one_token_over(
        exact, tokenizer, next(reversed(exact["questions"])))
    if aggregate_shape["max_branch_tokens"] != 32768:
        raise ValueError("aggregate overflow also changed maximum branch")
    branch_over, branch_shape = one_token_over(exact, tokenizer, "route")
    if branch_shape["max_branch_tokens"] != 32769:
        raise ValueError("branch request is not one token beyond its limit")

    runtime = InferenceRuntime(model, tokenizer, "kev-laya-native-context-probe", Limits())
    client = TestClient(create_app(runtime, ServeSettings(api_keys=("local-generated-probe",))))
    cases = []
    for name, data, shape in (("exact", exact, exact_shape),
                              ("aggregate_plus_one", aggregate_over, aggregate_shape),
                              ("branch_plus_one", branch_over, branch_shape)):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        response = client.post("/v1/systemone", json=data,
                               headers={"Authorization": "Bearer local-generated-probe"})
        torch.cuda.synchronize(device)
        row = {"name": name, "shape": shape,
               "request_sha256": hashlib.sha256(SystemOneRequest.model_validate(data).model_dump_json().encode()).hexdigest(),
               "serialized_bytes": len(json.dumps(data, ensure_ascii=False).encode()),
               "status_code": response.status_code, "elapsed_seconds": time.perf_counter() - started,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        if name == "exact":
            if response.status_code != 200:
                raise ValueError(f"exact request failed: {response.status_code} {response.text[:300]}")
            result = SystemOneResponse.model_validate(response.json())
            if result.usage.input_tokens != 65536 or len(result.answers) != shape["questions"]:
                raise ValueError("exact response lost questions or tokens")
            row.update({"answers": len(result.answers), "returned_input_tokens": result.usage.input_tokens,
                        "returned_forward_tokens": result.usage.forward_tokens,
                        "calibration_status": result.calibration_status})
        else:
            detail = response.json().get("detail")
            expected = "aggregate_input" if name == "aggregate_plus_one" else "state_plus_branch"
            if (response.status_code != 422 or not isinstance(detail, dict) or
                    detail.get("code") != "context_overflow" or detail.get("limit") != expected):
                raise ValueError(f"over-limit request was not rejected as {expected}: {response.status_code} {detail}")
            row["rejection"] = detail
        cases.append(row)
        print(f"{name}: HTTP {response.status_code}", flush=True)
    atomic_json(args.output, {"status": "passed_fixed_limit_native_probe",
                              "evidence_class": "generated_exact_context_mechanics_not_quality",
                              "checkpoint_sha256": args.checkpoint_sha256,
                              "position": args.position, "language": args.language,
                              "device": torch.cuda.get_device_name(device),
                              "torch_version": torch.__version__,
                              "gpu_process_memory_fraction": args.gpu_process_fraction,
                              "cases": cases})
    print(sha256_file(args.output), flush=True)


if __name__ == "__main__":
    main()
