"""Score the preserved case=99999 on a pinned checkpoint, outside held-out quality.

This is a previously observed synthetic regression. Its result cannot establish
representative quality and must not select a checkpoint or fit calibration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from kev_laya.checkpoint import load_checkpoint
from kev_laya.cli import limits_for
from kev_laya import counterfactual as regression_module
from kev_laya.counterfactual import regression_data
from kev_laya.evaluation import evaluate
from kev_laya.io import sha256_file


CASE = "color=red; level=1; case=99999"
KINDS = ("choice", "noul", "score")
ORDERS = ("red-first", "blue-first")
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_REVISION = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def safe_digest(value) -> str | None:
    return value if isinstance(value, str) and SHA256.fullmatch(value) else None


def checkpoint_declarations(point: dict) -> dict:
    """Copy only bounded, content-free identifiers; this does not verify training."""
    provenance = point.get("provenance")
    if not isinstance(provenance, dict):
        return {}
    result = {}
    seed = provenance.get("seed")
    if type(seed) is int and 0 <= seed < 2**64:
        result["seed"] = seed
    for key in ("config_sha256", "data_sha256"):
        value = provenance.get(key)
        if safe_digest(value) is not None:
            result[key] = value
    commit = provenance.get("commit")
    if isinstance(commit, str) and GIT_REVISION.fullmatch(commit):
        result["commit"] = commit
    hashes = provenance.get("split_hashes")
    if isinstance(hashes, dict) and set(hashes) == {"train", "development", "calibration", "test"}:
        if all(safe_digest(value) is not None for value in hashes.values()):
            result["split_hashes"] = {name: hashes[name] for name in sorted(hashes)}
    return result


def run(checkpoint: Path, output: Path, *, expected_sha256: str,
        device: str = "cpu") -> dict:
    checkpoint, output = Path(checkpoint), Path(output)
    if not isinstance(expected_sha256, str) or SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected_checkpoint_sha256_required")
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("checkpoint_must_be_regular_file")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    model, tokenizer, point = load_checkpoint(checkpoint, device,
                                               expected_sha256=expected_sha256)
    if type(point.get("native_weights_loaded")) is not bool:
        raise ValueError("checkpoint_native_flag_invalid")
    if type(point.get("training_steps")) is not int or point["training_steps"] < 1:
        raise ValueError("checkpoint_declares_no_training_steps")
    selected = [datum for datum in regression_data() if datum.request.state == CASE]
    if len(selected) != 2 or {datum.meta.get("option_order") for datum in selected} != set(ORDERS):
        raise ValueError("preserved_regression_case_incomplete")
    report = evaluate(model, selected, tokenizer, limits_for(point),
                      split="observed-regression")
    rows = report["rows"]
    if (len(rows) != 6 or
            {(row["type"], row["option_order"]) for row in rows} !=
            {(kind, order) for kind in KINDS for order in ORDERS} or
            any(row["group"] != "known-regression/99999" or
                row["correct"] not in (0.0, 1.0) for row in rows)):
        raise ValueError("preserved_regression_readout_incomplete")
    if sha256_file(checkpoint) != expected_sha256:
        raise ValueError("checkpoint_changed_during_regression_readout")
    by_type = {kind: {"correct": int(sum(row["correct"] for row in rows
                                          if row["type"] == kind)), "total": 2}
               for kind in KINDS}
    by_order = {order: {"correct": int(sum(row["correct"] for row in rows
                                            if row["option_order"] == order)), "total": 3}
                for order in ORDERS}
    correct = sum(item["correct"] for item in by_type.values())
    result = {
        "schema_version": "layev-observed-regression/1",
        "status": "observed_regression_not_heldout",
        "scope": "previously observed synthetic case; not representative quality or checkpoint selection",
        "state": CASE,
        "regression_source_sha256": sha256_file(Path(regression_module.__file__)),
        "checkpoint_sha256": expected_sha256,
        "model_id": point["model_id"],
        "checkpoint_training_steps_declared": point["training_steps"],
        "checkpoint_native_weights_loaded_declared": (
            point["native_weights_loaded"] if type(point["native_weights_loaded"]) is bool else None),
        "training_exposure_status": "not_verified_by_this_report",
        "tokenizer_identity": tokenizer.identity,
        "preprocessing": point.get("preprocessing"),
        "parent_sha256": safe_digest(point.get("parent_sha256")),
        "loaded_backbone_sha256": safe_digest(point.get("loaded_backbone_sha256")),
        "checkpoint_declarations": checkpoint_declarations(point),
        "correct": correct,
        "total": 6,
        "all_six_correct": correct == 6,
        "by_type": by_type,
        "by_option_order": by_order,
        "evaluation": report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = run(args.checkpoint, args.out, expected_sha256=args.expected_sha256,
                 device=args.device)
    print(json.dumps({"status": result["status"], "correct": result["correct"],
                      "total": result["total"], "all_six_correct": result["all_six_correct"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
