"""Read-only preparation gate for a representative-quality experiment.

Validates declarations and frozen bytes, not the truth of a data review, signing,
training chronology, native execution, measured quality, or permission to spend.
No tokenizer construction, model call, remote request, or source write occurs.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path, PureWindowsPath
import re
from typing import Any

from .schema import strict_loads

SCHEMA = "layev-quality-protocol/1"
MAX_DOCUMENT_BYTES = 1024 * 1024
DEFAULT_MAX_SUITE_BYTES = 256 * 1024 * 1024
PARTITIONS = ("train", "development", "calibration", "test")
METRICS = frozenset({"accuracy", "nll", "brier", "ece", "ordinal_error", "risk_coverage"})
SLICES = frozenset({"domain", "language", "question_type", "context_length", "option_order"})
VARIATIONS = frozenset({"option_order", "question_ids", "colors", "levels", "wording",
                        "domains", "languages", "context_lengths"})
REGRESSION_STATE = "color=red; level=1; case=99999"
KNOWN_FIXTURE_IDS = frozenset({"smoke-v1", "counterfactual-markers-v1"})


class ProtocolError(ValueError):
    """A stable, content-free rejection reason suitable for a receipt."""


class PrerequisiteMissing(ProtocolError):
    """Missing local input; CLI exit 2, never a passing skip."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ProtocolError(reason)


def _object(value: Any, keys: set[str], reason: str) -> dict:
    _require(isinstance(value, dict) and set(value) == keys, reason)
    return value


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _hash(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def _integer(value: Any, minimum: int = 1) -> bool:
    return type(value) is int and value >= minimum


def _positive(value: Any) -> bool:
    return ((type(value) is int and value > 0)
            or (type(value) is float and math.isfinite(value) and value > 0))


def _set(value: Any, required: frozenset[str], reason: str) -> None:
    _require(isinstance(value, list) and all(isinstance(item, str) for item in value), reason)
    _require(len(value) == len(set(value)) and set(value) == required, reason)


def validate_protocol(protocol: dict, review: dict) -> dict:
    """Check a complete preregistration, without executing it or trusting claims."""
    p = _object(protocol, {"schema_version", "protocol_id", "repository_revision",
        "suite_manifest_sha256", "data_review_sha256", "seeds", "training", "budget",
        "selection", "calibration", "baselines", "metrics", "slices", "variations",
        "uncertainty", "fixture_accuracy_threshold", "regression"}, "protocol_fields_invalid")
    _require(p["schema_version"] == SCHEMA and _text(p["protocol_id"]), "protocol_identity_invalid")
    _require(_hash(p["repository_revision"], 40), "repository_revision_not_pinned")
    for field in ("suite_manifest_sha256", "data_review_sha256"):
        _require(_hash(p[field]), field + "_invalid")
    seeds = p["seeds"]
    # The trainer passes these values to torch.manual_seed (unsigned 64-bit range).
    _require(isinstance(seeds, list) and len(seeds) >= 3
             and all(_integer(seed, 0) and seed < 2**64 for seed in seeds),
             "three_distinct_seeds_required")
    _require(len(seeds) == len(set(seeds)), "three_distinct_seeds_required")
    training = _object(p["training"], {"parent_checkpoint_sha256", "arms"}, "training_fields_invalid")
    _require(_hash(training["parent_checkpoint_sha256"]), "parent_checkpoint_not_pinned")
    arms = _object(training["arms"], {"supervised", "reward"}, "paired_arms_required")
    for arm in arms.values():
        _object(arm, {"optimizer_steps", "useful_token_budget"}, "arm_fields_invalid")
        _require(_integer(arm["optimizer_steps"]) and _integer(arm["useful_token_budget"]),
                 "arm_budget_invalid")
    _require(arms["supervised"] == arms["reward"], "unmatched_control_budget")
    total_steps = len(seeds) * sum(arm["optimizer_steps"] for arm in arms.values())
    budget = _object(p["budget"], {"max_total_optimizer_steps", "max_wall_seconds",
                                  "max_peak_gpu_memory_bytes", "paid_usd_limit"}, "budget_fields_invalid")
    _require(_integer(budget["max_total_optimizer_steps"])
             and budget["max_total_optimizer_steps"] >= total_steps, "total_step_budget_exceeded")
    _require(_positive(budget["max_wall_seconds"])
             and _integer(budget["max_peak_gpu_memory_bytes"]), "resource_budget_invalid")
    # This local preparation gate does not authorize or implement paid execution.
    _require(type(budget["paid_usd_limit"]) in (int, float)
             and budget["paid_usd_limit"] == 0, "paid_execution_not_authorized_by_preflight")
    selection = _object(p["selection"], {"checkpoint_partition", "test_partition",
        "test_evaluations_per_arm_seed", "rule"}, "selection_fields_invalid")
    _require(selection["checkpoint_partition"] == "development"
             and selection["test_partition"] == "test"
             and type(selection["test_evaluations_per_arm_seed"]) is int
             and selection["test_evaluations_per_arm_seed"] == 1
             and _text(selection["rule"]), "heldout_selection_policy_invalid")
    calibration = _object(p["calibration"], {"fit_partition", "report"}, "calibration_fields_invalid")
    _require(calibration["fit_partition"] == "calibration", "calibration_partition_invalid")
    _set(calibration["report"], frozenset({"raw", "calibrated"}), "calibration_readouts_invalid")
    baselines = _object(p["baselines"], {"Kev", "Laya"}, "pinned_baselines_required")
    for baseline in baselines.values():
        _object(baseline, {"source_revision", "artifact_sha256"}, "baseline_fields_invalid")
        _require(_hash(baseline["source_revision"], 40) and _hash(baseline["artifact_sha256"]),
                 "baseline_not_pinned")
    for name, required in (("metrics", METRICS), ("slices", SLICES), ("variations", VARIATIONS)):
        _set(p[name], required, name + "_incomplete_or_invalid")
    uncertainty = _object(p["uncertainty"], {"resampling_unit", "bootstrap_replicates", "confidence"},
                          "uncertainty_fields_invalid")
    _require(uncertainty["resampling_unit"] == "source_group"
             and _integer(uncertainty["bootstrap_replicates"], 1000)
             and type(uncertainty["confidence"]) in (int, float)
             and uncertainty["confidence"] == 0.95, "uncertainty_policy_invalid")
    _require(type(p["fixture_accuracy_threshold"]) in (int, float)
             and p["fixture_accuracy_threshold"] == 0.70, "fixture_threshold_must_remain_070")
    regression = _object(p["regression"], {"state", "required_types", "failure_policy"},
                         "regression_fields_invalid")
    _require(regression["state"] == REGRESSION_STATE
             and regression["failure_policy"] == "retain_and_report", "known_regression_not_preserved")
    _set(regression["required_types"], frozenset({"choice", "noul", "score"}), "regression_types_missing")
    r = _object(review, {"schema_version", "review_id", "reviewer", "data_kind", "license_id",
        "license_reference", "intended_population", "sampling_method", "grouping_policy",
        "near_duplicate_review", "limitations", "approved_uses"}, "data_review_fields_invalid")
    _require(type(r["schema_version"]) is int and r["schema_version"] == 1, "data_review_version_invalid")
    _require(r["data_kind"] == "representative_candidate", "representative_review_required")
    for key in ("review_id", "reviewer", "license_id", "license_reference", "intended_population",
                "sampling_method", "grouping_policy", "near_duplicate_review"):
        _require(_text(r[key]), "data_review_declaration_missing")
    _require(isinstance(r["limitations"], list) and bool(r["limitations"])
             and all(_text(value) for value in r["limitations"]), "data_limitations_required")
    _set(r["approved_uses"], frozenset({"local_training", "local_evaluation"}), "local_data_uses_required")
    return {"declared_seeds": len(seeds), "declared_training_arms": len(arms),
            "declared_optimizer_steps": total_steps}


def _read(path: Path) -> bytes:
    try:
        _require(not path.is_symlink() and path.is_file(), "input_not_a_regular_local_file")
        with path.open("rb") as stream:
            raw = stream.read(MAX_DOCUMENT_BYTES + 1)
    except FileNotFoundError:
        raise PrerequisiteMissing("required_input_missing") from None
    except OSError:
        raise PrerequisiteMissing("required_input_unreadable") from None
    _require(len(raw) <= MAX_DOCUMENT_BYTES, "document_size_limit_exceeded")
    return raw


def _json(raw: bytes) -> dict:
    try:
        value = strict_loads(raw)
    except (ValueError, UnicodeError):
        raise ProtocolError("invalid_strict_json") from None
    _require(isinstance(value, dict), "document_must_be_an_object")
    return value


def preflight(protocol_path: Path, review_path: Path, suite_path: Path,
              *, max_suite_bytes: int = DEFAULT_MAX_SUITE_BYTES) -> dict:
    """Validate local snapshots and the existing real dataset loader, read-only.

    The bounded input check and final hash readback detect ordinary concurrent
    edits. Inputs must be immutable during validation; this is not an adversarial
    filesystem snapshot or an authenticated preregistration service.
    """
    _require(_integer(max_suite_bytes), "suite_byte_budget_invalid")
    protocol_path, review_path, suite_path = map(Path, (protocol_path, review_path, suite_path))
    for path in (protocol_path, review_path, suite_path / "manifest.json"):
        if not path.exists():
            raise PrerequisiteMissing("required_input_missing")
    protocol_raw, review_raw = _read(protocol_path), _read(review_path)
    manifest_path = suite_path / "manifest.json"
    manifest_raw = _read(manifest_path)
    protocol, review, manifest = map(_json, (protocol_raw, review_raw, manifest_raw))
    counts = validate_protocol(protocol, review)
    _require(hashlib.sha256(review_raw).hexdigest() == protocol["data_review_sha256"],
             "data_review_hash_mismatch")
    _require(hashlib.sha256(manifest_raw).hexdigest() == protocol["suite_manifest_sha256"],
             "suite_manifest_hash_mismatch")
    _require(_text(manifest.get("id")), "suite_identity_invalid")
    _require(manifest["id"] not in KNOWN_FIXTURE_IDS, "known_fixture_is_not_representative_evidence")
    _require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
             "suite_manifest_version_invalid")
    _require(_text(manifest.get("license")) and manifest["license"] == review["license_id"],
             "suite_license_declaration_mismatch")
    _require(_text(manifest.get("provenance")) and _text(manifest.get("grouping")),
             "suite_provenance_declarations_missing")
    partitions = _object(manifest.get("partitions"), set(PARTITIONS), "suite_partitions_invalid")
    paths, total_bytes = [], 0
    for split in PARTITIONS:
        info = _object(partitions[split], {"file", "sha256", "rows"}, "partition_fields_invalid")
        name = info["file"]
        _require(isinstance(name, str) and bool(name) and name not in {".", ".."}
                 and "/" not in name and "\\" not in name and ":" not in name
                 and not PureWindowsPath(name).drive, "partition_must_be_a_portable_basename")
        _require(_hash(info["sha256"]) and _integer(info["rows"]), "partition_identity_invalid")
        path = suite_path / name
        if not path.exists():
            raise PrerequisiteMissing("partition_file_missing")
        _require(not path.is_symlink() and path.is_file(), "partition_not_a_regular_file")
        size = path.stat().st_size
        _require(size > 0, "partition_empty")
        total_bytes += size
        _require(total_bytes <= max_suite_bytes, "suite_byte_budget_exceeded")
        paths.append(path)
    _require(len(set(paths)) == len(PARTITIONS), "partition_paths_not_distinct")
    # Reuse Layev's actual schema, soft/hard labels, ID/group separation and
    # normalized-state leakage checks. No duplicate replacement parser.
    from .data import load_suite
    from .io import sha256_file
    try:
        suite, loaded_manifest = load_suite(suite_path)
    except Exception:
        raise ProtocolError("dataset_validation_failed") from None
    _require(loaded_manifest == manifest, "manifest_changed_during_validation")
    for split, path in zip(PARTITIONS, paths, strict=True):
        _require(sha256_file(path) == partitions[split]["sha256"], "partition_changed_during_validation")
    for path, raw in ((protocol_path, protocol_raw), (review_path, review_raw), (manifest_path, manifest_raw)):
        _require(_read(path) == raw, "input_changed_during_validation")
    return {"schema_version": SCHEMA, "status": "preflight_validated", **counts,
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "suite_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "data_review_sha256": hashlib.sha256(review_raw).hexdigest(),
        "suite_bytes": total_bytes, "max_suite_bytes": max_suite_bytes,
        "partition_rows": {split: len(suite[split]) for split in PARTITIONS},
        "declared_repository_revision": protocol["repository_revision"],
        "data_review_authenticity": "not_verified", "representativeness": "not_established",
        "preregistration_chronology": "not_verified", "baseline_execution": "not_tested",
        "native_model": "not_tested", "trained_context": "not_tested", "cuda": "not_tested",
        "quality": "not_measured", "regression_outcome": "not_tested", "jev_parity": "unknown",
        "source_writes": 0, "network_calls": 0, "training_calls": 0, "paid_calls": 0}
