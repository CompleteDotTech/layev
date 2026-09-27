"""Software-only protocol tests. All data/reviews/hashes below are artificial.

A passing test is NOT a representative dataset, authentic license approval,
native-model result, repaired quality regression, or permission for paid use.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from kev_laya import quality_protocol as gate
from kev_laya.schema import canonical


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(canonical(value) + "\n", encoding="utf-8")


def review():
    return {"schema_version": 1, "review_id": "TEST-ONLY-REVIEW", "reviewer": "test fixture, not an approver",
        "data_kind": "representative_candidate", "license_id": "TEST-ONLY-NOT-A-LICENSE",
        "license_reference": "artificial test declaration; no external data is authorized",
        "intended_population": "artificial schema fixtures", "sampling_method": "four generated records",
        "grouping_policy": "one synthetic group per split", "near_duplicate_review": "test declaration only",
        "limitations": ["Entirely synthetic; not representative evidence."],
        "approved_uses": ["local_training", "local_evaluation"]}


def protocol():
    return {"schema_version": gate.SCHEMA, "protocol_id": "TEST-ONLY-PROTOCOL",
        "repository_revision": "a" * 40, "suite_manifest_sha256": "b" * 64,
        "data_review_sha256": "c" * 64, "seeds": [42, 43, 44],
        "training": {"parent_checkpoint_sha256": "d" * 64, "arms": {
            "supervised": {"optimizer_steps": 10, "useful_token_budget": 1000},
            "reward": {"optimizer_steps": 10, "useful_token_budget": 1000}}},
        "budget": {"max_total_optimizer_steps": 60, "max_wall_seconds": 300,
            "max_peak_gpu_memory_bytes": 1000000, "paid_usd_limit": 0},
        "selection": {"checkpoint_partition": "development", "test_partition": "test",
            "test_evaluations_per_arm_seed": 1, "rule": "minimum development NLL; tie: earliest step"},
        "calibration": {"fit_partition": "calibration", "report": ["raw", "calibrated"]},
        "baselines": {name: {"source_revision": "e" * 40, "artifact_sha256": "f" * 64}
                      for name in ("Kev", "Laya")},
        "metrics": sorted(gate.METRICS), "slices": sorted(gate.SLICES),
        "variations": sorted(gate.VARIATIONS),
        "uncertainty": {"resampling_unit": "source_group", "bootstrap_replicates": 1000, "confidence": .95},
        "fixture_accuracy_threshold": .70,
        "regression": {"state": gate.REGRESSION_STATE, "required_types": ["choice", "noul", "score"],
                       "failure_policy": "retain_and_report"}}


def row(index):
    return {"state": f"TEST-ONLY-SECRET-CONTENT-{index}", "questions": {
        "choice": {"type": "choice", "instructions": "choose", "criteria": {"a": None, "b": None}},
        "noul": {"type": "noul", "instructions": "boolean"},
        "score": {"type": "score", "instructions": "score", "criteria": ["low", "high"]}},
        "gold": {"choice": "a", "noul": True, "score": 0},
        "meta": {"id": f"record-{index}", "group": f"group-{index}",
                 "domain": "explicit-test-fixture", "language": "en"}}


@pytest.fixture
def inputs(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    r = review()
    review_path = tmp_path / "review.json"
    write_json(review_path, r)
    manifest = {"schema_version": 1, "id": "explicit-artificial-test-data",
        "license": r["license_id"], "provenance": "generated solely for software tests",
        "grouping": "artificial disjoint cases", "partitions": {}}
    for i, split in enumerate(gate.PARTITIONS):
        path = suite / (split + ".jsonl")
        write_json(path, row(i))
        manifest["partitions"][split] = {"file": path.name, "sha256": digest(path), "rows": 1}
    write_json(suite / "manifest.json", manifest)
    p = protocol()
    p["suite_manifest_sha256"] = digest(suite / "manifest.json")
    p["data_review_sha256"] = digest(review_path)
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, p)
    return protocol_path, review_path, suite


def edit_manifest(inputs, mutate):
    ppath, _, suite = inputs
    manifest = json.loads((suite / "manifest.json").read_text())
    mutate(manifest)
    write_json(suite / "manifest.json", manifest)
    p = json.loads(ppath.read_text())
    p["suite_manifest_sha256"] = digest(suite / "manifest.json")
    write_json(ppath, p)


def test_complete_declarations_are_not_acceptance():
    p, r = protocol(), review()
    before = deepcopy((p, r))
    assert gate.validate_protocol(p, r) == {"declared_seeds": 3, "declared_training_arms": 2,
                                          "declared_optimizer_steps": 60}
    assert (p, r) == before


@pytest.mark.parametrize("value", [None, [], {}, [1], [1, 2], [1, 1, 2], [True, 2, 3],
    [-1, 2, 3], [1., 2, 3], [[], 2, 3], [1, 2, 2**64]])
def test_seed_policy_rejects_missing_repeated_boolean_noninteger_or_out_of_range(value):
    p = protocol(); p["seeds"] = value
    with pytest.raises(gate.ProtocolError, match="three_distinct_seeds_required"):
        gate.validate_protocol(p, review())


@pytest.mark.parametrize("seed", [0, 2**64 - 1])
def test_seed_endpoints_are_accepted_by_the_protocol_and_trainer_rng(seed):
    import torch

    p = protocol(); p["seeds"] = [seed, 42, 43]
    assert gate.validate_protocol(p, review())["declared_seeds"] == 3
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        assert torch.initial_seed() == seed


@pytest.mark.parametrize("field,value,reason", [
    ("fixture_accuracy_threshold", .69, "fixture_threshold"),
    ("fixture_accuracy_threshold", True, "fixture_threshold"),
    ("fixture_accuracy_threshold", float("nan"), "fixture_threshold"),
    ("repository_revision", "main", "repository_revision"),
    ("suite_manifest_sha256", "x" * 64, "suite_manifest"),
    ("data_review_sha256", [], "data_review_sha256"),
    ("metrics", ["accuracy"], "metrics"),
    ("slices", [], "slices"),
    ("variations", ["option_order"], "variations"),
    ("metrics", [[], "accuracy"], "metrics"),
    ("metrics", sorted(gate.METRICS) + ["accuracy"], "metrics"),
])
def test_incomplete_protocol_rejected(field, value, reason):
    p = protocol(); p[field] = value
    with pytest.raises(gate.ProtocolError, match=reason):
        gate.validate_protocol(p, review())


@pytest.mark.parametrize("path,value,reason", [
    (("training", "arms", "reward", "optimizer_steps"), 11, "unmatched"),
    (("training", "arms", "reward", "useful_token_budget"), 1001, "unmatched"),
    (("training", "arms", "reward", "optimizer_steps"), True, "arm_budget"),
    (("training", "parent_checkpoint_sha256"), "unpinned", "parent_checkpoint"),
    (("budget", "max_total_optimizer_steps"), 59, "total_step_budget"),
    (("budget", "max_total_optimizer_steps"), True, "total_step_budget"),
    (("budget", "max_wall_seconds"), float("inf"), "resource_budget"),
    (("budget", "max_wall_seconds"), -1, "resource_budget"),
    (("budget", "max_wall_seconds"), True, "resource_budget"),
    (("budget", "max_peak_gpu_memory_bytes"), 0, "resource_budget"),
    (("budget", "paid_usd_limit"), 1, "paid_execution"),
    (("budget", "paid_usd_limit"), False, "paid_execution"),
    (("selection", "checkpoint_partition"), "test", "heldout_selection"),
    (("selection", "test_evaluations_per_arm_seed"), 2, "heldout_selection"),
    (("selection", "test_evaluations_per_arm_seed"), True, "heldout_selection"),
    (("selection", "rule"), "  ", "heldout_selection"),
    (("calibration", "fit_partition"), "test", "calibration_partition"),
    (("calibration", "report"), ["calibrated"], "calibration_readouts"),
    (("baselines", "Kev", "source_revision"), "latest", "baseline_not_pinned"),
    (("uncertainty", "resampling_unit"), "question", "uncertainty_policy"),
    (("uncertainty", "bootstrap_replicates"), 999, "uncertainty_policy"),
    (("uncertainty", "confidence"), .9, "uncertainty_policy"),
    (("regression", "state"), "case=100000", "known_regression"),
    (("regression", "failure_policy"), "drop_failed", "known_regression"),
    (("regression", "required_types"), ["choice"], "regression_types"),
])
def test_specific_acceptance_boundaries(path, value, reason):
    p = protocol(); target = p
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    with pytest.raises(gate.ProtocolError, match=reason):
        gate.validate_protocol(p, review())


@pytest.mark.parametrize("key", list(protocol()))
def test_every_protocol_field_is_required(key):
    p = protocol(); del p[key]
    with pytest.raises(gate.ProtocolError, match="protocol_fields"):
        gate.validate_protocol(p, review())


@pytest.mark.parametrize("key", list(review()))
def test_every_review_field_is_required(key):
    r = review(); del r[key]
    with pytest.raises(gate.ProtocolError, match="data_review_fields"):
        gate.validate_protocol(protocol(), r)


@pytest.mark.parametrize("key,value", [("data_kind", "synthetic_fixture"), ("reviewer", ""),
    ("limitations", []), ("approved_uses", ["local_training", "remote_jev"]), ("schema_version", True)])
def test_review_declaration_boundaries(key, value):
    r = review(); r[key] = value
    with pytest.raises(gate.ProtocolError): gate.validate_protocol(protocol(), r)


def test_real_loader_preflight_is_content_free_read_only_and_not_quality(inputs):
    before = {p: p.read_bytes() for p in inputs[0].parent.rglob("*") if p.is_file()}
    result = gate.preflight(*inputs)
    assert result["status"] == "preflight_validated"
    assert result["quality"] == "not_measured"
    assert result["representativeness"] == "not_established"
    assert result["data_review_authenticity"] == result["preregistration_chronology"] == "not_verified"
    assert result["native_model"] == result["cuda"] == result["trained_context"] == "not_tested"
    assert result["jev_parity"] == "unknown" and result["regression_outcome"] == "not_tested"
    assert result["partition_rows"] == dict.fromkeys(gate.PARTITIONS, 1)
    assert all(result[k] == 0 for k in ("source_writes", "network_calls", "paid_calls", "training_calls"))
    encoded = json.dumps(result)
    assert "SECRET-CONTENT" not in encoded and "TEST-ONLY-REVIEW" not in encoded
    assert {p: p.read_bytes() for p in inputs[0].parent.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("which", [0, 1, 2])
def test_missing_input_is_a_blocker_not_a_pass(inputs, which):
    values = list(inputs); values[which] = values[which].parent / "absent"
    with pytest.raises(gate.PrerequisiteMissing): gate.preflight(*values)


@pytest.mark.parametrize("which,reason", [(1, "data_review_hash_mismatch"), (2, "suite_manifest_hash_mismatch")])
def test_changed_review_or_manifest_bytes_fail(inputs, which, reason):
    path = inputs[which] if which == 1 else inputs[2] / "manifest.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(gate.ProtocolError, match=reason): gate.preflight(*inputs)


def test_changed_partition_rejected_by_actual_loader(inputs):
    path = inputs[2] / "train.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(gate.ProtocolError, match="dataset_validation_failed"): gate.preflight(*inputs)


@pytest.mark.parametrize("leak", ["group", "id", "state"])
def test_actual_loader_still_rejects_leakage(inputs, leak):
    suite = inputs[2]
    train = json.loads((suite / "train.jsonl").read_text())
    other = json.loads((suite / "test.jsonl").read_text())
    if leak == "state": other["state"] = train["state"].lower()
    else: other["meta"][leak] = train["meta"][leak]
    write_json(suite / "test.jsonl", other)
    edit_manifest(inputs, lambda m: m["partitions"]["test"].update(sha256=digest(suite / "test.jsonl")))
    with pytest.raises(gate.ProtocolError, match="dataset_validation_failed"): gate.preflight(*inputs)


@pytest.mark.parametrize("fixture_id", sorted(gate.KNOWN_FIXTURE_IDS))
def test_known_fixture_cannot_satisfy_representative_gate(inputs, fixture_id):
    edit_manifest(inputs, lambda m: m.update(id=fixture_id))
    with pytest.raises(gate.ProtocolError, match="known_fixture"): gate.preflight(*inputs)


@pytest.mark.parametrize("name", ["../train.jsonl", "a/train.jsonl", r"a\train.jsonl", r"C:\train.jsonl", "C:train.jsonl", ".", "..", "", None, []])
def test_portable_path_boundary(inputs, name):
    edit_manifest(inputs, lambda m: m["partitions"]["train"].update(file=name))
    with pytest.raises(gate.ProtocolError, match="portable_basename"): gate.preflight(*inputs)


@pytest.mark.parametrize("value", [True, 1.0, 0, -1, "1"])
def test_rows_not_boolean_float_or_nonpositive(inputs, value):
    edit_manifest(inputs, lambda m: m["partitions"]["train"].update(rows=value))
    with pytest.raises(gate.ProtocolError, match="partition_identity"): gate.preflight(*inputs)


def test_declared_license_must_match_manifest(inputs):
    edit_manifest(inputs, lambda m: m.update(license="different"))
    with pytest.raises(gate.ProtocolError, match="license_declaration"): gate.preflight(*inputs)


def test_suite_size_is_bounded_before_loading(inputs, monkeypatch):
    from kev_laya import data
    monkeypatch.setattr(data, "load_suite", lambda _: pytest.fail("must not load oversized data"))
    with pytest.raises(gate.ProtocolError, match="suite_byte_budget_exceeded"):
        gate.preflight(*inputs, max_suite_bytes=1)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'[]', b'\xff'])
def test_strict_documents_reject_duplicate_keys_nonfinite_and_nonobjects(inputs, raw):
    inputs[0].write_bytes(raw)
    with pytest.raises(gate.ProtocolError): gate.preflight(*inputs)


def test_document_size_bound(inputs):
    inputs[0].write_bytes(b" " * (gate.MAX_DOCUMENT_BYTES + 1))
    with pytest.raises(gate.ProtocolError, match="document_size_limit"): gate.preflight(*inputs)


def test_final_readback_detects_changed_input(inputs, monkeypatch):
    from kev_laya import data
    original = data.load_suite
    def changed(directory):
        result = original(directory)
        inputs[1].write_bytes(inputs[1].read_bytes() + b"\n")
        return result
    monkeypatch.setattr(data, "load_suite", changed)
    with pytest.raises(gate.ProtocolError, match="input_changed_during_validation"): gate.preflight(*inputs)


def test_partition_symlink_rejected_without_loading_target(inputs):
    path = inputs[2] / "train.jsonl"
    target = path.with_name("target.jsonl")
    path.rename(target)
    try: path.symlink_to(target.name)
    except OSError: pytest.skip("platform does not permit test symlinks")
    with pytest.raises(gate.ProtocolError, match="partition_not_a_regular_file"): gate.preflight(*inputs)


def command(*args):
    script = Path(__file__).resolve().parents[1] / "scripts/verify_quality_protocol.py"
    return subprocess.run([sys.executable, str(script), *args], text=True, capture_output=True, timeout=15)


def test_cli_missing_prerequisites_exits_two_without_quality_claim():
    result = command()
    assert result.returncode == 2 and not result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "blocked" and receipt["quality"] == "not_measured"
    assert receipt["training_calls"] == receipt["network_calls"] == 0


def test_cli_preflight_is_not_a_native_or_quality_receipt(inputs):
    result = command("--protocol", str(inputs[0]), "--data-review", str(inputs[1]), "--suite", str(inputs[2]))
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "preflight_validated"
    assert receipt["quality"] == "not_measured" and receipt["jev_parity"] == "unknown"


def test_cli_rejection_does_not_echo_dataset_content(inputs):
    write_json(inputs[2] / "train.jsonl", {"SECRET-PRIVATE-CONTENT": "not a datum"})
    result = command("--protocol", str(inputs[0]), "--data-review", str(inputs[1]), "--suite", str(inputs[2]))
    assert result.returncode == 1
    assert "SECRET-PRIVATE-CONTENT" not in result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "rejected"


@pytest.mark.parametrize("value", [None, [], {}, 1, True, ""])
def test_manifest_id_is_a_string_before_membership(inputs, value):
    edit_manifest(inputs, lambda m: m.update(id=value))
    with pytest.raises(gate.ProtocolError, match="suite_identity_invalid"):
        gate.preflight(*inputs)


def test_large_integer_resource_budget_is_checked_without_float_overflow():
    p = protocol(); p["budget"]["max_wall_seconds"] = 10 ** 400
    assert gate.validate_protocol(p, review())["declared_optimizer_steps"] == 60
