"""Synthetic transaction tests; these are not representative quality evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from kev_laya.checkpoint import load_checkpoint, save_checkpoint
from kev_laya.cli import main
from kev_laya.data import load_suite
from kev_laya.encoding import ByteTokenizer, Limits
from kev_laya.evaluation import evaluate
from kev_laya.io import atomic_json, sha256_file
from kev_laya.quality_protocol import ProtocolError
from kev_laya.quality_readout import ReadoutError, QualityReadoutLedger, validate_raw_checkpoint
from test_quality_protocol import inputs, write_json  # noqa: F401  (register pytest fixture)


@pytest.fixture
def readout_case(inputs, tiny):  # noqa: F811  (pytest fixture injection)
    protocol_path, review_path, suite_path = inputs
    root = protocol_path.parent
    raw = root / "raw.pt"
    calibrated = root / "calibrated.pt"
    tiny.training_steps = 1
    tokenizer = ByteTokenizer()
    suite, manifest = load_suite(suite_path)
    provenance = {"context_limits": {"branch": 512, "aggregate": 8192},
                  "seed": 42,
                  "split_hashes": {name: info["sha256"] for name, info in manifest["partitions"].items()}}
    tiny.calibration_provenance = {"status": "unfitted-after-weight-training"}
    save_checkpoint(raw, tiny, tokenizer,
                    provenance=provenance)
    dev = evaluate(tiny, suite["development"], tokenizer, Limits(512, 8192),
                   split="development")
    dev.update(checkpoint_sha256=sha256_file(raw),
               split_sha256=manifest["partitions"]["development"]["sha256"])
    dev_path = root / "development.json"
    atomic_json(dev_path, dev)
    tiny.temperatures = {name: 1.1 for name in tiny.temperatures}
    tiny.calibration_provenance = {"status": "fitted-held-out", "partition": "calibration",
        "sha256": manifest["partitions"]["calibration"]["sha256"]}
    save_checkpoint(calibrated, tiny, tokenizer,
                    provenance=provenance,
                    parent_sha256=sha256_file(raw), parent_checkpoint=raw,
                    exposure_operation="calibration")
    selection = {"schema_version": "layev-quality-selection/1",
                 "protocol_sha256": sha256_file(protocol_path),
                 "suite_manifest_sha256": sha256_file(suite_path / "manifest.json"),
                 "arm": "supervised", "seed": 42,
                 "development_report_sha256": sha256_file(dev_path),
                 "raw_checkpoint_sha256": sha256_file(raw),
                 "calibrated_checkpoint_sha256": sha256_file(calibrated)}
    selection_path = root / "selection.json"
    write_json(selection_path, selection)
    return {"protocol": protocol_path, "review": review_path, "suite": suite_path,
            "selection": selection_path, "development": dev_path, "raw": raw,
            "calibrated": calibrated, "ledger": root / "readout-ledger.json",
            "output": root / "paired.json"}


def argv(case):
    return ["evaluate", "--split", "test", "--checkpoint", str(case["raw"]),
            "--calibrated-checkpoint", str(case["calibrated"]),
            "--suite", str(case["suite"]), "--out", str(case["output"]),
            "--quality-protocol", str(case["protocol"]),
            "--quality-data-review", str(case["review"]),
            "--quality-selection", str(case["selection"]),
            "--development-report", str(case["development"]),
            "--quality-readout-ledger", str(case["ledger"])]


def test_test_split_requires_claim_before_suite_load(readout_case, monkeypatch):
    from kev_laya import quality_readout
    original = quality_readout.preflight
    def checked(*args, **kwargs):
        state = json.loads(readout_case["ledger"].read_text())
        assert state["claims"]["supervised:42"]["status"] == "claimed"
        return original(*args, **kwargs)
    monkeypatch.setattr(quality_readout, "preflight", checked)
    assert main(argv(readout_case)) == 0
    result = json.loads(readout_case["output"].read_text())
    assert set(result["readouts"]) == {"raw", "calibrated"}
    assert result["readouts"]["raw"]["split"] == "test"
    state = json.loads(readout_case["ledger"].read_text())
    assert state["claims"]["supervised:42"]["status"] == "completed"
    assert state["claims"]["supervised:42"]["output_sha256"] == sha256_file(readout_case["output"])
    with pytest.raises(ReadoutError, match="already_claimed"):
        main(argv(readout_case))


def test_crash_after_claim_blocks_rerun_without_test_read(readout_case, monkeypatch):
    from kev_laya import quality_readout
    calls = []
    def crash(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("synthetic crash after durable claim")
    monkeypatch.setattr(quality_readout, "preflight", crash)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        main(argv(readout_case))
    assert calls == [1] and not readout_case["output"].exists()
    with pytest.raises(ReadoutError, match="already_claimed"):
        main(argv(readout_case))
    assert calls == [1]
    with QualityReadoutLedger(readout_case["ledger"],
            protocol_sha256=sha256_file(readout_case["protocol"]),
            suite_manifest_sha256=sha256_file(readout_case["suite"] / "manifest.json")) as ledger:
        with pytest.raises(ReadoutError, match="no_reconcilable_output"):
            ledger.reconcile_output("supervised:42", readout_case["output"])


def test_reconcile_only_existing_complete_paired_output(readout_case, monkeypatch):
    from kev_laya import quality_readout
    original = quality_readout.QualityReadoutLedger.complete
    def crash(_ledger, _key, _output):
        raise RuntimeError("synthetic crash after output write")
    monkeypatch.setattr(quality_readout.QualityReadoutLedger, "complete", crash)
    with pytest.raises(RuntimeError, match="after output write"):
        main(argv(readout_case))
    monkeypatch.setattr(quality_readout.QualityReadoutLedger, "complete", original)
    with QualityReadoutLedger(readout_case["ledger"],
            protocol_sha256=sha256_file(readout_case["protocol"]),
            suite_manifest_sha256=sha256_file(readout_case["suite"] / "manifest.json")) as ledger:
        ledger.reconcile_output("supervised:42", readout_case["output"])
    state = json.loads(readout_case["ledger"].read_text())
    assert state["claims"]["supervised:42"]["status"] == "completed"
    with pytest.raises(ReadoutError, match="already_claimed"):
        main(argv(readout_case))


def test_mutated_selection_fails_before_claim(readout_case):
    selection = json.loads(readout_case["selection"].read_text())
    selection["raw_checkpoint_sha256"] = "0" * 64
    write_json(readout_case["selection"], selection)
    with pytest.raises(ReadoutError, match="selection_identity_mismatch"):
        main(argv(readout_case))
    assert not readout_case["ledger"].exists()


@pytest.mark.parametrize("mutation,reason", [
    ("zero_steps", "not_trained"),
    ("wrong_seed", "seed_mismatch"),
    ("wrong_split", "splits_mismatch"),
    ("already_calibrated", "already_calibrated"),
    ("nonunit_temperature", "nonunit_temperature"),
])
def test_selected_raw_checkpoint_rejects_invalid_training_identity(readout_case, mutation, reason):
    model, _, point = load_checkpoint(readout_case["raw"])
    selection = json.loads(readout_case["selection"].read_text())
    manifest = json.loads((readout_case["suite"] / "manifest.json").read_text())
    if mutation == "zero_steps":
        point["training_steps"] = 0
    elif mutation == "wrong_seed":
        point["provenance"]["seed"] = 43
    elif mutation == "wrong_split":
        point["provenance"]["split_hashes"]["train"] = "0" * 64
    elif mutation == "already_calibrated":
        model.calibration_provenance = {"status": "fitted-held-out"}
    else:
        model.temperatures["choice"] = 1.1
    with pytest.raises(ReadoutError, match=reason):
        validate_raw_checkpoint(model, point, selection, manifest)


def test_selected_raw_checkpoint_matches_training_provenance(readout_case):
    model, _, point = load_checkpoint(readout_case["raw"])
    selection = json.loads(readout_case["selection"].read_text())
    manifest = json.loads((readout_case["suite"] / "manifest.json").read_text())
    validate_raw_checkpoint(model, point, selection, manifest)


def test_calibrated_checkpoint_must_descend_from_selected_raw(readout_case):
    manifest_path = readout_case["calibrated"].with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["parent_sha256"] = "0" * 64
    write_json(manifest_path, manifest)
    with pytest.raises(ReadoutError, match="calibrated_checkpoint_parent_mismatch"):
        main(argv(readout_case))
    assert not readout_case["ledger"].exists()


def test_suite_change_after_claim_cannot_trigger_retry(readout_case, monkeypatch):
    from kev_laya import quality_readout
    original = quality_readout.preflight
    def changed(*args, **kwargs):
        test_path = readout_case["suite"] / "test.jsonl"
        test_path.write_bytes(test_path.read_bytes() + b"\n")
        return original(*args, **kwargs)
    monkeypatch.setattr(quality_readout, "preflight", changed)
    with pytest.raises(ProtocolError, match="dataset_validation_failed"):
        main(argv(readout_case))
    state = json.loads(readout_case["ledger"].read_text())
    assert state["claims"]["supervised:42"]["status"] == "claimed"
    with pytest.raises(ReadoutError, match="already_claimed"):
        main(argv(readout_case))


def test_unguarded_test_cli_rejected_before_suite_load(readout_case, monkeypatch):
    from kev_laya import cli
    monkeypatch.setattr(cli, "load_suite", lambda *_: pytest.fail("test was read without a claim"))
    with pytest.raises(ValueError, match="paired quality readout claim"):
        main(["evaluate", "--split", "test", "--checkpoint", str(readout_case["raw"]),
              "--suite", str(readout_case["suite"]), "--out", str(readout_case["output"])])


def test_diagnostic_test_rejects_representative_suite_before_loading(readout_case, monkeypatch):
    from kev_laya import cli
    manifest_path = readout_case["suite"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["id"] = "representative-quality-v1"
    write_json(manifest_path, manifest)
    monkeypatch.setattr(cli, "load_suite", lambda *_: pytest.fail("test was read without a claim"))
    with pytest.raises(ValueError, match="known synthetic fixture"):
        main(["evaluate", "--split", "test", "--diagnostic-test", "--checkpoint",
              str(readout_case["raw"]), "--suite", str(readout_case["suite"]),
              "--out", str(readout_case["output"])])


@pytest.mark.parametrize("mutation,reason", [
    ("weights", "changed_model_weights"),
    ("context_limits", "changed_model_contract"),
    ("training_steps", "changed_model_contract"),
    ("execution", "changed_model_contract"),
])
def test_calibrated_child_rejects_changed_model_or_contract(readout_case, mutation, reason):
    calibrated = readout_case["calibrated"]
    payload = torch.load(calibrated, map_location="cpu", weights_only=True)
    if mutation == "weights":
        first = next(iter(payload["model"].values()))
        first.view(-1)[0] += 1
    elif mutation == "context_limits":
        payload["provenance"]["context_limits"]["branch"] = 256
    elif mutation == "training_steps":
        payload["training_steps"] += 1
    else:
        payload["execution"]["max_branches"] += 1
    torch.save(payload, calibrated)
    manifest_path = calibrated.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = sha256_file(calibrated)
    write_json(manifest_path, manifest)
    selection = json.loads(readout_case["selection"].read_text())
    selection["calibrated_checkpoint_sha256"] = manifest["sha256"]
    write_json(readout_case["selection"], selection)
    with pytest.raises(ReadoutError, match=reason):
        main(argv(readout_case))
    assert not readout_case["output"].exists()


def test_direct_test_evaluator_requires_claim(readout_case, tiny):
    suite, _ = load_suite(readout_case["suite"])
    with pytest.raises(ValueError, match="durable readout claim"):
        evaluate(tiny, suite["test"], ByteTokenizer(), Limits(512, 8192), split="test")


def test_claimed_python_evaluator_consumes_each_kind_once(readout_case, tiny):
    case = readout_case
    selection = json.loads(case["selection"].read_text())
    manifest = json.loads((case["suite"] / "manifest.json").read_text())
    with QualityReadoutLedger(case["ledger"], protocol_sha256=sha256_file(case["protocol"]),
                              suite_manifest_sha256=sha256_file(case["suite"] / "manifest.json")) as ledger:
        ledger.claim("supervised:42", selection_sha256=sha256_file(case["selection"]),
                     selection=selection, test_split_sha256=manifest["partitions"]["test"]["sha256"],
                     output=case["output"])
        suite, _ = load_suite(case["suite"])
        args = (tiny, suite["test"], ByteTokenizer(), Limits(512, 8192))
        evaluate(*args, split="test", test_claim=ledger, test_readout_kind="raw",
                 checkpoint_sha256=selection["raw_checkpoint_sha256"])
        with pytest.raises(ReadoutError, match="kind_already_used"):
            evaluate(*args, split="test", test_claim=ledger, test_readout_kind="raw",
                     checkpoint_sha256=selection["raw_checkpoint_sha256"])
        with pytest.raises(ReadoutError, match="checkpoint_not_claimed"):
            evaluate(*args, split="test", test_claim=ledger, test_readout_kind="calibrated",
                     checkpoint_sha256=selection["raw_checkpoint_sha256"])
        evaluate(*args, split="test", test_claim=ledger, test_readout_kind="calibrated",
                 checkpoint_sha256=selection["calibrated_checkpoint_sha256"])
        with pytest.raises(ReadoutError, match="kind_already_used"):
            evaluate(*args, split="test", test_claim=ledger, test_readout_kind="calibrated",
                     checkpoint_sha256=selection["calibrated_checkpoint_sha256"])
    with pytest.raises(ReadoutError, match="already_claimed"):
        main(argv(case))


def test_failed_first_forward_consumes_kind_and_leaves_claimed(readout_case, tiny, monkeypatch):
    case = readout_case
    selection = json.loads(case["selection"].read_text())
    manifest = json.loads((case["suite"] / "manifest.json").read_text())
    with QualityReadoutLedger(case["ledger"], protocol_sha256=sha256_file(case["protocol"]),
                              suite_manifest_sha256=sha256_file(case["suite"] / "manifest.json")) as ledger:
        ledger.claim("supervised:42", selection_sha256=sha256_file(case["selection"]),
                     selection=selection, test_split_sha256=manifest["partitions"]["test"]["sha256"],
                     output=case["output"])
        suite, _ = load_suite(case["suite"])
        def fail_forward(*_args, **_kwargs):
            raise RuntimeError("synthetic forward failure")
        monkeypatch.setattr(tiny, "forward", fail_forward)
        args = (tiny, suite["test"], ByteTokenizer(), Limits(512, 8192))
        with pytest.raises(RuntimeError, match="synthetic forward failure"):
            evaluate(*args, split="test", test_claim=ledger, test_readout_kind="raw",
                     checkpoint_sha256=selection["raw_checkpoint_sha256"])
        with pytest.raises(ReadoutError, match="kind_already_used"):
            evaluate(*args, split="test", test_claim=ledger, test_readout_kind="raw",
                     checkpoint_sha256=selection["raw_checkpoint_sha256"])
    with QualityReadoutLedger(case["ledger"], protocol_sha256=sha256_file(case["protocol"]),
                              suite_manifest_sha256=sha256_file(case["suite"] / "manifest.json")) as ledger:
        with pytest.raises(ReadoutError, match="no_reconcilable_output"):
            ledger.reconcile_output("supervised:42", case["output"])
    with pytest.raises(ReadoutError, match="already_claimed"):
        main(argv(case))


def test_second_process_cannot_acquire_active_readout_ledger(readout_case):
    protocol_hash = sha256_file(readout_case["protocol"])
    manifest_hash = sha256_file(readout_case["suite"] / "manifest.json")
    code = ("from pathlib import Path; from kev_laya.quality_readout import QualityReadoutLedger; "
            f"p=Path({str(readout_case['ledger'])!r}); "
            f"with QualityReadoutLedger(p, protocol_sha256={protocol_hash!r}, "
            f"suite_manifest_sha256={manifest_hash!r}): pass")
    # Compound `with` needs a real statement boundary, not a one-line semicolon.
    code = code.replace("; with QualityReadoutLedger", "\nwith QualityReadoutLedger")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    with QualityReadoutLedger(readout_case["ledger"], protocol_sha256=protocol_hash,
                              suite_manifest_sha256=manifest_hash):
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                text=True, timeout=10)
    assert result.returncode != 0
    assert "lock" in result.stderr.lower() or "permission" in result.stderr.lower()
