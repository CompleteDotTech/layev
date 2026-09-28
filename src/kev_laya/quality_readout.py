"""Single-use, paired untouched-test readout for a frozen quality protocol.

This ledger is a crash barrier, not an authentication or data-review service.
Once claimed, an arm/seed cannot run again under the same protocol. A completed
output can be reconciled after a crash without opening the test partition.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import torch

from .io import atomic_json, sha256_file
from .quality_budget import _lock, _unlock
from .quality_protocol import MAX_DOCUMENT_BYTES, preflight, validate_protocol
from .schema import strict_loads


class ReadoutError(RuntimeError):
    pass


MAX_OUTPUT_BYTES = 1024 * 1024 * 1024


def validate_calibrated_child(raw_model, raw_point: dict, calibrated_model,
                              calibrated_point: dict) -> None:
    """Temperature fitting must leave the decision model and input contract intact."""
    from .cli import limits_for

    if (raw_model.config_dict() != calibrated_model.config_dict()
            or raw_model.batch_policy.to_dict() != calibrated_model.batch_policy.to_dict()
            or raw_model.training_steps != calibrated_model.training_steps
            or limits_for(raw_point) != limits_for(calibrated_point)):
        raise ReadoutError("calibrated_checkpoint_changed_model_contract")
    for name in ("config", "execution", "preprocessing", "training_steps",
                 "native_weights_loaded", "source_provenance", "backbone_source",
                 "loaded_backbone_sha256"):
        if raw_point.get(name) != calibrated_point.get(name):
            raise ReadoutError("calibrated_checkpoint_changed_model_contract")
    for name in ("seed", "split_hashes", "data_sha256", "config_sha256"):
        recorded = (raw_point.get("provenance") or {}).get(name)
        if recorded is not None and recorded != (calibrated_point.get("provenance") or {}).get(name):
            raise ReadoutError("calibrated_checkpoint_changed_model_contract")
    raw_weights = raw_model.state_dict()
    calibrated_weights = calibrated_model.state_dict()
    if raw_weights.keys() != calibrated_weights.keys() or any(
        not torch.equal(weight, calibrated_weights[name]) for name, weight in raw_weights.items()
    ):
        raise ReadoutError("calibrated_checkpoint_changed_model_weights")


def _document(path: Path, *, max_bytes: int = MAX_DOCUMENT_BYTES) -> tuple[bytes, dict]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ReadoutError("readout_input_not_regular_file")
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ReadoutError("readout_document_too_large")
    value = strict_loads(raw)
    if not isinstance(value, dict):
        raise ReadoutError("readout_document_not_object")
    return raw, value


def validate_selection(selection: dict, protocol: dict, *, protocol_sha256: str,
                       development_report: dict, development_report_sha256: str,
                       raw_checkpoint_sha256: str, calibrated_checkpoint_sha256: str,
                       calibrated_parent_sha256: str) -> str:
    keys = {"schema_version", "protocol_sha256", "suite_manifest_sha256", "arm", "seed",
            "development_report_sha256", "raw_checkpoint_sha256", "calibrated_checkpoint_sha256"}
    if set(selection) != keys or selection["schema_version"] != "layev-quality-selection/1":
        raise ReadoutError("selection_fields_invalid")
    arm, seed = selection["arm"], selection["seed"]
    if (not isinstance(arm, str) or arm not in protocol["training"]["arms"]
            or type(seed) is not int or seed not in protocol["seeds"]):
        raise ReadoutError("selection_arm_seed_not_preregistered")
    expected = {"protocol_sha256": protocol_sha256,
                "suite_manifest_sha256": protocol["suite_manifest_sha256"],
                "development_report_sha256": development_report_sha256,
                "raw_checkpoint_sha256": raw_checkpoint_sha256,
                "calibrated_checkpoint_sha256": calibrated_checkpoint_sha256}
    if any(selection[name] != value for name, value in expected.items()):
        raise ReadoutError("selection_identity_mismatch")
    if (development_report.get("split") != "development"
            or development_report.get("split_sha256") is None
            or development_report.get("checkpoint_sha256") != raw_checkpoint_sha256):
        raise ReadoutError("selection_development_report_invalid")
    if calibrated_parent_sha256 != raw_checkpoint_sha256:
        raise ReadoutError("calibrated_checkpoint_parent_mismatch")
    return f"{arm}:{seed}"


class QualityReadoutLedger:
    """Exclusive ledger lock covers claim, test access, and paired output commit."""

    def __init__(self, path: Path, *, protocol_sha256: str, suite_manifest_sha256: str):
        self.path = Path(path)
        self.protocol_sha256 = protocol_sha256
        self.suite_manifest_sha256 = suite_manifest_sha256
        self.fd: int | None = None
        self.current_key: str | None = None
        self.used_readouts: set[str] = set()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path.with_name(self.path.name + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            _lock(self.fd)
            if self.path.exists():
                self.state = strict_loads(self.path.read_bytes())
                if (self.state.get("schema_version") != "layev-quality-test-readout/1"
                        or self.state.get("protocol_sha256") != self.protocol_sha256
                        or self.state.get("suite_manifest_sha256") != self.suite_manifest_sha256
                        or not isinstance(self.state.get("claims"), dict)):
                    raise ReadoutError("test_readout_ledger_identity_mismatch")
            else:
                self.state = {"schema_version": "layev-quality-test-readout/1",
                              "protocol_sha256": self.protocol_sha256,
                              "suite_manifest_sha256": self.suite_manifest_sha256, "claims": {}}
                self._save()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.fd is not None:
            _unlock(self.fd)
            os.close(self.fd)
            self.fd = None
        self.current_key = None
        self.used_readouts.clear()

    def _save(self):
        temporary = self.path.with_name(self.path.name + ".tmp")
        if temporary.exists():
            raise ReadoutError("unreconciled_readout_ledger_temp")
        data = (json.dumps(self.state, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def claim(self, key: str, *, selection_sha256: str, selection: dict,
              test_split_sha256: str, output: Path):
        if key in self.state["claims"]:
            raise ReadoutError("test_readout_already_claimed")
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        self.state["claims"][key] = {"status": "claimed", "selection_sha256": selection_sha256,
                                      "raw_checkpoint_sha256": selection["raw_checkpoint_sha256"],
                                      "calibrated_checkpoint_sha256": selection["calibrated_checkpoint_sha256"],
                                      "test_split_sha256": test_split_sha256,
                                      "output": str(output.resolve()), "output_sha256": None}
        self._save()  # Must complete before any test-partition read.
        self.current_key = key
        self.used_readouts.clear()

    def consume_test_readout(self, kind: str, checkpoint_sha256: str) -> None:
        if not isinstance(kind, str) or kind not in {"raw", "calibrated"}:
            raise ReadoutError("test_readout_kind_invalid")
        if self.fd is None or self.current_key is None:
            raise ReadoutError("test_readout_claim_not_active")
        claim = self.state["claims"].get(self.current_key, {})
        if (claim.get("status") != "claimed"
                or checkpoint_sha256 != claim.get(kind + "_checkpoint_sha256")):
            raise ReadoutError("test_readout_checkpoint_not_claimed")
        if kind in self.used_readouts:
            raise ReadoutError("test_readout_kind_already_used")
        # Consume before any model operation, including a failing forward pass.
        self.used_readouts.add(kind)

    def complete(self, key: str, output: Path):
        claim = self.state["claims"][key]
        if claim["status"] != "claimed" or claim["output"] != str(output.resolve()):
            raise ReadoutError("test_readout_claim_mismatch")
        if self.current_key != key or self.used_readouts != {"raw", "calibrated"}:
            raise ReadoutError("paired_test_readouts_not_both_consumed")
        claim.update(status="completed", output_sha256=self._verified_output_digest(key, output))
        self._save()

    def _verified_output_digest(self, key: str, output: Path) -> str:
        claim = self.state["claims"][key]
        if not output.is_file() or output.is_symlink():
            raise ReadoutError("claimed_readout_has_no_reconcilable_output")
        _, report = _document(output, max_bytes=MAX_OUTPUT_BYTES)
        readouts = report.get("readouts")
        if (report.get("schema_version") != "layev-quality-paired-test/1"
                or report.get("protocol_sha256") != self.protocol_sha256
                or report.get("suite_manifest_sha256") != self.suite_manifest_sha256
                or report.get("selection_sha256") != claim["selection_sha256"]
                or report.get("arm_seed") != key
                or not isinstance(readouts, dict) or set(readouts) != {"raw", "calibrated"}):
            raise ReadoutError("claimed_readout_output_invalid")
        for name, expected in (("raw", claim["raw_checkpoint_sha256"]),
                               ("calibrated", claim["calibrated_checkpoint_sha256"])):
            row = readouts[name]
            if (not isinstance(row, dict) or row.get("split") != "test"
                    or row.get("quality_gate") != "claimed_paired_test"
                    or row.get("split_sha256") != claim["test_split_sha256"]
                    or row.get("checkpoint_sha256") != expected):
                raise ReadoutError("claimed_readout_output_invalid")
        return sha256_file(output)

    def reconcile_output(self, key: str, output: Path):
        """After a crash, record an already-written result; never evaluate again."""
        claim = self.state["claims"].get(key)
        if claim is None or claim["output"] != str(output.resolve()):
            raise ReadoutError("test_readout_claim_mismatch")
        if claim["status"] == "completed":
            if sha256_file(output) != claim["output_sha256"]:
                raise ReadoutError("completed_readout_output_changed")
            return
        if claim["status"] != "claimed":
            raise ReadoutError("claimed_readout_has_no_reconcilable_output")
        claim.update(status="completed", output_sha256=self._verified_output_digest(key, output))
        self._save()


def prepare_selection(protocol_path: Path, review_path: Path, selection_path: Path,
                      development_report_path: Path, raw_checkpoint: Path,
                      calibrated_checkpoint: Path, suite_manifest_path: Path):
    protocol_raw, protocol = _document(protocol_path)
    review_raw, review = _document(review_path)
    validate_protocol(protocol, review)
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    selection_raw, selection = _document(selection_path)
    _, development = _document(development_report_path)
    manifest_raw, manifest = _document(suite_manifest_path)
    if (hashlib.sha256(manifest_raw).hexdigest() != protocol["suite_manifest_sha256"]
            or hashlib.sha256(review_raw).hexdigest() != protocol["data_review_sha256"]):
        raise ReadoutError("quality_input_identity_mismatch")
    raw_hash = sha256_file(raw_checkpoint)
    calibrated_hash = sha256_file(calibrated_checkpoint)
    _, calibrated_manifest = _document(calibrated_checkpoint.with_suffix(".manifest.json"))
    if calibrated_manifest.get("sha256") != calibrated_hash:
        raise ReadoutError("calibrated_checkpoint_manifest_mismatch")
    key = validate_selection(selection, protocol, protocol_sha256=protocol_sha256,
        development_report=development,
        development_report_sha256=sha256_file(development_report_path),
        raw_checkpoint_sha256=raw_hash, calibrated_checkpoint_sha256=calibrated_hash,
        calibrated_parent_sha256=calibrated_manifest.get("parent_sha256"))
    if development["split_sha256"] != manifest["partitions"]["development"]["sha256"]:
        raise ReadoutError("development_split_identity_mismatch")
    return (key, protocol_sha256, protocol, hashlib.sha256(selection_raw).hexdigest(),
            selection, manifest["partitions"]["test"]["sha256"])


def validate_raw_checkpoint(model, point: dict, selection: dict, manifest: dict) -> None:
    """The selected raw arm must be an unfitted training checkpoint on this suite."""
    if type(point.get("training_steps")) is not int or point["training_steps"] <= 0:
        raise ReadoutError("selected_raw_checkpoint_not_trained")
    provenance = point.get("provenance")
    if not isinstance(provenance, dict) or type(provenance.get("seed")) is not int or provenance["seed"] != selection["seed"]:
        raise ReadoutError("selected_raw_checkpoint_seed_mismatch")
    expected_splits = {name: row["sha256"] for name, row in manifest["partitions"].items()}
    if provenance.get("split_hashes") != expected_splits:
        raise ReadoutError("selected_raw_checkpoint_splits_mismatch")
    if model.calibration_provenance.get("status") != "unfitted-after-weight-training":
        raise ReadoutError("selected_raw_checkpoint_already_calibrated")
    temperatures = model.temperatures
    if (set(temperatures) != {"choice", "score", "noul"}
            or any(type(value) not in (int, float) or value != 1.0 for value in temperatures.values())):
        raise ReadoutError("selected_raw_checkpoint_nonunit_temperature")


def run_paired_test(*, protocol_path: Path, review_path: Path, selection_path: Path,
                    development_report_path: Path, raw_checkpoint: Path,
                    calibrated_checkpoint: Path, suite_path: Path, ledger_path: Path,
                    output: Path, device: str) -> dict:
    """Claim before preflight/load_suite; never retry a claimed arm/seed."""
    from .checkpoint import load_checkpoint
    from .cli import limits_for
    from .data import load_suite
    from .evaluation import evaluate

    suite_path, output = Path(suite_path), Path(output)
    key, protocol_sha256, protocol, selection_sha256, selection, test_split_sha256 = prepare_selection(
        protocol_path, review_path, selection_path, development_report_path,
        raw_checkpoint, calibrated_checkpoint, suite_path / "manifest.json")
    with QualityReadoutLedger(ledger_path, protocol_sha256=protocol_sha256,
                           suite_manifest_sha256=protocol["suite_manifest_sha256"]) as ledger:
        ledger.claim(key, selection_sha256=selection_sha256, selection=selection,
                     test_split_sha256=test_split_sha256, output=output)
        # The test partition is first opened by preflight, only after claim is durable.
        receipt = preflight(protocol_path, review_path, suite_path)
        if (receipt["protocol_sha256"] != protocol_sha256
                or receipt["suite_manifest_sha256"] != protocol["suite_manifest_sha256"]):
            raise ReadoutError("quality_inputs_changed_after_claim")
        suite, manifest = load_suite(suite_path)
        if sha256_file(suite_path / "manifest.json") != protocol["suite_manifest_sha256"]:
            raise ReadoutError("suite_manifest_changed_after_claim")
        raw_hash = sha256_file(raw_checkpoint)
        calibrated_hash = sha256_file(calibrated_checkpoint)
        raw_model, raw_tokenizer, raw_point = load_checkpoint(raw_checkpoint, device,
                                                              expected_sha256=raw_hash)
        validate_raw_checkpoint(raw_model, raw_point, selection, manifest)
        calibrated_model, calibrated_tokenizer, calibrated_point = load_checkpoint(
            calibrated_checkpoint, device, expected_sha256=calibrated_hash)
        _, selection = _document(selection_path)
        if (sha256_file(selection_path) != selection_sha256
                or raw_hash != selection["raw_checkpoint_sha256"]
                or calibrated_hash != selection["calibrated_checkpoint_sha256"]
                or calibrated_point.get("parent_sha256") != raw_hash
                or raw_tokenizer.identity != calibrated_tokenizer.identity):
            raise ReadoutError("selected_checkpoint_changed_after_claim")
        calibration = calibrated_model.calibration_provenance
        if (calibration.get("status") != "fitted-held-out"
                or calibration.get("partition") != "calibration"
                or calibration.get("sha256") != manifest["partitions"]["calibration"]["sha256"]):
            raise ReadoutError("calibrated_checkpoint_not_fitted_on_frozen_calibration")
        validate_calibrated_child(raw_model, raw_point, calibrated_model, calibrated_point)
        raw_report = evaluate(raw_model, suite["test"], raw_tokenizer,
                              limits_for(raw_point), split="test", test_claim=ledger,
                              test_readout_kind="raw", checkpoint_sha256=raw_hash)
        calibrated_report = evaluate(calibrated_model, suite["test"], calibrated_tokenizer,
                                     limits_for(calibrated_point), split="test", test_claim=ledger,
                                     test_readout_kind="calibrated", checkpoint_sha256=calibrated_hash)
        for report, point in ((raw_report, raw_point), (calibrated_report, calibrated_point)):
            report.update(checkpoint_sha256=point["checkpoint_sha256"], model_id=point["model_id"],
                          split_sha256=manifest["partitions"]["test"]["sha256"])
        result = {"schema_version": "layev-quality-paired-test/1", "protocol_sha256": protocol_sha256,
                  "selection_sha256": selection_sha256,
                  "suite_manifest_sha256": protocol["suite_manifest_sha256"],
                  "arm_seed": key, "readouts": {"raw": raw_report, "calibrated": calibrated_report}}
        atomic_json(output, result)
        ledger.complete(key, output)
        return result
