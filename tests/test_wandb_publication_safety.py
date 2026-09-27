"""Publication policy regressions using explicit SDK doubles, not live W&B proof."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from kev_laya.encoding import ByteTokenizer, Limits
from kev_laya.objectives import ObjectiveConfig
from kev_laya.publishing import WandbPublisher, publish_wandb
from kev_laya.telemetry import new_snapshot
from kev_laya.training import TrainSettings, train

IDENTITY = {"entity": "entity", "project": "project", "run_id": "existing-run"}


@pytest.fixture
def telemetry():
    provenance = {
        "model": "fixture", "backbone": "fixture", "backbone_revision": "fixture-v1",
        "tokenizer": "bytes", "config_sha256": "1" * 64, "data_sha256": "2" * 64,
        "split_hashes": {}, "seed": 42, "repository": None, "commit": None,
        "precision": "fp32", "hardware": "cpu", "context_limits": {"branch": 512, "aggregate": 8192},
        "evidence_class": "tiny-synthetic-fixture",
    }
    return new_snapshot("experiment", "local-run", "attempt-0", provenance, wandb=IDENTITY)


class SummaryDouble(dict):
    def __init__(self):
        super().__init__()
        self.assignments = 0
        self.updates = 0
        self.error = None

    def __setitem__(self, key, value):
        self.assignments += 1
        super().__setitem__(key, value)

    def update(self):
        self.updates += 1
        if self.error is not None:
            raise self.error


class RunDouble:
    def __init__(self, framework=None):
        self.config = {"framework": framework, "unrelated": "preserve"}
        self.summary = SummaryDouble()
        self.updates = 0
        self.error = None

    def update(self):
        self.updates += 1
        if self.error is not None:
            raise self.error

    def finish(self, *args, **kwargs):
        raise AssertionError("publisher must not finish the existing run")


class SDKDouble:
    def __init__(self, *runs):
        self.runs = runs
        self.api_calls = []
        self.run_paths = []
        self.error = None

    def Api(self, **kwargs):
        self.api_calls.append(kwargs)
        return self

    def run(self, path):
        self.run_paths.append(path)
        if self.error is not None:
            raise self.error
        return self.runs[min(len(self.run_paths) - 1, len(self.runs) - 1)]

    def init(self, *args, **kwargs):
        raise AssertionError("publisher must not create a run")


@pytest.mark.parametrize("framework", ["other-framework", "", False, 7, ["kev_laya"], {"name": "kev_laya"}])
def test_rejected_run_never_becomes_a_cached_write_target(telemetry, framework):
    run = RunDouble(framework)
    original_config = copy.deepcopy(run.config)
    sdk = SDKDouble(run)
    publisher = WandbPublisher(IDENTITY, enabled=True, sdk=sdk)
    for _ in range(3):
        with pytest.raises(ValueError, match="different framework"):
            publisher.publish(telemetry)
        assert publisher.run is None
    assert run.config == original_config
    assert run.updates == run.summary.assignments == run.summary.updates == 0
    assert sdk.run_paths == ["entity/project/existing-run"] * 3


def test_retry_retrieves_a_new_run_after_rejection(telemetry):
    rejected = RunDouble("other-framework")
    accepted = RunDouble("kev_laya")
    sdk = SDKDouble(rejected, accepted)
    publisher = WandbPublisher(IDENTITY, enabled=True, sdk=sdk)
    with pytest.raises(ValueError):
        publisher.publish(telemetry)
    assert publisher.publish(telemetry)
    assert publisher.run is accepted
    assert rejected.summary.assignments == 0
    assert accepted.summary.updates == 1
    assert len(sdk.run_paths) == 2


def test_setup_failure_does_not_skip_setup_on_retry(telemetry):
    broken = RunDouble()
    broken.error = OSError("simulated configuration write failure")
    recovered = RunDouble()
    sdk = SDKDouble(broken, recovered)
    publisher = WandbPublisher(IDENTITY, enabled=True, sdk=sdk)
    with pytest.raises(OSError):
        publisher.publish(telemetry)
    assert publisher.run is None
    assert broken.summary.assignments == 0
    assert publisher.publish(telemetry)
    assert publisher.run is recovered
    assert recovered.updates == recovered.summary.updates == 1
    assert len(sdk.run_paths) == 2


def test_every_setup_failure_requires_another_setup_attempt(telemetry):
    run = RunDouble()
    run.error = OSError("simulated configuration write failure")
    sdk = SDKDouble(run)
    publisher = WandbPublisher(IDENTITY, enabled=True, sdk=sdk)
    for _ in range(3):
        with pytest.raises(OSError):
            publisher.publish(telemetry)
    assert publisher.run is None
    assert run.updates == 3
    assert run.summary.assignments == run.summary.updates == 0


@pytest.mark.parametrize("framework", [None, "kev_laya"])
def test_valid_run_is_reused_without_lifecycle_operations(telemetry, framework):
    run = RunDouble(framework)
    sdk = SDKDouble(run)
    publisher = WandbPublisher(IDENTITY, enabled=True, sdk=sdk)
    before = copy.deepcopy(telemetry)
    for _ in range(3):
        assert publisher.publish(telemetry)
    assert sdk.api_calls == [{"timeout": 10}]
    assert sdk.run_paths == ["entity/project/existing-run"]
    assert run.updates == 1
    assert run.summary.updates == 3
    assert run.config == {"framework": "kev_laya", "unrelated": "preserve", "telemetry_schema_version": 1}
    assert telemetry == before


def test_cached_run_framework_is_checked_before_each_summary_write(telemetry):
    run = RunDouble()
    sdk = SDKDouble(run)
    publisher = WandbPublisher(IDENTITY, enabled=True, sdk=sdk)
    assert publisher.publish(telemetry)
    prior = copy.deepcopy(dict(run.summary))
    run.config["framework"] = "different-framework"
    for _ in range(2):
        with pytest.raises(ValueError, match="different framework"):
            publisher.publish(telemetry)
    assert dict(run.summary) == prior
    assert run.summary.assignments == run.summary.updates == 1
    assert run.config["framework"] == "different-framework"


def test_invalid_cached_config_cannot_skip_guard_after_error(telemetry):
    run = RunDouble()
    publisher = WandbPublisher(IDENTITY, enabled=True, sdk=SDKDouble(run))
    assert publisher.publish(telemetry)
    run.config = None
    for _ in range(2):
        with pytest.raises(TypeError):
            publisher.publish(telemetry)
    assert run.summary.updates == 1


def test_disabled_publisher_makes_no_sdk_calls(telemetry):
    sdk = SDKDouble(RunDouble("other-framework"))
    publisher = WandbPublisher(IDENTITY, enabled=False, sdk=sdk)
    for _ in range(3):
        assert publisher.publish(telemetry) is False
    assert publisher.run is None
    assert not sdk.api_calls and not sdk.run_paths


@pytest.mark.parametrize("cached", [False, True])
def test_wrong_snapshot_identity_fails_before_any_remote_operation(telemetry, cached):
    run = RunDouble()
    sdk = SDKDouble(run)
    publisher = WandbPublisher(IDENTITY, enabled=True, sdk=sdk)
    if cached:
        publisher.publish(telemetry)
    before = (len(sdk.api_calls), len(sdk.run_paths), run.updates, run.summary.updates)
    wrong = copy.deepcopy(telemetry)
    wrong["wandb"]["run_id"] = "another-run"
    with pytest.raises(ValueError, match="identity differs"):
        publisher.publish(wrong)
    assert before == (len(sdk.api_calls), len(sdk.run_paths), run.updates, run.summary.updates)


def test_run_lookup_failure_is_retryable_without_a_cached_target(telemetry):
    run = RunDouble()
    sdk = SDKDouble(run)
    sdk.error = OSError("lookup failure")
    publisher = WandbPublisher(IDENTITY, enabled=True, sdk=sdk)
    with pytest.raises(OSError):
        publisher.publish(telemetry)
    assert publisher.run is None
    sdk.error = None
    assert publisher.publish(telemetry)
    assert len(sdk.api_calls) == 2


def test_summary_failure_propagates_without_changing_run_ownership(telemetry):
    run = RunDouble()
    run.summary.error = OSError("summary failure")
    sdk = SDKDouble(run)
    publisher = WandbPublisher(IDENTITY, enabled=True, sdk=sdk)
    with pytest.raises(OSError):
        publisher.publish(telemetry)
    assert publisher.run is run
    run.summary.error = None
    assert publisher.publish(telemetry)
    assert len(sdk.api_calls) == run.updates == 1
    assert run.summary.updates == 2


def test_file_entry_point_preserves_target_and_content_boundaries(tmp_path, telemetry):
    path = tmp_path / "snapshot.json"
    raw = json.dumps(telemetry).encode()
    path.write_bytes(raw)
    run = RunDouble()
    sdk = SDKDouble(run)
    assert publish_wandb(path, entity=IDENTITY["entity"], project=IDENTITY["project"], sdk=sdk)
    assert path.read_bytes() == raw
    assert sdk.run_paths == ["entity/project/existing-run"]
    assert run.summary["kev_laya/snapshot"]["wandb"] == IDENTITY


@pytest.mark.parametrize("failure", ["framework", "setup", "summary"])
def test_publication_failures_do_not_abort_real_cpu_training(tmp_path, tiny, suite, failure):
    """Real local optimizer/checkpoint path, synthetic data and injected remote SDK."""
    run = RunDouble("other-framework" if failure == "framework" else None)
    marker = "PRIVATE-ERROR-PAYLOAD"
    if failure == "setup":
        run.error = OSError(marker)
    if failure == "summary":
        run.summary.error = OSError(marker)
    sdk = SDKDouble(run)
    data, manifest = suite
    with pytest.warns(RuntimeWarning) as caught:
        result = train(tiny, ByteTokenizer(), data["train"], manifest, tmp_path / "run",
                       TrainSettings(steps=3, save_every=1), ObjectiveConfig(), Limits(512, 8192),
                       wandb_ref=IDENTITY, publish_wandb=True, wandb_sdk=sdk)
    point = torch.load(result["checkpoint"], weights_only=True)
    snapshot = json.loads(Path(result["snapshot"]).read_text())
    assert point["training_state"]["step"] == 3
    assert point["training_state"]["wandb_identity"] == IDENTITY
    assert snapshot["wandb"] == IDENTITY
    assert snapshot["monitoring_export_failures"] >= 3
    assert marker not in " ".join(str(w.message) for w in caught)
    if failure in {"framework", "setup"}:
        assert run.summary.assignments == run.summary.updates == 0
