import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kev_laya.checkpoint import load_checkpoint
from kev_laya.cli import main
from kev_laya.data import freeze_smoke
from kev_laya.encoding import ByteTokenizer, Limits
from kev_laya.objectives import ObjectiveConfig
from kev_laya.quality_budget import BudgetExceeded, StepTokenBudget, configure_cuda_allocator_limit
from kev_laya.training import TrainSettings, train


def test_aggregate_token_cap_rejects_the_crossing_step(tmp_path):
    path = tmp_path / "budget.json"
    with StepTokenBudget(path, b"frozen protocol", max_steps=3, max_useful_tokens=100) as budget:
        budget.verify_run("seed-1", steps=0, useful_tokens=0)
        budget.reserve_step("seed-1", useful_tokens=20)
        budget.commit_step("seed-1", useful_tokens=20)
        with pytest.raises(BudgetExceeded, match="useful_token_budget_exceeded"):
            budget.reserve_step("seed-2", useful_tokens=81)
        assert budget.state["steps"] == 1
        assert budget.state["useful_tokens"] == 20
    with StepTokenBudget(path, b"frozen protocol", max_steps=3, max_useful_tokens=100) as reopened:
        reopened.verify_run("seed-1", steps=1, useful_tokens=20)
        reopened.verify_run("seed-2", steps=0, useful_tokens=0)
        assert json.loads(path.read_text())["pending"] is None


def test_pending_step_and_competing_owner_fail_closed(tmp_path):
    path = tmp_path / "budget.json"
    with StepTokenBudget(path, b"one", max_steps=2, max_useful_tokens=30) as budget:
        with pytest.raises(OSError):
            StepTokenBudget(path, b"one", max_steps=2, max_useful_tokens=30)
        budget.reserve_step("run", useful_tokens=10)
    with pytest.raises(BudgetExceeded, match="in_flight_step"):
        StepTokenBudget(path, b"one", max_steps=2, max_useful_tokens=30)
    with pytest.raises(BudgetExceeded, match="identity"):
        StepTokenBudget(path, b"two", max_steps=2, max_useful_tokens=30)


def test_one_run_owns_each_seed_arm_allocation(tmp_path):
    with StepTokenBudget(tmp_path / "budget.json", b"frozen protocol", max_steps=4,
                         max_useful_tokens=100, max_run_steps=2,
                         max_run_useful_tokens=25) as budget:
        budget.claim_allocation("supervised:19", "run-a")
        budget.claim_allocation("supervised:19", "run-a")
        with pytest.raises(BudgetExceeded, match="already_owned"):
            budget.claim_allocation("supervised:19", "run-b")
        budget.reserve_step("run-a", useful_tokens=20)
        budget.commit_step("run-a", useful_tokens=20)
        with pytest.raises(BudgetExceeded, match="run_useful_token_budget_exceeded"):
            budget.reserve_step("run-a", useful_tokens=6)
        budget.claim_allocation("reward:19", "run-b")
        budget.reserve_step("run-b", useful_tokens=6)
        budget.commit_step("run-b", useful_tokens=6)


def test_wall_deadline_persists_across_runs_and_rejects_backward_clock(tmp_path):
    clock = [100.0]
    path = tmp_path / "budget.json"
    kwargs = dict(max_steps=3, max_useful_tokens=100, max_wall_seconds=5, now=lambda: clock[0])
    with StepTokenBudget(path, b"frozen protocol", **kwargs) as budget:
        budget.reserve_step("run-a", useful_tokens=10)
        budget.commit_step("run-a", useful_tokens=10)
    clock[0] = 99.0
    with StepTokenBudget(path, b"frozen protocol", **kwargs) as budget:
        with pytest.raises(BudgetExceeded, match="wall_clock_reversed"):
            budget.reserve_step("run-b", useful_tokens=10)
    clock[0] = 105.0
    with StepTokenBudget(path, b"frozen protocol", **kwargs) as budget:
        with pytest.raises(BudgetExceeded, match="wall_budget_exceeded"):
            budget.reserve_step("run-b", useful_tokens=10)
        assert budget.state["steps"] == 1
        assert budget.state["pending"] is None


def test_measured_gpu_peak_violation_keeps_pending_step_for_audit(tmp_path):
    path = tmp_path / "budget.json"
    kwargs = dict(max_steps=2, max_useful_tokens=100, max_peak_gpu_memory_bytes=100)
    with StepTokenBudget(path, b"frozen protocol", **kwargs) as budget:
        budget.reserve_step("run", useful_tokens=10, peak_gpu_bytes=60)
        with pytest.raises(BudgetExceeded, match="gpu_memory_budget_exceeded"):
            budget.commit_step("run", useful_tokens=10, peak_gpu_bytes=101)
        assert budget.state["pending"] == {"run_id": "run", "useful_tokens": 10}
    with pytest.raises(BudgetExceeded, match="in_flight_step"):
        StepTokenBudget(path, b"frozen protocol", **kwargs)


def test_cuda_allocator_cap_uses_declared_bytes_before_model_load(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(total_memory=400))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device: calls.append(("cap", fraction, str(device))))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                        lambda device: calls.append(("reset", str(device))))
    result = configure_cuda_allocator_limit("cuda:0", 100)
    assert result["allocator_fraction"] == 0.25
    assert calls == [("cap", 0.25, "cuda:0"), ("reset", "cuda:0")]
    assert configure_cuda_allocator_limit("cpu", 100)["allocator_cap_applied"] is False


def test_training_stops_before_next_step_and_checkpoint_resumes_with_same_ledger(tmp_path, tiny, suite):
    data, manifest = suite
    settings = TrainSettings(steps=3, accumulation=1, save_every=3, seed=19)
    path = tmp_path / "budget.json"
    with StepTokenBudget(path, b"frozen protocol", max_steps=1, max_useful_tokens=100000) as budget:
        first = train(tiny, ByteTokenizer(), data["train"], manifest, tmp_path / "run",
                      settings, ObjectiveConfig(), Limits(512, 8192), quality_budget=budget)
        assert first["state"]["step"] == 1
        assert first["quality_budget_stop_reason"] == "step_budget_exceeded"
        assert Path(first["checkpoint"]).name == "checkpoint-000001.pt"
        assert budget.state["useful_tokens"] == first["state"]["forward_tokens"]
        weights = copy.deepcopy(tiny.state_dict())
    resumed, tokenizer, _ = load_checkpoint(Path(first["checkpoint"]))
    with StepTokenBudget(path, b"frozen protocol", max_steps=1, max_useful_tokens=100000) as budget:
        second = train(resumed, tokenizer, data["train"], manifest, tmp_path / "run",
                       settings, ObjectiveConfig(), Limits(512, 8192),
                       resume=Path(first["checkpoint"]), quality_budget=budget)
        assert second["state"]["step"] == 1
        assert second["quality_budget_stop_reason"] == "step_budget_exceeded"
        assert budget.state["steps"] == 1
    for name, value in weights.items():
        torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)


def test_choice_permutation_lookahead_charges_actual_forward_tokens(tmp_path, tiny, suite):
    data, manifest = suite
    with StepTokenBudget(tmp_path / "budget.json", b"frozen protocol",
                         max_steps=2, max_useful_tokens=100000) as budget:
        result = train(tiny, ByteTokenizer(), data["train"], manifest, tmp_path / "run",
                       TrainSettings(steps=2, accumulation=2, choice_permutation=True),
                       ObjectiveConfig(), Limits(512, 8192), quality_budget=budget)
        assert result["state"]["step"] == budget.state["steps"] == 2
        assert result["state"]["forward_tokens"] == budget.state["useful_tokens"]
        assert budget.state["pending"] is None


def test_token_cap_below_first_step_prevents_optimizer_work(tmp_path, tiny, suite):
    data, manifest = suite
    before = copy.deepcopy(tiny.state_dict())
    with StepTokenBudget(tmp_path / "budget.json", b"frozen protocol",
                         max_steps=3, max_useful_tokens=1) as budget:
        result = train(tiny, ByteTokenizer(), data["train"], manifest, tmp_path / "run",
                       TrainSettings(steps=3, accumulation=1), ObjectiveConfig(),
                       Limits(512, 8192), quality_budget=budget)
        assert result["quality_budget_stop_reason"] == "useful_token_budget_exceeded"
        assert result["state"]["step"] == budget.state["steps"] == 0
        assert budget.state["pending"] is None
    for name, value in before.items():
        torch.testing.assert_close(value, tiny.state_dict()[name], atol=0, rtol=0)


def test_wall_deadline_during_forward_fails_closed_before_optimizer(tmp_path, tiny, suite, monkeypatch):
    data, manifest = suite
    clock = [0.0]
    before = copy.deepcopy(tiny.state_dict())
    original_forward = tiny.forward
    def expire_during_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        clock[0] = 5.0
        return result
    monkeypatch.setattr(tiny, "forward", expire_during_forward)
    path = tmp_path / "budget.json"
    with StepTokenBudget(path, b"frozen protocol", max_steps=2,
                         max_useful_tokens=100000, max_wall_seconds=5,
                         now=lambda: clock[0]) as budget:
        with pytest.raises(BudgetExceeded, match="wall_budget_exceeded"):
            train(tiny, ByteTokenizer(), data["train"], manifest, tmp_path / "run",
                  TrainSettings(steps=2, accumulation=1), ObjectiveConfig(),
                  Limits(512, 8192), quality_budget=budget)
        assert budget.state["steps"] == 0
        assert budget.state["pending"]["run_id"]
    with pytest.raises(BudgetExceeded, match="in_flight_step"):
        StepTokenBudget(path, b"frozen protocol", max_steps=2,
                        max_useful_tokens=100000, max_wall_seconds=5)
    for name, value in before.items():
        torch.testing.assert_close(value, tiny.state_dict()[name], atol=0, rtol=0)


def test_expired_wall_budget_stops_before_first_step(tmp_path, tiny, suite):
    data, manifest = suite
    clock = [0.0]
    with StepTokenBudget(tmp_path / "budget.json", b"frozen protocol", max_steps=2,
                         max_useful_tokens=100000, max_wall_seconds=5,
                         now=lambda: clock[0]) as budget:
        clock[0] = 5.0
        result = train(tiny, ByteTokenizer(), data["train"], manifest, tmp_path / "run",
                       TrainSettings(steps=2), ObjectiveConfig(), Limits(512, 8192),
                       quality_budget=budget)
        assert result["quality_budget_stop_reason"] == "wall_budget_exceeded"
        assert result["state"]["step"] == budget.state["steps"] == 0
        assert budget.state["pending"] is None


def test_quality_cli_wires_frozen_budget_into_training(tmp_path, tiny, monkeypatch):
    suite_path = tmp_path / "suite"
    freeze_smoke(suite_path, count=64)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"backbone": tiny.config_dict(),
        "training": {"steps": 1, "seed": 19, "accumulation": 1},
        "limits": {"branch": 512, "aggregate": 8192}}))
    protocol_path = tmp_path / "protocol.json"
    review_path = tmp_path / "injected-review.json"
    review_path.write_text('{"test_only":true}')
    protocol_path.write_text(json.dumps({"seeds": [19],
        "training": {"arms": {arm: {"optimizer_steps": 1, "useful_token_budget": 100000}
                               for arm in ("supervised", "reward")}},
        "budget": {"max_total_optimizer_steps": 2, "max_wall_seconds": 120,
                   "max_peak_gpu_memory_bytes": 1000000000},
        "data_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "suite_manifest_sha256": hashlib.sha256((suite_path / "manifest.json").read_bytes()).hexdigest()}))
    from kev_laya import cli
    monkeypatch.setattr(cli, "preflight", lambda *_: {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()})
    ledger = tmp_path / "budget.json"
    order = []
    original_engine = cli.DecisionEngine
    def checked_engine(config):
        assert order == ["allocator_before_model"]
        return original_engine(config)
    monkeypatch.setattr(cli, "configure_cuda_allocator_limit",
                        lambda *_: order.append("allocator_before_model") if ledger.exists()
                        else pytest.fail("budget ledger was not acquired before model loading"))
    monkeypatch.setattr(cli, "DecisionEngine", checked_engine)
    assert main(["train", "--suite", str(suite_path), "--config", str(config_path),
                 "--out", str(tmp_path / "run"), "--run-id", "test-run",
                 "--quality-protocol", str(protocol_path),
                 "--quality-data-review", str(review_path),
                 "--quality-budget-ledger", str(ledger)]) == 0
    state = json.loads(ledger.read_text())
    assert state["steps"] == 1 and state["useful_tokens"] > 0
    assert state["allocations"] == {"supervised:19": "test-run"}
    assert state["pending"] is None
