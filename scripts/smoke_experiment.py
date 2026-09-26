"""Measured miniature experiment; fixture evidence, never native replacement evidence."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "overwatch" / "added" / "src"))
from kev_laya.checkpoint import load_checkpoint, save_checkpoint
from kev_laya.data import freeze_smoke, load_suite
from kev_laya.encoding import ByteTokenizer, Limits
from kev_laya.evaluation import calibrate, evaluate
from kev_laya.io import atomic_json, sha256_file
from kev_laya.model import BackboneConfig, DecisionEngine
from kev_laya.objectives import ObjectiveConfig
from kev_laya.registration import register
from kev_laya.telemetry import TelemetryWriter, artifact
from kev_laya.training import TrainSettings, train

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--steps", type=int, default=160)
p.add_argument("--extension-steps", type=int, default=40)
p.add_argument("--choice-permutation", action="store_true")
a = p.parse_args()
if a.out.exists() and any(a.out.iterdir()):
    raise FileExistsError("experiment output must be empty")
a.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
freeze_smoke(a.out / "suite")
suite, manifest = load_suite(a.out / "suite")
# Freeze protocol BEFORE reading any evaluation results.
experiment_id = "miniature-permutation-regression-v2" if a.choice_permutation else "miniature-ablations-v1"
protocol = {"id": experiment_id, "seed": 42, "sft_steps": a.steps, "extension_steps": a.extension_steps,
            "synthetic_accuracy_threshold": 0.70, "choice_permutation": a.choice_permutation,
            "readout_status": "engineering regression on previously observed fixture; not fresh benchmark evidence" if a.choice_permutation else "first frozen fixture read",
            "choice_order_flip_threshold": 0.10, "evidence_class": "tiny-synthetic-fixture",
            "arms": ["supervised", "supervised-compute-matched", "added-pgps", "added-pgps-plus-temperature"],
            "not_evaluated": ["native long context", "general multilingual quality", "Kev baseline", "Laya baseline", "Jev", "paid deployments"]}
atomic_json(a.out / "protocol.json", protocol)
started = time.perf_counter()
torch.manual_seed(42)
model = DecisionEngine(BackboneConfig())
tokenizer, limits = ByteTokenizer(), Limits(branch=512, aggregate=8192, max_questions=64)
settings = TrainSettings(steps=a.steps, save_every=max(1, a.steps // 4),choice_permutation=a.choice_permutation)
base = train(model, tokenizer, suite["train"], manifest, a.out / "supervised", settings,
             ObjectiveConfig(ordinal=.1), limits, experiment_id=experiment_id, run_id="supervised")
reports = {"supervised": evaluate(model, suite["test"], tokenizer, limits, split="test")}
results = {"supervised": base}
for label, reward in (("supervised-compute-matched", 0.0), ("added-pgps", 0.1)):
    arm, tokenizer, point = load_checkpoint(Path(base["checkpoint"]))
    result = train(arm, tokenizer, suite["train"], manifest, a.out / label,
                   TrainSettings(steps=a.extension_steps, learning_rate=.001, save_every=a.extension_steps, seed=43,choice_permutation=a.choice_permutation),
                   ObjectiveConfig(ordinal=.1, reinforce=reward), limits,
                   experiment_id=experiment_id, run_id=label, parent_sha256=point["checkpoint_sha256"])
    reports[label] = evaluate(arm, suite["test"], tokenizer, limits, split="test")
    results[label] = result
    if reward:
        calibration = calibrate(arm, suite["calibration"], tokenizer, limits, split="calibration",
                                split_sha256=manifest["partitions"]["calibration"]["sha256"])
        calibrated = a.out / label / "calibrated.pt"
        trained_point = load_checkpoint(Path(result["checkpoint"]))[2]
        save_checkpoint(calibrated, arm, tokenizer, provenance=trained_point["provenance"],
                         parent_sha256=sha256_file(Path(result["checkpoint"])),
                         parent_checkpoint=Path(result["checkpoint"]), exposure_operation="calibration")
        reports["added-pgps-plus-temperature"] = evaluate(arm, suite["test"], tokenizer, limits, split="test")
        atomic_json(a.out / label / "calibration.json", calibration)
        results["calibrated_checkpoint"] = str(calibrated)
for name, report in reports.items():
    atomic_json(a.out / (name + "-evaluation.json"), report)
for name, result in [(k, v) for k, v in results.items() if isinstance(v, dict)]:
    report = reports[name]
    # Add artifacts using the same exporter; do not write into the collector cache here.
    snapshot_path = Path(result["snapshot"])
    snapshot = json.loads(snapshot_path.read_text())
    exporter = TelemetryWriter(snapshot_path, snapshot)
    summary = report["summary"]
    metrics = dict(snapshot["metrics"]) | {"validation/" + k: float(summary[k]) for k in ("nll", "brier", "ece", "accuracy")}
    exporter.update(phase="completed", metrics=metrics, artifacts=snapshot["artifacts"] + [artifact(a.out / (name + "-evaluation.json"), "evaluation")])
    register(a.out / "registry.json", snapshot_path, name)
# Real run exports -> actual new provider -> pure adapter -> isolated fixture cache -> presentation.
# This is NOT the unavailable original Overwatch report_service/front-end build.
from overwatch.providers.kev_laya import collect_snapshots
from overwatch.kev_laya_adapter import merge_snapshots, presentation
collected = collect_snapshots(a.out / "registry.json")
envelope = merge_snapshots({}, collected)
cache = a.out / "isolated-collector-cache" / "runs-v2.json"
atomic_json(cache, envelope)
presented = presentation(json.loads(cache.read_text()))
atomic_json(a.out / "adapter-presentation.json", presented)
result = {"protocol_sha256": sha256_file(a.out / "protocol.json"), "elapsed_seconds": time.perf_counter() - started,
          "summaries": {k: v["summary"] | {"reliability": "see full report", "risk_coverage": "see full report"} for k, v in reports.items()},
          "fixture_threshold_pass": reports["added-pgps-plus-temperature"]["summary"]["accuracy"] >= .70,
          "training_results": results, "adapter_ingested_runs": len(envelope["records"]), "adapter_presented_runs": len(presented["runs"]),
          "adapter_warnings": presented["warnings"], "original_overwatch_end_to_end": "NOT RUN: primary checkout and supported dependencies not available",
          "native_model_context": "NOT RUN", "actual_cost_usd": None, "cost_reason": "CPU runtime measured; no billing/power meter attached"}
atomic_json(a.out / "experiment-report.json", result)
print(json.dumps({k: v for k, v in result.items() if k != "training_results"}, indent=2))
