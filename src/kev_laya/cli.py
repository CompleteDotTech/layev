"""Reproducible local training, reward, calibration, evaluation and serving stages."""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import os
import time
from contextlib import nullcontext
import torch
from .checkpoint import load_checkpoint, save_checkpoint
from .data import freeze_smoke, load_suite
from .encoding import ByteTokenizer, QwenTokenizer, Limits
from .evaluation import calibrate, evaluate
from .io import atomic_json, sha256_file
from .model import BackboneConfig, DecisionEngine
from .execution import BatchPolicy
from .objectives import ObjectiveConfig
from .schema import strict_loads
from .training import TrainSettings, train
from .quality_budget import StepTokenBudget, configure_cuda_allocator_limit
from .quality_protocol import KNOWN_FIXTURE_IDS, preflight


def limits_for(payload: dict) -> Limits:
    recorded = payload.get("provenance", {}).get("context_limits")
    return Limits(**recorded) if recorded else Limits(branch=payload["config"]["max_position_embeddings"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kev-laya")
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze-smoke")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=256)
    p = sub.add_parser("native-init")
    p.add_argument("--backbone-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--checkpointing", action="store_true")
    for command in ("train", "reward-train"):
        p = sub.add_parser(command)
        p.add_argument("--suite", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--init", type=Path)
        p.add_argument("--resume", type=Path)
        p.add_argument("--stop-after", type=int)
        p.add_argument("--device", default="cpu")
        p.add_argument("--experiment-id", help="New-run experiment ID; resume inherits it unless explicitly checked")
        p.add_argument("--run-id")
        p.add_argument("--wandb-entity")
        p.add_argument("--wandb-project")
        p.add_argument("--wandb-run-id")
        p.add_argument("--wandb-publish", action="store_true", help="Opt-in summary/config update to an existing W&B run; no lifecycle ownership")
        p.add_argument("--quality-protocol", type=Path, help="Frozen representative-quality protocol")
        p.add_argument("--quality-data-review", type=Path, help="Reviewed data-use declaration")
        p.add_argument("--quality-budget-ledger", type=Path, help="Exclusive durable aggregate budget ledger")
    for command in ("calibrate", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--suite", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--telemetry", type=Path)
        p.add_argument("--experiment-id", default="local-stages")
        p.add_argument("--run-id")
        if command == "evaluate":
            p.add_argument("--split", choices=("development", "test"), default="development")
            p.add_argument("--quality-protocol", type=Path)
            p.add_argument("--quality-data-review", type=Path)
            p.add_argument("--quality-selection", type=Path)
            p.add_argument("--development-report", type=Path)
            p.add_argument("--calibrated-checkpoint", type=Path)
            p.add_argument("--quality-readout-ledger", type=Path)
            p.add_argument("--diagnostic-test", action="store_true",
                           help="Explicit untracked fixture/debug test; no representative quality claim")
    p = sub.add_parser("serve")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8009)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-auth", action="store_true")
    p.add_argument("--batch-policy", type=Path, help="Inference-only override; weights and model identity are unchanged")
    p.add_argument("--stable-alias", action="store_true", help="Operator promotion, not a quality certification")
    p.add_argument("--telemetry", type=Path, help="Optional content-free serving snapshot; explicit independent run")
    p.add_argument("--experiment-id", default="local-serving")
    p.add_argument("--run-id")
    p.add_argument("--requests-per-minute", type=int, default=120)
    p.add_argument("--tokens-per-minute", type=int, default=262144)
    p = sub.add_parser("verify-exposure")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--parent", type=Path, action="append", default=[])
    p.add_argument("--expected-sha256")
    p.add_argument("--out", type=Path)
    p = sub.add_parser("register")
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--source-id", required=True)
    p = sub.add_parser("reconcile-test-readout")
    p.add_argument("--quality-protocol", type=Path, required=True)
    p.add_argument("--quality-readout-ledger", type=Path, required=True)
    p.add_argument("--arm", choices=("supervised", "reward"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "freeze-smoke":
        print(json.dumps(freeze_smoke(args.out, args.count), indent=2))
    elif args.command == "native-init":
        # Fail cheap prerequisites before architecture/weight allocation.
        if (args.out.exists() or args.out.is_symlink()
                or args.out.with_suffix(".manifest.json").exists()
                or args.out.with_suffix(".manifest.json").is_symlink()):
            raise FileExistsError(args.out)
        src = args.backbone_dir
        provenance = strict_loads((src / "source.json").read_bytes())
        if not isinstance(provenance, dict) or provenance.get("repository") != "Qwen/Qwen2.5-0.5B" or provenance.get("revision") != "060db6499f32faf8b98477b0a26969ef7d8b9987":
            raise ValueError("backbone provenance is not the frozen revision")
        digests = provenance.get("sha256")
        if not isinstance(digests, dict) or set(digests) != {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "LICENSE"}:
            raise ValueError("complete pinned-backbone file manifest required")
        import re
        if any(not isinstance(d, str) or re.fullmatch(r"[0-9a-f]{64}", d) is None
               for d in digests.values()):
            raise ValueError("backbone manifest requires lowercase SHA-256 values")
        # Retain config bytes from the same read that was verified. Tokenizer
        # construction similarly binds its own single read to its identity.
        config_raw = (src / "config.json").read_bytes()
        if hashlib.sha256(config_raw).hexdigest() != digests["config.json"]:
            raise ValueError("backbone file checksum mismatch")
        for file in ("tokenizer_config.json", "LICENSE"):
            if sha256_file(src / file) != digests[file]:
                raise ValueError("backbone file checksum mismatch")
        tokenizer = QwenTokenizer(src / "tokenizer.json")
        if tokenizer.identity != "qwen-tokenizer-sha256:" + digests["tokenizer.json"]:
            raise ValueError("backbone tokenizer checksum mismatch")
        config_document = strict_loads(config_raw)
        if not isinstance(config_document, dict):
            raise ValueError("native backbone config must be a JSON object")
        config = BackboneConfig.from_qwen(config_document,
                    source=provenance["repository"], revision=provenance["revision"], lora_rank=args.lora_rank,
                    activation_checkpointing=args.checkpointing)
        if max(tokenizer.special) >= config.vocab_size or tokenizer.metadata()["vocab_size"] > config.vocab_size:
            raise ValueError("tokenizer vocabulary does not fit native embeddings")
        model = DecisionEngine(config)
        model.load_qwen_weights(src / "model.safetensors", expected_sha256=digests["model.safetensors"])
        model.backbone_source = provenance
        # Recheck before export; exclusive output ownership is still required.
        if (args.out.exists() or args.out.is_symlink()
                or args.out.with_suffix(".manifest.json").exists()
                or args.out.with_suffix(".manifest.json").is_symlink()):
            raise FileExistsError(args.out)
        result = save_checkpoint(args.out, model, tokenizer,
                                 provenance={"backbone_source": provenance, "context_limits": {"branch": 32768, "aggregate": 65536}})
        print(json.dumps(result, indent=2))
    elif args.command in {"train", "reward-train"}:
        quality_paths = (args.quality_protocol, args.quality_data_review, args.quality_budget_ledger)
        if any(path is not None for path in quality_paths) and not all(path is not None for path in quality_paths):
            raise ValueError("quality protocol, data review and budget ledger must be supplied together")
        quality_protocol = None
        quality_raw = None
        if all(path is not None for path in quality_paths):
            receipt = preflight(args.quality_protocol, args.quality_data_review, args.suite)
            quality_raw = args.quality_protocol.read_bytes()
            if hashlib.sha256(quality_raw).hexdigest() != receipt["protocol_sha256"]:
                raise ValueError("quality protocol changed after preflight")
            quality_protocol = strict_loads(quality_raw)
        cfg = strict_loads(args.config.read_bytes())
        suite, manifest = load_suite(args.suite)
        settings = TrainSettings(**cfg.get("training", {}))
        if quality_protocol is not None:
            if args.run_id is None and args.resume is None:
                raise ValueError("quality run requires an explicit run ID")
            if settings.seed not in quality_protocol["seeds"]:
                raise ValueError("quality seed is absent from the frozen protocol")
            arm = "supervised" if args.command == "train" else "reward"
            arm_budget = quality_protocol["training"]["arms"][arm]
            if settings.steps != arm_budget["optimizer_steps"]:
                raise ValueError("quality optimizer steps differ from the frozen protocol")
            if sha256_file(args.quality_data_review) != quality_protocol["data_review_sha256"]:
                raise ValueError("quality data review changed after preflight")
            if sha256_file(args.suite / "manifest.json") != quality_protocol["suite_manifest_sha256"]:
                raise ValueError("quality suite changed after preflight")
        if quality_protocol is None:
            budget_context = nullcontext(None)
        else:
            arms = quality_protocol["training"]["arms"]
            budget_context = StepTokenBudget(args.quality_budget_ledger, quality_raw,
                max_steps=quality_protocol["budget"]["max_total_optimizer_steps"],
                max_useful_tokens=len(quality_protocol["seeds"]) *
                    sum(value["useful_token_budget"] for value in arms.values()),
                max_run_steps=arm_budget["optimizer_steps"],
                max_run_useful_tokens=arm_budget["useful_token_budget"],
                max_wall_seconds=quality_protocol["budget"]["max_wall_seconds"],
                max_peak_gpu_memory_bytes=quality_protocol["budget"]["max_peak_gpu_memory_bytes"])
        with budget_context as quality_budget:
            if quality_protocol is not None:
                configure_cuda_allocator_limit(
                    args.device, quality_protocol["budget"]["max_peak_gpu_memory_bytes"])
            torch.manual_seed(settings.seed)
            initial = args.resume or args.init
            if initial:
                model, tokenizer, point = load_checkpoint(initial, args.device)
            else:
                if "backbone" not in cfg:
                    raise ValueError("native training requires an initialized checkpoint")
                model = DecisionEngine(BackboneConfig(**cfg["backbone"])).to(args.device)
                tokenizer, point = ByteTokenizer(), {}
            if "execution" in cfg:
                model.batch_policy = BatchPolicy.from_dict(cfg["execution"])
            loss_cfg = ObjectiveConfig(**cfg.get("objective", {}))
            if args.command == "reward-train" and loss_cfg.reinforce <= 0:
                raise ValueError("reward-train requires a positive reinforce coefficient")
            scheduler_ref = None
            if os.environ.get("KEV_LAYA_SCHEDULER_NAMESPACE"):
                scheduler_ref = {"provider": "skypilot", "namespace": os.environ["KEV_LAYA_SCHEDULER_NAMESPACE"],
                                 "job_id": int(os.environ["KEV_LAYA_JOB_ID"])}
            supplied_wandb = (args.wandb_entity, args.wandb_project, args.wandb_run_id)
            if any(supplied_wandb) and not all(supplied_wandb):
                raise ValueError("all W&B identity components are required")
            wandb_ref = dict(zip(("entity", "project", "run_id"), supplied_wandb)) if all(supplied_wandb) else None
            parent_sha256 = point.get("checkpoint_sha256") if not args.resume else None
            del point  # release deserialized weights and optimizer state before training
            result = train(model, tokenizer, suite["train"], manifest, args.out, settings,
                           loss_cfg, Limits(**cfg.get("limits", {})), resume=args.resume, stop_after=args.stop_after,
                           experiment_id=args.experiment_id, run_id=args.run_id, scheduler_ref=scheduler_ref,
                           wandb_ref=wandb_ref, publish_wandb=args.wandb_publish,
                           parent_sha256=parent_sha256,
                           quality_budget=quality_budget,
                           quality_allocation_id=f"{arm}:{settings.seed}" if quality_protocol is not None else None)
        print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
    elif args.command in {"calibrate", "evaluate"}:
        if args.command == "evaluate":
            quality_paths = (args.quality_protocol, args.quality_data_review,
                             args.quality_selection, args.development_report,
                             args.calibrated_checkpoint, args.quality_readout_ledger)
            if args.split == "test" and not args.diagnostic_test:
                if not all(path is not None for path in quality_paths):
                    raise ValueError("test evaluation requires the paired quality readout claim")
                from .quality_readout import run_paired_test
                report = run_paired_test(protocol_path=args.quality_protocol,
                    review_path=args.quality_data_review, selection_path=args.quality_selection,
                    development_report_path=args.development_report,
                    raw_checkpoint=args.checkpoint, calibrated_checkpoint=args.calibrated_checkpoint,
                    suite_path=args.suite, ledger_path=args.quality_readout_ledger,
                    output=args.out, device=args.device)
                print(json.dumps({name: value["summary"] for name, value in report["readouts"].items()}, indent=2))
                return 0
            if args.diagnostic_test and (args.split != "test" or any(path is not None for path in quality_paths)):
                raise ValueError("diagnostic test cannot be combined with a quality readout")
            if args.diagnostic_test:
                fixture_manifest = strict_loads((args.suite / "manifest.json").read_bytes())
                if fixture_manifest.get("id") not in KNOWN_FIXTURE_IDS:
                    raise ValueError("diagnostic test requires a known synthetic fixture")
            if args.split == "development" and any(path is not None for path in quality_paths):
                raise ValueError("quality readout arguments require the test split")
        suite, manifest = load_suite(args.suite)
        model, tokenizer, point = load_checkpoint(args.checkpoint, args.device)
        limits = limits_for(point)
        if args.out.exists():
            raise FileExistsError(args.out)
        from .stages import stage_writer
        from .telemetry import artifact
        writer = stage_writer(args.telemetry or args.out.with_suffix(".telemetry.json"), args.command,
                              model, tokenizer, point, limits, manifest=manifest,
                              experiment_id=args.experiment_id, run_id=args.run_id)
        if args.command == "evaluate" and args.diagnostic_test:
            provenance = dict(writer.snapshot["provenance"])
            provenance["evidence_class"] = "untracked-diagnostic-test-ineligible-for-quality"
            writer.update(provenance=provenance)
        started = time.perf_counter()
        try:
            if args.command == "calibrate":
                fitted = calibrate(model, suite["calibration"], tokenizer, limits, split="calibration",
                                   split_sha256=manifest["partitions"]["calibration"]["sha256"])
                # Inference-only: old optimizer does not describe the new temperature fit.
                save_checkpoint(args.out, model, tokenizer, provenance=writer.snapshot["provenance"], parent_sha256=point["checkpoint_sha256"],
                                parent_checkpoint=args.checkpoint, exposure_operation="calibration")
                calibration_path=args.out.with_suffix(".calibration.json")
                atomic_json(calibration_path, fitted)
                from .provenance import compact_calibration
                extension = dict(writer.snapshot["extensions"], calibration=compact_calibration(fitted))
                writer.update(phase="completed", extensions=extension, artifacts=[artifact(args.out,"checkpoint",point["checkpoint_sha256"]),
                              artifact(calibration_path,"evaluation")])
                print(json.dumps(fitted, indent=2))
            else:
                report = evaluate(model, suite[args.split], tokenizer, limits, split=args.split,
                                  diagnostic=args.diagnostic_test)
                report.update(checkpoint_sha256=point["checkpoint_sha256"], model_id=point["model_id"],
                              split_sha256=manifest["partitions"][args.split]["sha256"])
                atomic_json(args.out, report)
                summary=report["summary"]
                writer.update(phase="completed", metrics={"validation/"+k:summary[k] for k in ("nll","brier","ece","accuracy")},
                              artifacts=[artifact(args.out,"evaluation")])
                print(json.dumps(report["summary"], indent=2))
        except BaseException:
            writer.update(phase="failed")
            raise
        finally:
            progress=dict(writer.snapshot["progress"])
            progress["elapsed_seconds"]=time.perf_counter()-started
            writer.update(progress=progress)
            writer.close()
    elif args.command == "verify-exposure":
        from .exposure import verify_exposure
        result = verify_exposure(args.checkpoint, parents=args.parent, expected_sha256=args.expected_sha256)
        if args.out:
            if args.out.exists():
                raise FileExistsError(args.out)
            atomic_json(args.out, result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "verified" else 2
    elif args.command == "register":
        from .registration import register
        print(json.dumps(register(args.registry, args.snapshot, args.source_id), indent=2))
    elif args.command == "reconcile-test-readout":
        from .quality_readout import QualityReadoutLedger
        protocol_raw = args.quality_protocol.read_bytes()
        protocol = strict_loads(protocol_raw)
        if args.arm not in protocol["training"]["arms"] or args.seed not in protocol["seeds"]:
            raise ValueError("readout arm/seed absent from frozen protocol")
        with QualityReadoutLedger(args.quality_readout_ledger,
                protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),
                suite_manifest_sha256=protocol["suite_manifest_sha256"]) as ledger:
            ledger.reconcile_output(f"{args.arm}:{args.seed}", args.out)
        print(json.dumps({"status": "existing_output_reconciled", "arm": args.arm,
                          "seed": args.seed, "output_sha256": sha256_file(args.out)}, indent=2))
    elif args.command == "serve":
        from .service import InferenceRuntime, ServeSettings, create_app
        import uvicorn
        if args.no_auth and args.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("unauthenticated serving is restricted to loopback")
        model, tokenizer, point = load_checkpoint(args.checkpoint, args.device)
        if args.batch_policy:
            model.batch_policy = BatchPolicy.from_dict(strict_loads(args.batch_policy.read_bytes()))
        keys = tuple(strict_loads(os.environ.get("KEV_LAYA_API_KEYS", "[]")))
        settings = ServeSettings(api_keys=keys, allow_unauthenticated=args.no_auth,
                                 requests_per_minute=args.requests_per_minute, input_tokens_per_minute=args.tokens_per_minute)
        runtime = InferenceRuntime(model, tokenizer, point["model_id"], limits_for(point), stable=args.stable_alias)
        from .telemetry import ServingHook
        from .stages import stage_writer
        writer = stage_writer(args.telemetry, "serve", model, tokenizer, point, limits_for(point),
                               experiment_id=args.experiment_id, run_id=args.run_id) if args.telemetry else None
        try:
            from .provenance import ResourceSampler
            hook = ServingHook(writer, resource_sampler=ResourceSampler(device=next(model.parameters()).device))
            uvicorn.run(create_app(runtime, settings, hook), host=args.host, port=args.port, access_log=False)
        finally:
            if writer:
                writer.update(phase="completed")
                writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
