"""Reproducible supervised and PGPS optimization with complete resume state."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import copy
import hashlib
import math
import random
import uuid
import torch
from .checkpoint import load_checkpoint, restore_rng, save_checkpoint
from .data import Datum, Sampler, permute_choices
from .encoding import Limits, preprocessing_identity
from .io import atomic_json
from .model import DecisionEngine
from .objectives import ObjectiveConfig, objective
from .quality_budget import BudgetExceeded, StepTokenBudget
from .schema import canonical
from .telemetry import TelemetryWriter, artifact, new_snapshot


@dataclass(frozen=True)
class TrainSettings:
    steps: int = 160
    accumulation: int = 2
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    seed: int = 42
    gradient_clip: float = 1.0
    save_every: int = 40
    keep_checkpoints: int = 3
    precision: str = "fp32"
    choice_permutation: bool = False
    resource_sample_seconds: float = 1.0
    def __post_init__(self):
        if min(self.steps, self.accumulation, self.save_every, self.keep_checkpoints) < 1:
            raise ValueError("invalid training counts")
        if not all(math.isfinite(v) for v in (self.learning_rate, self.weight_decay, self.gradient_clip)) or self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("invalid optimizer settings")
        if not math.isfinite(self.resource_sample_seconds) or not .01 <= self.resource_sample_seconds <= 3600:
            raise ValueError("invalid resource sampling interval")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("unsupported precision")


def train(model: DecisionEngine, tokenizer, data: list[Datum], manifest: dict, output: Path,
          settings: TrainSettings, loss_config: ObjectiveConfig, limits: Limits, *,
          resume: Path | None = None, stop_after: int | None = None,
          experiment_id: str | None = None, run_id: str | None = None,
          scheduler_ref: dict | None = None, wandb_ref: dict | None = None,
          parent_sha256: str | None = None, publish_wandb: bool = False, wandb_sdk=None,
          source_root: Path | None = None, quality_budget: StepTokenBudget | None = None,
          quality_allocation_id: str | None = None) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("output is not empty; use resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    recorded_settings = asdict(settings)
    if not settings.choice_permutation:
        recorded_settings.pop("choice_permutation")  # preserve the original v1 no-augmentation config hash
    cfg = {"settings": recorded_settings, "objective": asdict(loss_config), "limits": asdict(limits),
           "backbone": model.config_dict(), "preprocessing": preprocessing_identity(tokenizer),
           "execution": model.batch_policy.to_dict()}
    data_hash = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    config_hash = hashlib.sha256(canonical(cfg).encode()).hexdigest()
    if limits.branch > model.cfg.max_position_embeddings:
        raise ValueError("declared branch budget exceeds actual backbone window")
    device = next(model.parameters()).device
    if settings.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 training profile requires CUDA; no silent fallback")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: max(0.05, 1 - n / settings.steps))
    sampler = Sampler(len(data), settings.seed)
    state = {"step": 0, "microbatches": 0, "examples": 0, "forward_tokens": 0, "accumulation_position": 0,
             "data_sha256": data_hash, "config_sha256": config_hash, "elapsed_seconds": 0.0,
             "maximum_branch_tokens_seen": 0, "maximum_aggregate_tokens_seen": 0,
             "wandb_identity": wandb_ref, "scheduler_identity": scheduler_ref,
             "experiment_id": experiment_id or "local-experiment", "run_id": run_id or uuid.uuid4().hex,
             "attempt_id": uuid.uuid4().hex, "attempt_index": 0, "parent_attempt_id": None,
             "execution_counters": {"prefix_passes": 0, "branch_passes": 0, "compute_tokens": 0,
                                    "padding_tokens": 0, "max_batch_size": 0, "batch_size_histogram": {}}}
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if resume is not None:
        restored, restored_tokenizer, payload = load_checkpoint(resume, str(device))
        if preprocessing_identity(restored_tokenizer) != preprocessing_identity(tokenizer):
            raise ValueError("resume tokenizer/preprocessing differs")
        old = payload["training_state"]
        if "execution" not in payload:
            raise ValueError("legacy serial checkpoint has no batching configuration; use --init for a new batched run")
        if old is None or payload["optimizer"] is None or payload["sampler"] is None:
            raise ValueError("checkpoint is inference-only, not resumable")
        if old["data_sha256"] != data_hash or old["config_sha256"] != config_hash:
            raise ValueError("resume configuration or frozen data differs")
        if experiment_id is not None and experiment_id != old["experiment_id"]:
            raise ValueError("resume experiment identity differs")
        model.load_state_dict(restored.state_dict())
        model.training_steps = restored.training_steps
        model.temperatures = restored.temperatures
        model.calibration_provenance = restored.calibration_provenance
        model.native_weights_loaded = restored.native_weights_loaded
        model.training_exposure = restored.training_exposure
        model.backbone_source = restored.backbone_source
        model.loaded_backbone_sha256 = restored.loaded_backbone_sha256
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        sampler.load_state_dict(payload["sampler"])
        restore_rng(payload["rng"])
        state = dict(old)
        if run_id is not None and run_id != old["run_id"]:
            raise ValueError("resume run identity differs")
        old_wandb = old.get("wandb_identity")
        if wandb_ref is not None and old_wandb != wandb_ref:
            raise ValueError("resume W&B identity differs")
        wandb_ref = old_wandb
        scheduler_ref = scheduler_ref or old.get("scheduler_identity")
        state["scheduler_identity"] = scheduler_ref
        parent_sha256 = payload["checkpoint_sha256"]
    if quality_budget is not None:
        if quality_allocation_id is not None:
            quality_budget.claim_allocation(quality_allocation_id, state["run_id"])
        quality_budget.verify_run(state["run_id"], steps=state["step"],
                                  useful_tokens=state["forward_tokens"])
    from .exposure import start_exposure, observe
    from .lineage import begin_attempt
    from .provenance import capture_source, extensions, ResourceSampler, hardware_identity
    from .publishing import WandbPublisher
    if publish_wandb and wandb_ref is None:
        raise ValueError("publishing requires an explicit existing W&B run identity")
    publisher = WandbPublisher(wandb_ref, enabled=True, sdk=wandb_sdk) if publish_wandb else None
    source = capture_source(source_root)
    model.source_provenance = source
    preliminary_monitoring_failures = 0
    try:
        atomic_json(output / "source-provenance.json", source)
    except OSError:
        preliminary_monitoring_failures += 1
    if resume is None:
        # Exposure for a new run starts from this run's own observed inputs.
        # Inherited weights/parent identity are recorded but not promoted to new
        # native exposure without a separately verified derivation.
        model.training_exposure = start_exposure(model, tokenizer)
    elif model.training_exposure is None:
        raise ValueError("legacy resume lacks immutable exposure evidence; use --init for a new run")
    lineage = begin_attempt(output, state, resumed=resume is not None)
    preliminary_monitoring_failures += int(not state['lineage_durable'])
    sampler_metrics = ResourceSampler(interval=settings.resource_sample_seconds, device=device)
    # Further weight training invalidates any prior temperature fit.
    model.temperatures = {t: 1.0 for t in ("choice", "score", "noul")}
    model.calibration_provenance = {"status": "unfitted-after-weight-training"}
    model.train()
    provenance = {"model": "kev-laya-training", "backbone": model.cfg.source, "backbone_revision": model.cfg.revision,
                  "tokenizer": tokenizer.identity, "config_sha256": config_hash, "data_sha256": data_hash,
                  "split_hashes": {k: v["sha256"] for k, v in manifest["partitions"].items()}, "seed": settings.seed,
                  "repository": source["repository"], "commit": source["commit"], "precision": settings.precision, "hardware": hardware_identity(device),
                  "context_limits": asdict(limits) | {}, "evidence_class": "pretrained-backbone" if model.native_weights_loaded else "tiny-synthetic-fixture"}
    provenance["context_limits"].pop("max_questions")
    writer = TelemetryWriter(output / "telemetry.json", new_snapshot(state["experiment_id"], state["run_id"], state["attempt_id"], provenance,
                            attempt_index=state["attempt_index"], parent_attempt_id=state["parent_attempt_id"],
                            scheduler=scheduler_ref, wandb=wandb_ref, schema_version=2,
                            extensions=extensions(tokenizer, source, lineage=lineage, lineage_durable=state['lineage_durable'], calibration=model.calibration_provenance)))
    writer.failures = preliminary_monitoring_failures
    config_artifact = None
    try:
        atomic_json(output / "config.json", cfg)
        config_artifact = artifact(output / "config.json", "configuration")
        # state.config_sha256 is the canonical optimization config used by
        # exact resume. Telemetry/artifact provenance identifies actual bytes.
        provenance["config_sha256"] = config_artifact["sha256"]
    except OSError:
        writer.failures += 1
        provenance["config_sha256"] = None
    # A resumed attempt initially reports its restored progress, not an invented
    # zero until the next optimizer step finishes (which may take a long time).
    restored_elapsed = state["elapsed_seconds"]
    initial_progress = {"optimizer_steps": state["step"], "microbatches": state["microbatches"],
                        "examples": state["examples"], "forward_tokens": state["forward_tokens"],
                        "epoch": sampler.epoch + sampler.cursor / len(data),
                        "totals": {"optimizer_steps": settings.steps, "examples": settings.steps * settings.accumulation,
                                   "microbatches": settings.steps * settings.accumulation, "forward_tokens": None},
                        "elapsed_seconds": restored_elapsed,
                        "eta_seconds": restored_elapsed / state["step"] * (settings.steps - state["step"]) if state["step"] else None,
                        "tokens_per_second": state["forward_tokens"] / restored_elapsed if restored_elapsed else None}
    initial_execution = dict(state["execution_counters"], useful_forward_tokens=state["forward_tokens"],
                             questions=sum(int(k)*v for k,v in state["execution_counters"]["batch_size_histogram"].items()))
    writer.update(phase="train", provenance=provenance, progress=initial_progress,
                  extensions=extensions(tokenizer, source, lineage=lineage, execution=initial_execution,
                                        resources=sampler_metrics.sample(), lineage_durable=state["lineage_durable"],
                                        calibration=model.calibration_provenance),
                  artifacts=[config_artifact] if config_artifact is not None else [])
    writer.start_heartbeat()
    def publish():
        if publisher is None:
            return
        try:
            import copy
            with writer._lock:
                published = copy.deepcopy(writer.snapshot)
            publisher.publish(published)
        except Exception as exc:
            import warnings
            with writer._lock:
                writer.failures += 1
            warnings.warn(f"W&B publishing failed ({type(exc).__name__}); local training continues", RuntimeWarning)

    start, initial_elapsed = perf_counter(), state["elapsed_seconds"]
    logs, kept, last_path = [], [], None
    budget_stop_reason = None
    def quality_gpu_peak():
        return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    final_step = min(settings.steps, stop_after) if stop_after is not None else settings.steps
    if final_step < state["step"]:
        writer.close()
        raise ValueError("stop_after is before resumed position")
    try:
        while state["step"] < final_step:
            step_start_tokens = state["forward_tokens"]
            if quality_budget is not None:
                preview_sampler = copy.deepcopy(sampler)
                preview_random = random.Random()
                preview_random.setstate(random.getstate())
                planned_tokens = 0
                for _ in range(settings.accumulation):
                    preview_datum = data[preview_sampler.next()]
                    if settings.choice_permutation:
                        preview_datum = permute_choices(preview_datum, preview_random)
                    planned_tokens += preview_datum.encode(tokenizer, limits).logical_tokens
                try:
                    quality_budget.reserve_step(state["run_id"], useful_tokens=planned_tokens,
                                                peak_gpu_bytes=quality_gpu_peak())
                except BudgetExceeded as exc:
                    if str(exc) not in {"step_budget_exceeded", "useful_token_budget_exceeded",
                                        "run_step_budget_exceeded", "run_useful_token_budget_exceeded",
                                        "wall_budget_exceeded", "gpu_memory_budget_exceeded"}:
                        raise
                    budget_stop_reason = str(exc)
                    break
            optimizer.zero_grad(set_to_none=True)
            averaged = {}
            for _ in range(settings.accumulation):
                datum = data[sampler.next()]
                if settings.choice_permutation:
                    datum = permute_choices(datum, random)
                encoded = datum.encode(tokenizer, limits)
                observe(model.training_exposure, encoded)
                state["maximum_branch_tokens_seen"] = max(state.get("maximum_branch_tokens_seen", 0), len(encoded.state) + max(len(b.ids) for b in encoded.branches))
                state["maximum_aggregate_tokens_seen"] = max(state.get("maximum_aggregate_tokens_seen", 0), encoded.logical_tokens)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=settings.precision == "bf16"):
                    logits, usage = model(encoded)
                    loss, parts = objective(logits, datum.targets, [b.question.type for b in encoded.branches], loss_config)
                (loss / settings.accumulation).backward()
                if quality_budget is not None:
                    quality_budget.check_resources(peak_gpu_bytes=quality_gpu_peak())
                for key, value in parts.items():
                    averaged[key] = averaged.get(key, 0) + value / settings.accumulation
                state["microbatches"] += 1
                state["examples"] += 1
                state["forward_tokens"] += usage["forward_tokens"]
                execution = state["execution_counters"]
                for counter in ("prefix_passes", "branch_passes", "compute_tokens", "padding_tokens"):
                    execution[counter] += usage[counter]
                execution["max_batch_size"] = max(execution["max_batch_size"], max(usage["effective_batch_sizes"]))
                for size in usage["effective_batch_sizes"]:
                    key = str(size)
                    execution["batch_size_histogram"][key] = execution["batch_size_histogram"].get(key, 0) + 1
            trainable = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
                raise FloatingPointError("missing or nonfinite trainable parameter gradient")
            norm = torch.nn.utils.clip_grad_norm_(trainable, settings.gradient_clip, error_if_nonfinite=True)
            if quality_budget is not None:
                quality_budget.check_resources(peak_gpu_bytes=quality_gpu_peak())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            model.training_steps += 1
            model.training_exposure["observed"]["optimizer_steps"] += 1
            state["exposure_observed"] = dict(model.training_exposure["observed"])
            if quality_budget is not None:
                quality_budget.commit_step(state["run_id"],
                    useful_tokens=state["forward_tokens"] - step_start_tokens,
                    peak_gpu_bytes=quality_gpu_peak())
            elapsed = initial_elapsed + perf_counter() - start
            state["elapsed_seconds"] = elapsed
            averaged.update({"gradient/norm": float(norm), "learning_rate": float(scheduler.get_last_lr()[0])})
            logs.append({"step": state["step"], **averaged})
            progress = {"optimizer_steps": state["step"], "microbatches": state["microbatches"], "examples": state["examples"],
                        "forward_tokens": state["forward_tokens"], "epoch": sampler.epoch + sampler.cursor / len(data),
                        "totals": {"optimizer_steps": settings.steps, "examples": settings.steps * settings.accumulation,
                                   "microbatches": settings.steps * settings.accumulation, "forward_tokens": None},
                        "elapsed_seconds": elapsed, "eta_seconds": elapsed / state["step"] * (settings.steps - state["step"]),
                        "tokens_per_second": state["forward_tokens"] / elapsed if elapsed else None}
            execution_summary = dict(state["execution_counters"])
            execution_summary["useful_forward_tokens"] = state["forward_tokens"]
            execution_summary["questions"] = sum(int(k)*v for k,v in execution_summary["batch_size_histogram"].items())
            resources = sampler_metrics.sample()
            measured = dict(averaged)
            if resources["rss_bytes"] is not None:
                measured["resource/rss_bytes"] = resources["rss_bytes"]
            if resources["gpu_peak_allocated_bytes"] is not None:
                measured["resource/gpu_peak_bytes"] = resources["gpu_peak_allocated_bytes"]
            writer.update(metrics=measured, progress=progress,
                          extensions=extensions(tokenizer, source, lineage=lineage,
                                                execution=execution_summary, resources=resources,
                                                lineage_durable=state['lineage_durable'], calibration=model.calibration_provenance))
            if quality_budget is not None or state["step"] % settings.save_every == 0 or state["step"] == final_step:
                last_path = output / f"checkpoint-{state['step']:06d}.pt"
                saved = save_checkpoint(last_path, model, tokenizer, training_state=dict(state), optimizer=optimizer,
                                        scheduler=scheduler, sampler=sampler, provenance=provenance, parent_sha256=parent_sha256)
                parent_sha256 = saved["sha256"]
                kept.append(last_path)
                while len(kept) > settings.keep_checkpoints:
                    old_path = kept.pop(0)
                    old_path.unlink(missing_ok=True)
                    old_path.with_suffix(".manifest.json").unlink(missing_ok=True)
                    old_path.with_name(old_path.stem.replace("checkpoint-", "execution-") + ".json").unlink(missing_ok=True)
                # Extra diagnostics use an existing v1 artifact kind, not new telemetry
                # fields or relabelled training microbatches. Monitoring failure
                # cannot invalidate an already committed optimizer checkpoint.
                execution_artifacts = []
                try:
                    execution_path = output / f"execution-{state['step']:06d}.json"
                    atomic_json(execution_path, {"batch_policy": model.batch_policy.to_dict(),
                                "counters": state["execution_counters"],
                                "forward_tokens": state["forward_tokens"],
                                "accounting": "unpadded forward tokens; compute_tokens includes suffix padding; excludes backward recomputation"})
                    execution_artifacts = [artifact(execution_path, "evaluation")]
                except OSError:
                    with writer._lock:
                        writer.failures += 1
                writer.update(artifacts=[*([config_artifact] if config_artifact is not None else []), artifact(last_path, "checkpoint", saved["parent_sha256"], resumable=True), *execution_artifacts],
                              resume={"capable": True, "checkpoint_sha256": saved["sha256"]})
                try:
                    atomic_json(output / "latest.json", saved)
                except OSError:
                    writer.failures += 1  # checkpoint is already committed; pointer is optional telemetry
                publish()
        writer.update(phase="completed" if state["step"] == settings.steps else "train")
        publish()
    except BaseException:
        writer.update(phase="cancelled" if isinstance(__import__('sys').exc_info()[1], KeyboardInterrupt) else "failed")
        raise
    finally:
        writer.close()
    try:
        atomic_json(output / "training-metrics.json", {"records": logs, "state": state, "monitoring_export_failures": writer.failures})
    except OSError:
        writer.failures += 1  # committed checkpoint remains valid
    result = {"checkpoint": str(last_path if last_path is not None else resume), "state": state, "metrics": logs,
              "monitoring_export_failures": writer.failures, "snapshot": str(output / "telemetry.json")}
    if quality_budget is not None:
        result["quality_budget_stop_reason"] = budget_stop_reason
    return result
