"""Versioned complete optimization-boundary checkpoints; weights-only deserialization."""
from __future__ import annotations
from pathlib import Path
import random
import torch
from .encoding import ByteTokenizer, QwenTokenizer, SERIALIZATION, NATIVE_SERIALIZATION, LEGACY_ESCAPE, preprocessing_identity
from .io import atomic_file, atomic_json, sha256_file
from .model import BackboneConfig, DecisionEngine
from .execution import BatchPolicy

FORMAT = "kev-laya-checkpoint/1"


def rng_state() -> dict:
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume device topology differs")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, model: DecisionEngine, tokenizer, *, training_state: dict | None = None,
                    optimizer=None, scheduler=None, sampler=None, provenance=None, parent_sha256=None,
                    parent_checkpoint: Path | None = None, exposure_operation="export") -> dict:
    path = Path(path)
    if training_state is not None and training_state.get("accumulation_position", 0) != 0:
        raise ValueError("checkpoint must be written at an optimization boundary")
    token_meta = tokenizer.metadata()
    if token_meta["kind"] == "qwen":
        target = path.parent / "tokenizer.json"
        expected = tokenizer.identity.split(":", 1)[1]
        if target.exists() and sha256_file(target) != expected:
            raise ValueError("refusing to replace a different checkpoint tokenizer")
        if not target.exists():
            raw = tokenizer.path.read_bytes()
            atomic_file(target, lambda f: f.write(raw))
        token_meta = token_meta | {"path": "tokenizer.json"}
    from .exposure import snapshot_exposure
    exposure = snapshot_exposure(model, tokenizer, training=training_state is not None,
                                 parent_checkpoint=parent_checkpoint, operation=exposure_operation,
                                 parent_sha256=parent_sha256)
    payload = {"training_exposure": exposure, "source_provenance": getattr(model, "source_provenance", None),
               "backbone_source": getattr(model, "backbone_source", None),
               "loaded_backbone_sha256": getattr(model, "loaded_backbone_sha256", None), "format": FORMAT, "config": model.config_dict(), "model": model.state_dict(), "execution": model.batch_policy.to_dict(),
               "tokenizer": token_meta, "preprocessing": preprocessing_identity(tokenizer), "temperatures": dict(model.temperatures),
               "calibration": model.calibration_provenance, "native_weights_loaded": model.native_weights_loaded,
               "training_steps": model.training_steps, "training_state": training_state,
               "optimizer": None if optimizer is None else optimizer.state_dict(),
               "scheduler": None if scheduler is None else scheduler.state_dict(),
               "sampler": None if sampler is None else sampler.state_dict(),
               "rng": rng_state(), "provenance": provenance or {}, "parent_sha256": parent_sha256}
    atomic_file(path, lambda stream: torch.save(payload, stream))
    manifest = {"format": FORMAT, "file": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                "resumable": all(payload[k] is not None for k in ("training_state", "optimizer", "scheduler", "sampler")),
                "parent_sha256": parent_sha256, "training_steps": model.training_steps}
    atomic_json(path.with_suffix(".manifest.json"), manifest)
    return manifest


def load_checkpoint(path: Path, device="cpu", *, expected_sha256: str | None = None) -> tuple[DecisionEngine, object, dict]:
    path = Path(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("checkpoint checksum mismatch")
    manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != digest:
            raise ValueError("checkpoint differs from its manifest")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("unsupported checkpoint format")
    model = DecisionEngine(BackboneConfig(**payload["config"]),
                           BatchPolicy.from_dict(payload.get("execution", {})))
    model.load_state_dict(payload["model"], strict=True)
    model.temperatures = payload["temperatures"]
    for temperature in model.temperatures.values():
        if not isinstance(temperature, (float, int)) or not 0.2 <= temperature <= 5.0:
            raise ValueError("invalid checkpoint temperature")
    model.calibration_provenance = payload["calibration"]
    model.native_weights_loaded = payload["native_weights_loaded"]
    model.training_steps = payload["training_steps"]
    model.training_exposure = payload.get("training_exposure")
    model.source_provenance = payload.get("source_provenance")
    model.backbone_source = payload.get("backbone_source")
    model.loaded_backbone_sha256 = payload.get("loaded_backbone_sha256")
    meta = payload["tokenizer"]
    if meta["kind"] == "byte-fixture":
        if meta.get("literal_encoding", "utf8-bytes-v1") != "utf8-bytes-v1":
            raise ValueError("unknown byte-fixture preprocessing; explicit migration required")
        tokenizer = ByteTokenizer()
    elif meta["kind"] == "qwen":
        if Path(meta["path"]).name != meta["path"]:
            raise ValueError("tokenizer must be checkpoint-local")
        mode = meta.get("literal_encoding", meta.get("escape"))
        if "literal_encoding" in meta and "escape" in meta and meta["escape"] != mode:
            raise ValueError("conflicting tokenizer preprocessing versions")
        if mode is None:
            raise ValueError("Qwen checkpoint lacks preprocessing version; explicit migration required")
        tokenizer = QwenTokenizer(path.parent / meta["path"], literal_encoding=mode)
    else:
        raise ValueError("unknown tokenizer kind")
    recorded_serialization = meta.get("serialization", SERIALIZATION)
    if recorded_serialization != tokenizer.serialization:
        raise ValueError("checkpoint serialization/tokenizer mismatch; explicit migration required")
    if payload.get("preprocessing", preprocessing_identity(tokenizer)) != preprocessing_identity(tokenizer):
        raise ValueError("checkpoint preprocessing identity mismatch")
    if max(tokenizer.special) >= model.cfg.vocab_size or tokenizer.metadata()["vocab_size"] > model.cfg.vocab_size:
        raise ValueError("tokenizer vocabulary does not fit checkpoint embeddings")
    if tokenizer.identity != meta["identity"]:
        raise ValueError("tokenizer checksum mismatch")
    model.to(device)
    payload["checkpoint_sha256"] = digest
    payload["model_id"] = "kev-laya-0.1.0-" + digest[:16]
    return model, tokenizer, payload
