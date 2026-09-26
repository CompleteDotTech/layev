"""Bounded content-free telemetry export; it never writes Overwatch's cache."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
from pathlib import Path
import threading
import warnings
from .io import atomic_json, sha256_file
from .telemetry_contract import MAX_HISTORY, validate_snapshot


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact(path: Path, kind: str, parent: str | None = None, *, resumable=False) -> dict:
    path = Path(path).resolve()
    return {"kind": kind, "uri": path.as_uri(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            "parent_sha256": parent, "resumable": resumable}


def new_snapshot(experiment_id: str, run_id: str, attempt_id: str, provenance: dict,
                 *, attempt_index=0, parent_attempt_id=None, scheduler=None, wandb=None, schema_version=1, extensions=None) -> dict:
    now = utc_now()
    snapshot = {"schema_version": schema_version, "framework": "kev_laya", "framework_version": "0.1.0+stage3" if schema_version == 2 else "0.1.0",
            "experiment_id": experiment_id, "run_id": run_id, "attempt_id": attempt_id,
            "attempt_index": attempt_index, "parent_attempt_id": parent_attempt_id, "sequence": 0,
            "started_at": now, "updated_at": now, "heartbeat_at": now, "phase": "prepare",
            "scheduler": scheduler, "wandb": wandb,
            "progress": {"epoch": None, "optimizer_steps": 0, "microbatches": 0, "examples": 0, "forward_tokens": 0,
                         "totals": {}, "elapsed_seconds": 0.0, "eta_seconds": None, "tokens_per_second": None},
            "metrics": {}, "history": [], "provenance": provenance, "artifacts": [],
            "serving": {"requests": 0, "errors": 0, "input_tokens": 0, "forward_tokens": 0, "output_tokens": 0,
                        "questions": 0, "prefix_reuses": 0, "latency_ms": []},
            "resume": {"capable": False, "checkpoint_sha256": None},
            "recoveries": {"total": None, "infrastructure": None, "application": None},
            "cost": {"measured_usd": None, "estimated_usd": None, "basis": None},
            "monitoring_export_failures": 0}
    if schema_version == 2:
        if extensions is None:
            raise ValueError("telemetry v2 requires explicit extensions")
        snapshot["extensions"] = copy.deepcopy(extensions)
    return snapshot


class TelemetryWriter:
    def __init__(self, path: Path | None, snapshot: dict):
        self.path = None if path is None else Path(path)
        self.snapshot = copy.deepcopy(snapshot)
        self.failures = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, *, phase: str | None = None, metrics: dict | None = None, **fields) -> bool:
        with self._lock:
            self.snapshot.update(copy.deepcopy(fields))
            if phase is not None:
                self.snapshot["phase"] = phase
            if metrics is not None:
                self.snapshot["metrics"] = dict(metrics)
                row = {"step": self.snapshot["progress"]["optimizer_steps"], "phase": self.snapshot["phase"], "metrics": dict(metrics)}
                self.snapshot["history"] = (self.snapshot["history"] + [row])[-MAX_HISTORY:]
            return self.export()

    def export(self) -> bool:
        with self._lock:
            now = utc_now()
            self.snapshot["sequence"] += 1
            self.snapshot["updated_at"] = self.snapshot["heartbeat_at"] = now
            self.snapshot["monitoring_export_failures"] = self.failures
            try:
                validate_snapshot(self.snapshot)
                if self.path is not None:
                    atomic_json(self.path, self.snapshot)
                return True
            except Exception as exc:  # Monitoring may fail; model/checkpoint transactions must survive.
                self.failures += 1
                if self.failures == 1 or self.failures % 100 == 0:
                    warnings.warn(f"telemetry export failed ({type(exc).__name__}); training continues", RuntimeWarning)
                return False

    def start_heartbeat(self, seconds: float = 15):
        if seconds <= 0 or self._thread is not None:
            raise ValueError("invalid heartbeat configuration")
        def beat():
            while not self._stop.wait(seconds):
                self.export()
        self._thread = threading.Thread(target=beat, name="kev-laya-heartbeat", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.export()


class ServingHook:
    """Content-free measured serving aggregates; no request/answer content access.

    Counter updates and export scheduling are serialized together, so concurrent
    requests cannot publish an older counter snapshot after a newer one.
    """
    def __init__(self, writer: TelemetryWriter | None = None, *, resource_sampler=None):
        self.writer = writer
        self.resource_sampler = resource_sampler
        self._lock = threading.Lock()
        self.values = {"requests": 0, "errors": 0, "input_tokens": 0, "forward_tokens": 0, "output_tokens": 0,
                       "questions": 0, "prefix_reuses": 0, "latency_ms": []}
        self.execution = {"prefix_passes": 0, "branch_passes": 0, "compute_tokens": 0, "padding_tokens": 0,
                          "useful_forward_tokens": 0, "questions": 0, "max_batch_size": 0,
                          "batch_size_histogram": {}}

    def finish(self, *, elapsed_ms: float, error: bool, input_tokens=0, forward_tokens=0,
               output_tokens=0, questions=0, prefix_reuses=0, execution=None):
        with self._lock:
            self.values["requests"] += 1
            self.values["errors"] += int(error)
            for key, value in (("input_tokens", input_tokens), ("forward_tokens", forward_tokens),
                               ("output_tokens", output_tokens), ("questions", questions), ("prefix_reuses", prefix_reuses)):
                self.values[key] += value
            self.values["latency_ms"] = (self.values["latency_ms"] + [elapsed_ms])[-MAX_HISTORY:]
            if execution is not None:
                for key in ("prefix_passes", "branch_passes", "compute_tokens", "padding_tokens"):
                    self.execution[key] += execution[key]
                self.execution["useful_forward_tokens"] += forward_tokens
                self.execution["questions"] += questions
                sizes = execution["effective_batch_sizes"]
                self.execution["max_batch_size"] = max(self.execution["max_batch_size"], max(sizes, default=0))
                for size in sizes:
                    key = str(size)
                    self.execution["batch_size_histogram"][key] = self.execution["batch_size_histogram"].get(key, 0) + 1
            if self.writer is not None:
                fields = {"serving": copy.deepcopy(self.values)}
                if self.writer.snapshot["schema_version"] == 2:
                    ext = copy.deepcopy(self.writer.snapshot["extensions"])
                    ext["execution"] = copy.deepcopy(self.execution)
                    if self.resource_sampler is not None:
                        try:
                            ext["resources"] = self.resource_sampler.sample()
                        except Exception:
                            self.writer.failures += 1
                            ext["resources"] = None  # measurement is unknown, not a request failure
                    fields["extensions"] = ext
                self.writer.update(phase="serve", **fields)
