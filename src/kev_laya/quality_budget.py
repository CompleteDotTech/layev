"""Durable, single-owner step, token and measured-resource guards for quality runs.

This is one part of the representative-quality resource policy. A pending step
after a crash requires checkpoint reconciliation; it is never silently retried.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time


class BudgetExceeded(RuntimeError):
    pass


def _lock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


class StepTokenBudget:
    def __init__(self, path: Path, protocol_bytes: bytes, *, max_steps: int, max_useful_tokens: int,
                 max_run_steps: int | None = None, max_run_useful_tokens: int | None = None,
                 max_wall_seconds: float | None = None, max_peak_gpu_memory_bytes: int | None = None,
                 now=time.time):
        max_run_steps = max_steps if max_run_steps is None else max_run_steps
        max_run_useful_tokens = max_useful_tokens if max_run_useful_tokens is None else max_run_useful_tokens
        if any(type(value) is not int or value < 1 for value in
               (max_steps, max_useful_tokens, max_run_steps, max_run_useful_tokens)):
            raise ValueError("positive integer budgets required")
        if max_wall_seconds is not None and (type(max_wall_seconds) not in (int, float)
                                             or not math.isfinite(max_wall_seconds)
                                             or max_wall_seconds <= 0):
            raise ValueError("positive finite wall budget required")
        if max_peak_gpu_memory_bytes is not None and (type(max_peak_gpu_memory_bytes) is not int
                                                     or max_peak_gpu_memory_bytes < 1):
            raise ValueError("positive GPU memory budget required")
        self.path = Path(path)
        self.now = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_fd = os.open(self.path.with_name(self.path.name + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            _lock(self.lock_fd)
            identity = hashlib.sha256(protocol_bytes).hexdigest()
            limits = {"steps": max_steps, "useful_tokens": max_useful_tokens,
                      "run_steps": max_run_steps, "run_useful_tokens": max_run_useful_tokens,
                      "wall_seconds": max_wall_seconds, "peak_gpu_bytes": max_peak_gpu_memory_bytes}
            if self.path.exists():
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
                if self.state.get("protocol_sha256") != identity or self.state.get("limits") != limits:
                    raise BudgetExceeded("budget_identity_mismatch")
                if self.state.get("pending") is not None:
                    raise BudgetExceeded("in_flight_step_requires_reconciliation")
            else:
                wall_now = self.now()
                if not math.isfinite(wall_now):
                    raise ValueError("finite wall clock required")
                self.state = {"protocol_sha256": identity, "limits": limits, "steps": 0,
                              "useful_tokens": 0, "runs": {}, "allocations": {}, "pending": None,
                              "started_at_wall": wall_now, "last_wall": wall_now, "peak_gpu_bytes": 0}
                self._save()
        except BaseException:
            try:
                _unlock(self.lock_fd)
            except OSError:
                pass
            os.close(self.lock_fd)
            raise

    def close(self) -> None:
        if self.lock_fd is not None:
            _unlock(self.lock_fd)
            os.close(self.lock_fd)
            self.lock_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _save(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        if temporary.exists():
            raise BudgetExceeded("unreconciled_temporary_receipt")
        data = (json.dumps(self.state, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def verify_run(self, run_id: str, *, steps: int, useful_tokens: int) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id required")
        expected = self.state["runs"].get(run_id, {"steps": 0, "useful_tokens": 0})
        if expected != {"steps": steps, "useful_tokens": useful_tokens}:
            raise BudgetExceeded("run_checkpoint_budget_mismatch")

    def claim_allocation(self, allocation_id: str, run_id: str) -> None:
        if not isinstance(allocation_id, str) or not allocation_id or not isinstance(run_id, str) or not run_id:
            raise ValueError("allocation and run IDs required")
        owner = self.state["allocations"].get(allocation_id)
        if owner is not None and owner != run_id:
            raise BudgetExceeded("seed_arm_allocation_already_owned")
        if owner is None:
            self.state["allocations"][allocation_id] = run_id
            self._save()

    def reserve_step(self, run_id: str, *, useful_tokens: int,
                     peak_gpu_bytes: int | None = None) -> None:
        if self.state["pending"] is not None:
            raise BudgetExceeded("in_flight_step_requires_reconciliation")
        if type(useful_tokens) is not int or useful_tokens < 1:
            raise ValueError("positive planned useful-token count required")
        wall_now = self.check_resources(peak_gpu_bytes=peak_gpu_bytes)
        if self.state["steps"] + 1 > self.state["limits"]["steps"]:
            raise BudgetExceeded("step_budget_exceeded")
        if self.state["useful_tokens"] + useful_tokens > self.state["limits"]["useful_tokens"]:
            raise BudgetExceeded("useful_token_budget_exceeded")
        run = self.state["runs"].get(run_id, {"steps": 0, "useful_tokens": 0})
        if run["steps"] + 1 > self.state["limits"]["run_steps"]:
            raise BudgetExceeded("run_step_budget_exceeded")
        if run["useful_tokens"] + useful_tokens > self.state["limits"]["run_useful_tokens"]:
            raise BudgetExceeded("run_useful_token_budget_exceeded")
        self.state["pending"] = {"run_id": run_id, "useful_tokens": useful_tokens}
        self.state["last_wall"] = wall_now
        self._save()

    def check_resources(self, *, peak_gpu_bytes: int | None = None) -> float:
        wall_now = self.now()
        if not math.isfinite(wall_now) or wall_now < self.state["last_wall"]:
            raise BudgetExceeded("wall_clock_reversed_or_invalid")
        wall_limit = self.state["limits"]["wall_seconds"]
        if wall_limit is not None and wall_now - self.state["started_at_wall"] >= wall_limit:
            raise BudgetExceeded("wall_budget_exceeded")
        if peak_gpu_bytes is not None:
            if type(peak_gpu_bytes) is not int or peak_gpu_bytes < 0:
                raise ValueError("nonnegative measured GPU peak required")
            gpu_limit = self.state["limits"]["peak_gpu_bytes"]
            if gpu_limit is not None and peak_gpu_bytes > gpu_limit:
                raise BudgetExceeded("gpu_memory_budget_exceeded")
        self.state["last_wall"] = wall_now
        return wall_now

    def commit_step(self, run_id: str, *, useful_tokens: int,
                    peak_gpu_bytes: int | None = None) -> None:
        if self.state["pending"] != {"run_id": run_id, "useful_tokens": useful_tokens}:
            raise BudgetExceeded("step_reservation_mismatch")
        wall_now = self.check_resources(peak_gpu_bytes=peak_gpu_bytes)
        self.state["steps"] += 1
        self.state["useful_tokens"] += useful_tokens
        run = self.state["runs"].setdefault(run_id, {"steps": 0, "useful_tokens": 0})
        run["steps"] += 1
        run["useful_tokens"] += useful_tokens
        self.state["peak_gpu_bytes"] = max(self.state["peak_gpu_bytes"], peak_gpu_bytes or 0)
        self.state["last_wall"] = wall_now
        self.state["pending"] = None
        self._save()


def configure_cuda_allocator_limit(device, max_peak_gpu_memory_bytes: int) -> dict:
    """Cap this dedicated process's PyTorch CUDA allocator before model loading."""
    import torch
    selected = torch.device(device)
    if selected.type != "cuda":
        return {"device": str(selected), "allocator_cap_applied": False}
    if type(max_peak_gpu_memory_bytes) is not int or max_peak_gpu_memory_bytes < 1:
        raise ValueError("positive GPU memory budget required")
    total = torch.cuda.get_device_properties(selected).total_memory
    fraction = min(1.0, max_peak_gpu_memory_bytes / total)
    torch.cuda.set_per_process_memory_fraction(fraction, selected)
    torch.cuda.reset_peak_memory_stats(selected)
    return {"device": str(selected), "allocator_cap_applied": True,
            "allocator_fraction": fraction, "device_total_bytes": total}
