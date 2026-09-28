"""Dedicated process deadline for a frozen local quality training run.

The child uses the ordinary quality CLI and durable budget ledger. A timed-out
child is stopped, never restarted automatically; its pending ledger step needs
audited reconciliation before any later run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .quality_budget import StepTokenBudget, _lock, _unlock
from .quality_protocol import preflight
from .schema import strict_loads


@contextmanager
def supervisor_lease(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        _lock(fd)
    except BaseException:
        os.close(fd)
        raise
    try:
        yield
    finally:
        _unlock(fd)
        os.close(fd)


def supervise_child(command: list[str], remaining_seconds: float) -> tuple[str, int, float]:
    """Wait for exactly this owned child, then stop it if the deadline expires."""
    if remaining_seconds <= 0:
        return "deadline_exhausted", 124, 0.0
    started = time.monotonic()
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, **options)
    try:
        code = child.wait(timeout=remaining_seconds)
        return ("completed" if code == 0 else "child_failed"), code, time.monotonic() - started
    except subprocess.TimeoutExpired:
        _stop_child(child)
        return "deadline_killed", 124, time.monotonic() - started
    except BaseException:
        _stop_child(child)
        raise


def _stop_child(child: subprocess.Popen) -> None:
    """Stop the owned process group where the platform exposes that operation."""
    try:
        if os.name != "nt":
            os.killpg(child.pid, signal.SIGKILL)
        else:
            # CREATE_NEW_PROCESS_GROUP does not make child.kill() reach Windows
            # descendants. Stop the exact process tree owned by this supervisor.
            stopped = subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     check=False)
            if stopped.returncode != 0:
                if child.poll() is None:
                    child.kill()
                raise RuntimeError("owned_child_tree_termination_unverified")
    except (ProcessLookupError, OSError):
        pass
    finally:
        child.wait()


def _write_receipt(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kev-laya-quality", description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-review", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    training_args = args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args
    if not training_args or training_args[0] not in {"train", "reward-train"}:
        raise ValueError("supply train or reward-train after --")
    owned_flags = ("--suite", "--quality-protocol", "--quality-data-review",
                   "--quality-budget-ledger", "--wandb-publish")
    if any(value == flag or value.startswith(flag + "=")
           for value in training_args for flag in owned_flags):
        raise ValueError("suite and quality inputs belong to the supervisor; remote publishing is unavailable")
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    receipt = preflight(args.protocol, args.data_review, args.suite)
    raw = args.protocol.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != receipt["protocol_sha256"]:
        raise ValueError("quality protocol changed after preflight")
    protocol = strict_loads(raw)
    arm = "supervised" if training_args[0] == "train" else "reward"
    arm_budget = protocol["training"]["arms"][arm]
    arms = protocol["training"]["arms"]
    limits = protocol["budget"]
    budget_args = dict(max_steps=limits["max_total_optimizer_steps"],
                       max_useful_tokens=len(protocol["seeds"]) *
                           sum(value["useful_token_budget"] for value in arms.values()),
                       max_run_steps=arm_budget["optimizer_steps"],
                       max_run_useful_tokens=arm_budget["useful_token_budget"],
                       max_wall_seconds=limits["max_wall_seconds"],
                       max_peak_gpu_memory_bytes=limits["max_peak_gpu_memory_bytes"])
    child_args = [sys.executable, "-m", "kev_laya.cli", *training_args,
                  "--suite", str(args.suite), "--quality-protocol", str(args.protocol),
                  "--quality-data-review", str(args.data_review),
                  "--quality-budget-ledger", str(args.budget_ledger)]
    lease = args.budget_ledger.with_name(args.budget_ledger.name + ".supervisor.lock")
    with supervisor_lease(lease):
        with StepTokenBudget(args.budget_ledger, raw, **budget_args) as budget:
            started_at = budget.state["started_at_wall"]
            last_wall = budget.state["last_wall"]
        wall_now = time.time()
        remaining = -1.0 if wall_now < last_wall else limits["max_wall_seconds"] - (wall_now - started_at)
        status, code, elapsed = supervise_child(child_args, remaining)
        ledger_bytes = args.budget_ledger.read_bytes()
        ledger = strict_loads(ledger_bytes)
        _write_receipt(args.receipt, {
            "schema_version": "layev-quality-supervisor/1",
            "status": status, "child_exit_code": code,
            "protocol_sha256": digest, "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            "wall_limit_seconds": limits["max_wall_seconds"],
            "supervised_elapsed_seconds": elapsed,
            "pending_step_requires_reconciliation": ledger.get("pending") is not None,
            "limitations": "Dedicated child deadline while supervisor lives; non-PyTorch GPU allocations remain outside allocator accounting."})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
