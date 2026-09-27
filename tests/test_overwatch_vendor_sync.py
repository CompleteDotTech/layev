"""Pinned payload and isolated installer tests, not live Overwatch acceptance.

The eight core-file transformations and actual checkout revision are outside
these added-file fixtures. Their existing acceptance tests remain required.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations/overwatch"
PAYLOAD = INTEGRATION / "added"
UPGRADE = ROOT / "tests/fixtures/overwatch_guards_from_legacy.patch"
PINNED = {
    "src/overwatch/kev_laya_adapter.py": "99e5a25c1047f89750d7b12f6427307c7af7575d",
    "src/overwatch/providers/kev_laya.py": "9c08ed6b759429043e4ff25859caa97bb47e0730",
    "tests/test_kev_laya_cache_versions.py": "31c7b2a1d93af40b851c2e36060edd510c1d30ff",
    "tests/test_kev_laya_wandb_response_guards.py": "5e79b80cbeb0e63d89af62145ff7f1df72f82c12",
}
PREDECESSORS = {
    "src/overwatch/kev_laya_adapter.py": "73cd584845dcfd474fb7a71fd01f2620ee57f1f28b637fb2b2034cee9c4778bd",
    "src/overwatch/providers/kev_laya.py": "262e49932e88251462d5391b8de767b17213f74116dd1c3df1a39218edd999ad",
}


def canonical(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def installer(monkeypatch):
    spec = importlib.util.spec_from_file_location("vendor_sync_installer", INTEGRATION / "apply.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Added-file unit fixture only; this is NOT core-file or revision acceptance.
    monkeypatch.setattr(module, "BLOBS", {})
    return module


def legacy_target(tmp_path: Path, *, crlf: bool = False) -> Path:
    target = tmp_path / "target"
    target.mkdir()
    for name in PREDECESSORS:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(PAYLOAD / name))
    # This frozen source diff reconstructs the exact published predecessors.
    # Git for Windows treats CRLF in patch context as source bytes. Normalize
    # this text fixture just as the pinned payloads are normalized above.
    patch = tmp_path / "legacy-guards.patch"
    patch.write_bytes(canonical(UPGRADE))
    # No download, repository checkout, commit or signature operation is made.
    result = subprocess.run(
        ["git", "apply", "--reverse", "--unidiff-zero", str(patch)], cwd=target,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name, digest in PREDECESSORS.items():
        path = target / name
        assert hashlib.sha256(canonical(path)).hexdigest() == digest
        if crlf:
            path.write_bytes(canonical(path).replace(b"\n", b"\r\n"))
    (target / "unrelated.txt").write_bytes(b"unrelated user work\n")
    return target


@pytest.mark.parametrize("name,expected", PINNED.items())
def test_payload_matches_reviewed_overwatch_source(name, expected):
    raw = canonical(PAYLOAD / name)
    assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == expected


@pytest.mark.parametrize("name,digest", PREDECESSORS.items())
def test_existing_manifest_already_authorizes_exact_predecessor(name, digest):
    manifest = json.loads((INTEGRATION / "known-added-hashes.json").read_text(encoding="utf-8"))
    assert digest in manifest["files"][name]


@pytest.mark.parametrize("crlf", [False, True])
def test_exact_predecessor_upgrade_is_read_only_until_apply_and_idempotent(tmp_path, monkeypatch, crlf):
    module = installer(monkeypatch)
    target = legacy_target(tmp_path, crlf=crlf)
    before = {name: (target / name).read_bytes() for name in PREDECESSORS}
    changes = module.prepare(target, verify_revision=False)
    assert all((target / name).read_bytes() == raw for name, raw in before.items())
    assert all(target / name in changes for name in PREDECESSORS)
    backup = module.apply_transaction(target, changes)
    assert backup is not None
    for name, raw in before.items():
        assert (backup / name).read_bytes() == raw
        assert canonical(target / name) == canonical(PAYLOAD / name)
        if crlf:
            assert b"\r\n" in (target / name).read_bytes()
            assert b"\n" not in (target / name).read_bytes().replace(b"\r\n", b"")
    assert (target / "unrelated.txt").read_bytes() == b"unrelated user work\n"
    assert not module.prepare(target, verify_revision=False)


@pytest.mark.parametrize("name", PREDECESSORS)
def test_unknown_target_edit_is_not_overwritten(tmp_path, monkeypatch, name):
    module = installer(monkeypatch)
    target = legacy_target(tmp_path)
    changed = target / name
    changed.write_bytes(changed.read_bytes() + b"\n# unrelated local modification\n")
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="overwrite existing added file"):
        module.prepare(target, verify_revision=False)
    assert before == {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", PREDECESSORS)
def test_edit_after_preflight_is_not_overwritten(tmp_path, monkeypatch, name):
    module = installer(monkeypatch)
    target = legacy_target(tmp_path)
    changes = module.prepare(target, verify_revision=False)
    changed = target / name
    changed.write_bytes(changed.read_bytes() + b"\n# edit after preflight\n")
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="changed since preflight"):
        module.apply_transaction(target, changes)
    assert before == {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}


def test_failed_write_rolls_back_touched_payloads(tmp_path, monkeypatch):
    module = installer(monkeypatch)
    target = legacy_target(tmp_path)
    changes = module.prepare(target, verify_revision=False)
    before = dict(changes.before)
    replace = module.atomic_replace
    calls = 0

    def fail_second(path, raw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected payload write failure")
        return replace(path, raw)

    monkeypatch.setattr(module, "atomic_replace", fail_second)
    with pytest.raises(OSError, match="injected payload write failure"):
        module.apply_transaction(target, changes)
    for path, raw in before.items():
        assert (path.read_bytes() if path.exists() else None) == raw
    assert (target / "unrelated.txt").read_bytes() == b"unrelated user work\n"


def test_revision_guard_is_not_bypassed(tmp_path, monkeypatch):
    module = installer(monkeypatch)
    # Synthetic Git readback solely to exercise the existing refusal condition.
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="5a7e4418bcebee9c9ceccca4837e04f881d57ecf\n"))
    with pytest.raises(ValueError, match="checkout revision differs"):
        module.prepare(tmp_path)


def test_linked_worktree_guard_is_not_bypassed(tmp_path, monkeypatch):
    module = installer(monkeypatch)
    (tmp_path / ".git").write_text("gitdir: synthetic-placeholder\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worktree or linked Git directory refused"):
        module.prepare(tmp_path, verify_revision=False)


def test_regressions_against_installer_output_not_checkout_imports(tmp_path, monkeypatch):
    module = installer(monkeypatch)
    target = legacy_target(tmp_path)
    module.apply_transaction(target, module.prepare(target, verify_revision=False))
    # An empty package is fixture scaffolding, not a replacement Overwatch runtime.
    (target / "src/overwatch/__init__.py").write_text("", encoding="utf-8")
    consumer = tmp_path / "external-consumer"
    consumer.mkdir()
    driver = '''import pathlib, sys, pytest
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
from overwatch import kev_laya_adapter
from overwatch.providers import kev_laya
assert pathlib.Path(kev_laya_adapter.__file__).resolve() == root / "src/overwatch/kev_laya_adapter.py"
assert pathlib.Path(kev_laya.__file__).resolve() == root / "src/overwatch/providers/kev_laya.py"
raise SystemExit(pytest.main(["-q", "-ra", "--confcutdir=" + str(root / "tests"),
    str(root / "tests/test_kev_laya_cache_versions.py"),
    str(root / "tests/test_kev_laya_wandb_response_guards.py")]))
'''
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH="")
    result = subprocess.run([sys.executable, "-c", driver, str(target)],
                            cwd=consumer, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "94 passed" in result.stdout, result.stdout
    assert "skipped" not in result.stdout, result.stdout
