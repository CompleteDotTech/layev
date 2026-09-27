"""Source-gate mechanics, NOT native Qwen or successful real-weight acceptance.

Positive cases deliberately substitute a tiny test-only weight anchor in memory.
The production CLI has no override and cannot validate these synthetic snapshots.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/kev_laya/backbone_provenance.py"
SCRIPT = ROOT / "scripts/verify_backbone_source.py"
REAL_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
REAL_SIZE = 988097824


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location("tested_backbone_provenance", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_snapshot(tmp_path, gate, monkeypatch):
    """Synthetic content and explicitly overridden anchor: never native evidence."""
    root = tmp_path / "snapshot"
    root.mkdir()
    payloads = {
        "config.json": b'{"synthetic":true}', "tokenizer.json": b'{"test_only":true}',
        "tokenizer_config.json": b'{}', "LICENSE": b'Synthetic fixture, not a license proof.\n',
        "model.safetensors": b'Not safetensors: test byte matching only.' * 4,
    }
    for name, data in payloads.items():
        (root / name).write_bytes(data)
    monkeypatch.setattr(gate, "WEIGHT_SHA256", hashlib.sha256(payloads["model.safetensors"]).hexdigest())
    monkeypatch.setattr(gate, "WEIGHT_BYTES", len(payloads["model.safetensors"]))
    monkeypatch.setattr(gate, "CHUNK_BYTES", 13)
    manifest = {"repository": gate.REPOSITORY, "revision": gate.REVISION, "license": "Apache-2.0",
                "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
    (root / "source.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def replace_manifest(root, manifest):
    (root / "source.json").write_text(json.dumps(manifest, allow_nan=True), encoding="utf-8")


def assert_not_native(report):
    assert report["network_calls"] == report["native_cases_executed"] == 0
    assert report["auxiliary_upstream_origin"] == "unverified"
    assert report["upstream_signature"] == "not_verified"
    assert report["safetensors_structure"] == "not_tested"
    assert report["representative_quality"] == "unmeasured"
    assert report["jev_parity"] == "unknown"
    for name in ("tokenizer_runtime", "pretrained_initialization", "checkpoint_reload", "serving", "cuda", "trained_context"):
        assert report[name] == "not_run"


def test_production_anchor_is_exact_public_lfs_pointer(gate):
    assert gate.REPOSITORY == "Qwen/Qwen2.5-0.5B"
    assert gate.REVISION == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert gate.WEIGHT_SHA256 == REAL_SHA and gate.WEIGHT_BYTES == REAL_SIZE
    assert gate.ANCHOR_URL == f"https://huggingface.co/{gate.REPOSITORY}/raw/{gate.REVISION}/model.safetensors"


def test_no_input_is_blocked_without_io(gate, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected input read")
    monkeypatch.setattr(gate, "_stat", forbidden)
    report = gate.verify_source(None)
    assert report["status"] == "blocked" and report["reason"] == "source_directory_required"
    assert report["accounting"]["bytes_read"] == report["accounting"]["file_opens"] == 0
    assert_not_native(report)


def test_cli_no_input_needs_no_model_packages(tmp_path):
    # -I -S removes environment paths and site-packages, including Torch/pytest.
    run = subprocess.run([sys.executable, "-I", "-S", str(SCRIPT)], cwd=tmp_path,
                         text=True, capture_output=True, timeout=20)
    assert run.returncode == 2 and not run.stderr
    report = json.loads(run.stdout)
    assert report["status"] == "blocked"
    assert report["accounting"]["bytes_read"] == 0
    assert_not_native(report)
    assert list(tmp_path.iterdir()) == []


def test_tiny_positive_is_byte_mechanics_only(gate, tiny_snapshot):
    root, manifest = tiny_snapshot
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = gate.verify_source(root)
    assert result["status"] == "verified" and result["reason"] is None
    assert result["pinned_weight_bytes_verified"] is True
    assert result["all_local_file_hashes_verified"] is True
    assert {r["name"] for r in result["files"]} == set(manifest["sha256"])
    assert result["accounting"]["file_opens"] == 6
    assert result["accounting"]["bytes_read"] == sum(map(len, before.values()))
    assert result["accounting"]["largest_read_bytes"] <= 13
    assert result["source_manifest_sha256"] == hashlib.sha256(before["source.json"]).hexdigest()
    assert all(r["basis"] == ("upstream_anchor" if r["name"] == "model.safetensors" else "local_manifest_only")
               for r in result["files"])
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    assert str(root) not in json.dumps(result)
    assert_not_native(result)


def test_production_cli_rejects_self_consistent_fake_native_source(tiny_snapshot):
    root, _ = tiny_snapshot
    run = subprocess.run([sys.executable, "-I", "-S", str(SCRIPT), "--source", str(root)],
                         text=True, capture_output=True, timeout=20)
    assert run.returncode == 1 and not run.stderr
    result = json.loads(run.stdout)
    assert result["weight_anchor"]["sha256"] == REAL_SHA
    assert result["status"] == "invalid"
    assert result["reason"] == "manifest_weight_differs_from_upstream_anchor"
    assert result["accounting"]["file_opens"] == 1
    assert not result["pinned_weight_bytes_verified"]


@pytest.mark.parametrize("argument", ["--expected-sha256", "--expected-size", "--anchor", "--fixture"])
def test_cli_has_no_pin_override(argument, tmp_path):
    run = subprocess.run([sys.executable, "-I", "-S", str(SCRIPT), argument, "anything"],
                         text=True, capture_output=True, cwd=tmp_path, timeout=20)
    assert run.returncode == 2 and not run.stdout
    assert "unrecognized arguments" in run.stderr


@pytest.mark.parametrize("filename", ["source.json", "config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE", "model.safetensors"])
def test_missing_file_blocks_before_large_read(gate, tiny_snapshot, filename):
    root, _ = tiny_snapshot
    (root / filename).unlink()
    report = gate.verify_source(root)
    assert report["status"] == "blocked" and report["reason"] == "required_input_missing"
    assert report["file"] == filename and report["accounting"]["file_opens"] <= 1


def test_missing_directory_is_not_created(gate, tmp_path):
    path = tmp_path / "missing"
    result = gate.verify_source(path)
    assert result["status"] == "blocked"
    assert result["accounting"]["bytes_read"] == 0 and not path.exists()


@pytest.mark.parametrize("raw, reason", [
    (b'[]', "manifest_fields_mismatch"), (b'null', "manifest_fields_mismatch"),
    (b'"data"', "manifest_fields_mismatch"), (b'{', "invalid_manifest_json"),
    (b'\xff', "invalid_manifest_json"), (b'{"x":1,"x":2}', "duplicate_manifest_key"),
    (b'{"x":NaN}', "nonfinite_manifest_value"),
    (b'{"x":Infinity}', "nonfinite_manifest_value"),
    (b'{"sha256":{"x":1,"x":2}}', "duplicate_manifest_key"),
    pytest.param(b'[' * 2000 + b']' * 2000, None, id="deep-array-runtime-independent"),
])
def test_malformed_manifest_returns_controlled_reason(gate, tiny_snapshot, raw, reason):
    root, _ = tiny_snapshot
    (root / "source.json").write_bytes(raw)
    result = gate.verify_source(root)
    assert result["status"] == "invalid"
    # A deep but valid JSON array may be parsed or hit the runtime recursion limit.
    # Neither path may be accepted as a source manifest.
    assert result["reason"] == reason if reason else result["reason"] in {"manifest_fields_mismatch", "invalid_manifest_json"}
    assert result["accounting"]["file_opens"] == 1


@pytest.mark.parametrize("key,value,reason", [
    ("repository", "other/repo", "repository_or_revision_mismatch"),
    ("revision", "main", "repository_or_revision_mismatch"),
    ("license", "unknown", "license_declaration_mismatch"),
    ("sha256", [], "manifest_file_set_mismatch"),
    ("sha256", None, "manifest_file_set_mismatch"),
])
def test_changed_manifest_identity_rejected(gate, tiny_snapshot, key, value, reason):
    root, manifest = tiny_snapshot
    manifest[key] = value
    replace_manifest(root, manifest)
    result = gate.verify_source(root)
    assert result["status"] == "invalid" and result["reason"] == reason


@pytest.mark.parametrize("value", [True, None, 1, [], {}, "bad", "A" * 64, "0" * 63, "0" * 65])
def test_invalid_hash_values(gate, tiny_snapshot, value):
    root, manifest = tiny_snapshot
    manifest["sha256"]["config.json"] = value
    replace_manifest(root, manifest)
    assert gate.verify_source(root)["reason"] == "invalid_manifest_digest"


@pytest.mark.parametrize("key", ["../outside", "C:/absolute", "secret.txt", "tokenizer\\copy.json"])
def test_no_manifest_selected_paths(gate, tiny_snapshot, key):
    root, manifest = tiny_snapshot
    manifest["sha256"][key] = "0" * 64
    replace_manifest(root, manifest)
    result = gate.verify_source(root)
    assert result["reason"] == "manifest_file_set_mismatch"
    assert key not in json.dumps(result)


def test_extra_manifest_content_is_not_echoed(gate, tiny_snapshot):
    root, manifest = tiny_snapshot
    secret = "private-value-must-not-appear-in-receipt"
    manifest["unexpected"] = secret
    replace_manifest(root, manifest)
    result = gate.verify_source(root)
    assert result["reason"] == "manifest_fields_mismatch"
    assert secret not in json.dumps(result)


@pytest.mark.parametrize("filename", ["source.json", "config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE", "model.safetensors"])
def test_empty_files_are_invalid(gate, tiny_snapshot, filename):
    root, _ = tiny_snapshot
    (root / filename).write_bytes(b'')
    result = gate.verify_source(root)
    assert result["status"] == "invalid" and result["file"] == filename


@pytest.mark.parametrize("filename", ["source.json", "config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE", "model.safetensors"])
def test_oversize_rejection_happens_before_open(gate, tiny_snapshot, monkeypatch, filename):
    root, _ = tiny_snapshot
    if filename == "source.json":
        monkeypatch.setattr(gate, "MANIFEST_MAX_BYTES", 1)
    elif filename in gate.AUXILIARY_LIMITS:
        monkeypatch.setitem(gate.AUXILIARY_LIMITS, filename, 1)
        (root / filename).write_bytes(b'too large')
    else:
        (root / filename).write_bytes(b'x' * (gate.WEIGHT_BYTES + 1))
    result = gate.verify_source(root)
    assert result["reason"] == "input_exceeds_size_limit" and result["file"] == filename
    assert result["accounting"]["file_opens"] <= 1


def test_correct_claimed_digest_wrong_size_rejected(gate, tiny_snapshot):
    root, _ = tiny_snapshot
    (root / "model.safetensors").write_bytes(b'x')
    result = gate.verify_source(root)
    assert result["reason"] == "upstream_weight_size_mismatch"
    assert result["accounting"]["file_opens"] == 1


@pytest.mark.parametrize("filename", ["config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE", "model.safetensors"])
def test_same_length_corruption_fails_digest(gate, tiny_snapshot, filename):
    root, _ = tiny_snapshot
    path = root / filename
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    result = gate.verify_source(root)
    assert result["reason"] == "file_digest_mismatch" and result["file"] == filename
    assert not result["all_local_file_hashes_verified"] and not result["pinned_weight_bytes_verified"]


def make_symlink(path, target, *, directory=False):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege unavailable; not a source/native pass")
        raise


@pytest.mark.parametrize("filename", ["source.json", "config.json", "model.safetensors"])
def test_linked_files_refused(gate, tiny_snapshot, filename):
    root, _ = tiny_snapshot
    path = root / filename
    target = root.parent / "original"
    path.replace(target)
    make_symlink(path, target)
    result = gate.verify_source(root)
    assert result["reason"] == "links_or_reparse_points_not_supported"


def test_linked_root_refused(gate, tiny_snapshot):
    root, _ = tiny_snapshot
    path = root.parent / "linked"
    make_symlink(path, root, directory=True)
    assert gate.verify_source(path)["reason"] == "links_or_reparse_points_not_supported"


def test_linked_ancestor_refused(gate, tiny_snapshot):
    root, _ = tiny_snapshot
    path = root.parent / "linked-parent"
    make_symlink(path, root.parent, directory=True)
    assert gate.verify_source(path / root.name)["reason"] == "links_or_reparse_points_not_supported"


def test_windows_reparse_bit_refused_on_any_host(gate, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "lstat", lambda _: SimpleNamespace(st_mode=0o100600, st_file_attributes=0x400))
    with pytest.raises(gate.SourceGateError, match="links_or_reparse"):
        gate._stat(tmp_path / "not-followed")


def test_directory_cannot_replace_required_file(gate, tiny_snapshot):
    root, _ = tiny_snapshot
    path = root / "model.safetensors"
    path.unlink()
    path.mkdir()
    assert gate.verify_source(root)["reason"] == "input_not_regular_file"


def test_fifo_not_opened(gate, tiny_snapshot):
    if not hasattr(os, "mkfifo"):
        pytest.skip("host has no FIFO support")
    root, _ = tiny_snapshot
    path = root / "model.safetensors"
    path.unlink()
    os.mkfifo(path)
    result = gate.verify_source(root)
    assert result["reason"] == "input_not_regular_file" and result["accounting"]["file_opens"] == 1


def test_changed_file_between_admission_and_read(gate, tiny_snapshot, monkeypatch):
    root, _ = tiny_snapshot
    original = gate._read
    def changed(path, before, accounting, **kwargs):
        if path.name == "config.json":
            path.write_bytes(b'changed')
        return original(path, before, accounting, **kwargs)
    monkeypatch.setattr(gate, "_read", changed)
    assert gate.verify_source(root)["reason"] == "input_changed_before_read"


def test_changed_manifest_before_completion(gate, tiny_snapshot, monkeypatch):
    root, manifest = tiny_snapshot
    original = gate._read
    def changed(path, before, accounting, **kwargs):
        value = original(path, before, accounting, **kwargs)
        if path.name == "model.safetensors":
            manifest["license"] = "modified"
            replace_manifest(root, manifest)
        return value
    monkeypatch.setattr(gate, "_read", changed)
    assert gate.verify_source(root)["reason"] == "input_changed_before_completion"


def test_replaced_file_after_read_is_refused(gate, tiny_snapshot, monkeypatch):
    root, _ = tiny_snapshot
    original = gate._stat
    count = 0
    def replacement(path, name=None):
        nonlocal count
        if name == "model.safetensors":
            count += 1
            if count == 2:
                temp = root / "replacement"
                temp.write_bytes((root / name).read_bytes())
                temp.replace(root / name)
        return original(path, name)
    monkeypatch.setattr(gate, "_stat", replacement)
    assert gate.verify_source(root)["reason"] == "input_replaced_during_read"


@pytest.mark.parametrize("kind", ["grow", "shrink", "io-error"])
def test_bounded_stream_failure(gate, tiny_snapshot, monkeypatch, kind):
    root, _ = tiny_snapshot
    original = gate.os.fdopen
    class Instrumented:
        def __init__(self, stream):
            self.stream = stream
            self.changed = False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def read(self, size):
            assert 0 < size <= gate.CHUNK_BYTES
            if kind == "io-error":
                raise OSError("private content MUST NOT be in receipt")
            value = self.stream.read(size)
            if not self.changed:
                self.changed = True
                path = root / "model.safetensors"
                if kind == "grow":
                    with path.open("ab") as out:
                        out.write(b'x')
                else:
                    with path.open("r+b") as out:
                        out.truncate(1)
            return value
    def fdopen(fd, mode):
        stream = original(fd, mode)
        if os.fstat(fd).st_size == gate.WEIGHT_BYTES:
            return Instrumented(stream)
        return stream
    monkeypatch.setattr(gate.os, "fdopen", fdopen)
    result = gate.verify_source(root)
    assert result["status"] != "verified"
    assert result["reason"] in {"input_grew_during_read", "input_changed_during_read", "input_read_failed"}
    assert "MUST NOT" not in json.dumps(result)
    assert result["accounting"]["largest_read_bytes"] <= gate.CHUNK_BYTES


def test_permission_error_receipt_has_no_error_details(gate, tiny_snapshot, monkeypatch):
    root, _ = tiny_snapshot
    def denied(*args, **kwargs):
        raise PermissionError("sensitive OS detail")
    monkeypatch.setattr(gate.os, "open", denied)
    result = gate.verify_source(root)
    assert result["status"] == "blocked" and result["reason"] == "input_open_failed"
    assert "sensitive" not in json.dumps(result)


def test_module_does_not_load_models_or_network(gate, tiny_snapshot, monkeypatch):
    root, _ = tiny_snapshot
    def network_forbidden(*args, **kwargs):
        raise AssertionError("unexpected network access")
    monkeypatch.setattr(socket, "socket", network_forbidden)
    before = set(sys.modules)
    assert gate.verify_source(root)["status"] == "verified"
    imported = set(sys.modules) - before
    assert not any(name.split(".")[0] in {"torch", "transformers", "safetensors", "tokenizers"} for name in imported)


@pytest.mark.parametrize("source", [1, [], {}, "bad\x00path"])
def test_invalid_location_does_not_escape_or_echo(gate, source):
    result = gate.verify_source(source)
    assert result["status"] == "invalid" and result["reason"] == "invalid_source_location"


def test_parent_traversal_rejected(gate, tiny_snapshot):
    root, _ = tiny_snapshot
    assert gate.verify_source(root / ".." / root.name)["reason"] == "parent_traversal_not_supported"


def test_main_exit_codes(gate, tiny_snapshot, capsys):
    root, _ = tiny_snapshot
    assert gate.main([]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert gate.main(["--source", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    (root / "model.safetensors").write_bytes(b'x')
    assert gate.main(["--source", str(root)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"
