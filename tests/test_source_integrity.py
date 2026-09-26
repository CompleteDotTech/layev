"""Publication checks must inspect index bytes, not merely the working tree."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def checker():
    spec = importlib.util.spec_from_file_location("source_checker", ROOT / "scripts/check_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_files():
    names = ["LICENSE", "NOTICE", "licenses/Kev-LICENSE", "licenses/Laya-LICENSE",
             "src/kev_laya/telemetry_contract.py",
             "integrations/overwatch/added/src/overwatch/kev_laya_contract.py"]
    return {name: (ROOT / name).read_bytes() for name in names}


def test_valid_sources_and_crlf_identity():
    module = checker()
    files = valid_files()
    clean = module.audit(files)
    crlf = module.audit({k: v.replace(b"\n", b"\r\n") for k, v in files.items()})
    assert clean["passed"] and crlf["passed"]
    assert clean["lf_normalized_tree_sha256"] == crlf["lf_normalized_tree_sha256"]


@pytest.mark.parametrize("name", ["../outside.py", "C:/absolute.py", "bad\\file.py", "trailing."])
def test_reject_nonportable_names(name):
    assert not checker().portable_path(name)


@pytest.mark.parametrize("name", [".env", ".env.production", "checkpoint.pt", "runs/export.json"])
def test_reject_private_generated_and_weight_files(name):
    files = valid_files() | {name: b"not for publication\n"}
    assert not checker().audit(files)["passed"]


def test_credential_pattern_does_not_echo_value():
    token = "ghp_" + "x" * 36
    report = checker().audit(valid_files() | {"unexpected.txt": token.encode()})
    assert not report["passed"]
    assert token not in json.dumps(report)


def test_non_example_dotenv_allowed_only_exact_example():
    assert checker().audit(valid_files() | {".env.example": b"API_KEY=replace-me\n"})["passed"]


def test_telemetry_copy_and_license_mutations_fail():
    for name in ("licenses/Kev-LICENSE", "integrations/overwatch/added/src/overwatch/kev_laya_contract.py"):
        files = valid_files()
        files[name] += b"\n# drift\n"
        assert not checker().audit(files)["passed"]


def test_typescript_stem_collision_detected_on_all_platforms():
    report = checker().audit(valid_files() | {"ui/ModelRuns.tsx": b"", "ui/modelRuns.ts": b""})
    assert any("ambiguous TypeScript" in error for error in report["errors"])


def test_relative_links_checked_without_fetching_external_urls():
    files = valid_files() | {"docs/a.md": b"[license](../LICENSE) [external](https://example.org)"}
    assert checker().audit(files)["passed"]
    files["docs/a.md"] += b" [missing](missing.md)"
    assert not checker().audit(files)["passed"]


def test_index_bytes_not_working_tree_are_scanned(tmp_path):
    if not shutil.which("git"):
        pytest.skip("Git executable required to exercise actual index semantics")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "example.txt"
    token = "ghp_" + "a" * 36
    path.write_text(token)
    subprocess.run(["git", "-C", str(tmp_path), "add", "example.txt"], check=True)
    path.write_text("clean working tree, unsafe index")
    module = checker()
    captured = module.index_files(tmp_path)
    assert captured["example.txt"] == token.encode()
    assert not module.audit(valid_files() | captured)["passed"]


def archive_fixture(path, *, extra=False):
    raw = b"source\n"
    manifest = {"file_count": 1, "files": {"src.py": {
        "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle/src.py", raw)
        archive.writestr("bundle/MANIFEST.json", json.dumps(manifest))
        if extra:
            archive.writestr("bundle/undeclared.py", "unreviewed")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_coverage_and_checksum(tmp_path):
    module = checker()
    path = tmp_path / "delivery.zip"
    expected = archive_fixture(path)
    assert module.verify_archive(path, expected)["verified_payload_files"] == 1
    with pytest.raises(ValueError, match="checksum"):
        module.verify_archive(path, "0" * 64)
    expected = archive_fixture(path, extra=True)
    with pytest.raises(ValueError, match="coverage"):
        module.verify_archive(path, expected)


@pytest.mark.parametrize("name", ["CON.py", "aux.txt", "dir/LPT1", "bad?.py", "bad|.txt"])
def test_windows_reserved_names_rejected_on_linux_too(name):
    assert not checker().portable_path(name)


def test_snapshot_byte_budget_applies_before_reading(tmp_path, monkeypatch):
    module = checker()
    monkeypatch.setattr(module, "MAX_SOURCE_BYTES", 4)
    (tmp_path / "large.txt").write_bytes(b"too large")
    with pytest.raises(ValueError, match="before reading contents"):
        module.snapshot_files(tmp_path)


def test_index_byte_budget_applies_before_content_fetch(tmp_path, monkeypatch):
    if not shutil.which("git"):
        pytest.skip("Git executable required for index admission checks")
    module = checker()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "large.txt").write_bytes(b"too large")
    subprocess.run(["git", "-C", str(tmp_path), "add", "large.txt"], check=True)
    monkeypatch.setattr(module, "MAX_SOURCE_BYTES", 4)
    with pytest.raises(ValueError, match="before reading contents"):
        module.index_files(tmp_path)
