"""Real filesystem transactions; fixture anchors are not a live Overwatch pass."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def installer():
    spec = importlib.util.spec_from_file_location(
        "stage4_installer", ROOT / "integrations/overwatch/apply.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_checkout(module, root, monkeypatch, *, old_import=False, crlf=False):
    name = "src/overwatch/frontend/App.tsx"
    base = '{report && page === "resources" && <ResourcesPage report={report} />}\n'
    monkeypatch.setattr(module, "BLOBS", {name: module.blob_sha(base.encode())})
    text = module.transform(name, base) if old_import else base
    if old_import:
        text = text.replace("./ModelRunsView", "./ModelRuns")
    raw = text.replace("\n", "\r\n").encode() if crlf else text.encode()
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    return path, raw


@pytest.mark.parametrize("crlf", [False, True])
@pytest.mark.parametrize("old_import", [False, True])
def test_fresh_or_known_upgrade_removes_obsolete_and_is_idempotent(
    tmp_path, monkeypatch, crlf, old_import
):
    module = installer()
    app, original = seed_checkout(module, tmp_path, monkeypatch, old_import=old_import, crlf=crlf)
    obsolete = app.with_name("ModelRuns.tsx")
    if old_import:
        # These are exact known stage-three component bytes, not an invented hash.
        raw = (module.HERE / "added/src/overwatch/frontend/ModelRunsView.tsx").read_bytes()
        obsolete.write_bytes(raw.replace(b"\n", b"\r\n") if crlf else raw)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_bytes(b"preserve me\r\n")
    changes = module.prepare(tmp_path, verify_revision=False)
    assert app.read_bytes() == original  # --check is read-only
    assert (obsolete in changes and changes[obsolete] is None) == old_import
    backup = module.apply_transaction(tmp_path, changes)
    assert backup.is_dir()
    assert not obsolete.exists()
    assert app.with_name("ModelRunsView.tsx").exists()
    assert './ModelRunsView' in app.read_text()
    assert unrelated.read_bytes() == b"preserve me\r\n"
    assert module.prepare(tmp_path, verify_revision=False) == {}
    if crlf:
        assert b"\r\n" in app.read_bytes()


def test_modified_obsolete_component_refused_before_writes(tmp_path, monkeypatch):
    module = installer()
    app, original = seed_checkout(module, tmp_path, monkeypatch, old_import=True)
    obsolete = app.with_name("ModelRuns.tsx")
    obsolete.write_bytes(b"unrelated component edits\n")
    with pytest.raises(ValueError, match="refusing deletion"):
        module.prepare(tmp_path, verify_revision=False)
    assert app.read_bytes() == original
    assert not app.with_name("ModelRunsView.tsx").exists()


def test_edit_after_preflight_preserved(tmp_path, monkeypatch):
    module = installer()
    app, _ = seed_checkout(module, tmp_path, monkeypatch)
    changes = module.prepare(tmp_path, verify_revision=False)
    app.write_bytes(b"new unsaved review changes\n")
    with pytest.raises(ValueError, match="changed since preflight"):
        module.apply_transaction(tmp_path, changes)
    assert app.read_bytes() == b"new unsaved review changes\n"
    assert not list(tmp_path.glob(".kev-laya-stage3-backup-*"))


def test_failed_upgrade_rolls_back_deleted_component(tmp_path, monkeypatch):
    module = installer()
    app, original = seed_checkout(module, tmp_path, monkeypatch, old_import=True)
    obsolete = app.with_name("ModelRuns.tsx")
    component = (module.HERE / "added/src/overwatch/frontend/ModelRunsView.tsx").read_bytes()
    obsolete.write_bytes(component)
    changes = module.prepare(tmp_path, verify_revision=False)
    replace = module.atomic_replace
    deletion_observed = []

    def fail_new_component(target, raw):
        if target.name == "ModelRunsView.tsx":
            deletion_observed.append(not obsolete.exists())
            raise OSError("injected write failure")
        replace(target, raw)

    monkeypatch.setattr(module, "atomic_replace", fail_new_component)
    with pytest.raises(OSError, match="injected write failure"):
        module.apply_transaction(tmp_path, changes)
    assert deletion_observed == [True]
    assert app.read_bytes() == original
    assert obsolete.read_bytes() == component
    assert not app.with_name("ModelRunsView.tsx").exists()
    assert not list(app.parent.glob(".kev-laya-*"))


def test_unknown_case_variant_is_not_silently_installed(tmp_path, monkeypatch):
    module = installer()
    app, original = seed_checkout(module, tmp_path, monkeypatch)
    app.with_name("MODELRUNS.tsx").write_bytes(b"unknown\n")
    with pytest.raises(ValueError, match="unexpected obsolete component spelling"):
        module.prepare(tmp_path, verify_revision=False)
    assert app.read_bytes() == original


def test_known_v1_import_restores_only_exact_base(tmp_path, monkeypatch):
    module = installer()
    app, _ = seed_checkout(module, tmp_path, monkeypatch)
    name = app.relative_to(tmp_path).as_posix()
    base = app.read_text()
    operations = [[a, b.replace("./ModelRunsView", "./ModelRuns")] for a, b in module.LEGACY[name]]
    old = module.apply_edits(base, operations)
    assert module.base_text(name, old.encode()) == base
    with pytest.raises(ValueError, match="modified or unexpected"):
        module.base_text(name, (old + "// unrelated changes\n").encode())


@pytest.mark.parametrize("crlf", [False, True])
def test_hash_verified_original_v1_component_migrates(tmp_path, monkeypatch, crlf):
    module = installer()
    app, _ = seed_checkout(module, tmp_path, monkeypatch, old_import=True, crlf=crlf)
    obsolete = app.with_name("ModelRuns.tsx")
    raw = (ROOT / "tests/fixtures/overwatch-v1-component.txt").read_bytes()
    obsolete.write_bytes(raw.replace(b"\n", b"\r\n") if crlf else raw)
    changes = module.prepare(tmp_path, verify_revision=False)
    assert changes[obsolete] is None
    module.apply_transaction(tmp_path, changes)
    assert not obsolete.exists()
    assert module.prepare(tmp_path, verify_revision=False) == {}
