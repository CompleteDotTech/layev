"""Revision-checked primary-checkout integration, including a checked v1-to-v2 upgrade.

No files are written by --check. --apply backs up each touched file and rolls back
on a failed write. Unknown edits, worktrees, symlinks and review paths are refused.
The original provider filters and default cloud collection remain unchanged.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

BASE = "501bd99a0cb0c6b321feb022f5747d6dcbd5e9c4"
HERE = Path(__file__).resolve().parent
BLOBS = {
    "src/overwatch/collector.py": "34f87e1bd2c79bd3e101d1c06bc234eb98db2672",
    "src/overwatch/raw_cache.py": "0dd58422e416ca78e0699c5b14b9c25df40016de",
    "src/overwatch/cached_metrics.py": "b816fa7ff81923756bc4f5ecb7cfa720baf43bc1",
    "src/overwatch/report.py": "e2a956286135822d07e071f001565201da489144",
    "src/overwatch/report_service.py": "745ca9391e620adf4dc73b75748736784dc7ba0a",
    "src/overwatch/constants.py": "ca8e54b3c21701ee1599be81c5db1c72fd5cf57e",
    "src/overwatch/frontend/types.ts": "e53bbe349ecd739074201c7476de78b30e926eba",
    "src/overwatch/frontend/App.tsx": "38254c3a18a797dbd5eb932c911b103f5c239e78",
}
LEGACY = json.loads((HERE / 'legacy-edits.json').read_text())
LOCAL_COLLECTION = '''    # Explicit offline/local mode: use the REAL collector and cache without cloud I/O.
    # Default behavior and existing Flow/resource records remain unchanged.
    if options.local_models_only:
        if job_id is not None:
            raise ValueError("local-models-only cannot refresh a SkyPilot job")
        result = await collect_source("kev_laya", collect_kev_laya_cache)
        if result is not None:
            source_status["kev_laya"]["raw"] = result
            if result["warnings"]:
                source_status["kev_laya"]["status"] = "warning"
        for key in ("sky_jobs", "sky_clusters", "aws_billing", "gcp_billing", "wandb", "cloudwatch"):
            source_status[key] = {"status": "skipped", "duration_seconds": 0.0,
                                  "error": None, "reason": "explicit local-models-only"}
        manifest = {
            "cache_spec_version": RAW_CACHE_SPEC_VERSION, "cache_root": str(RAW_CACHE_ROOT),
            "updated_at": isoformat(datetime.now(UTC)),
            "duration_seconds": round(perf_counter() - collection_started, 3),
            "scope": {"job_id": None, "full_logs": False, "local_models_only": True},
            "sources": source_status,
        }
        write_json_atomically(manifest_path, manifest)
        return manifest

'''
COLLECTION_RESULT = '''    return {"runs": len(envelope["records"]), "warnings": len(envelope["warnings"]),
            "complete": collected.get("complete", False),
            "accounting": collected.get("accounting", {}),
            "source_status": collected.get("sources", []),
            "collection_warnings": collected.get("warnings", [])}'''


def edits(name: str) -> list[list[str]]:
    """Exact transformations on the published base, not AST guesses about changed code."""
    changes = [pair[:] for pair in LEGACY[name]]
    if name.endswith('/collector.py'):
        changes += [
            ['    gcp_billing_table: str | None = None',
             '    gcp_billing_table: str | None = None\n    local_models_only: bool = False'],
            ['    # A scoped job refresh reuses inventory and only advances that job\'s raw log cursor.',
             LOCAL_COLLECTION + '    # A scoped job refresh reuses inventory and only advances that job\'s raw log cursor.'],
            ['    path = model_runs_cache_path()\n    collected = collect_snapshots()',
             '    path = model_runs_cache_path()\n    collected = collect_snapshots()'],
            ['        envelope = merge_snapshots(read_json(path, {}), collected,',
             '        from overwatch.raw_cache import read_model_runs_cache\n        envelope = merge_snapshots(read_model_runs_cache(), collected,'],
            ['    return {"runs": len(envelope["records"]), "warnings": len(envelope["warnings"])}', COLLECTION_RESULT],
            ['        if kev_laya_result and kev_laya_result["warnings"]:\n            source_status["kev_laya"]["status"] = "warning"',
             '        if kev_laya_result is not None:\n            source_status["kev_laya"]["raw"] = kev_laya_result\n            if kev_laya_result["warnings"]:\n                source_status["kev_laya"]["status"] = "warning"'],
            ['    parser.add_argument("--full-logs", action="store_true")',
             '    parser.add_argument("--full-logs", action="store_true")\n    parser.add_argument("--local-models-only", action="store_true",\n                        default=os.environ.get("OVERWATCH_LOCAL_MODELS_ONLY") == "1")'],
            ['        gcp_billing_table=args.gcp_billing_table,',
             '        gcp_billing_table=args.gcp_billing_table,\n        local_models_only=args.local_models_only,'],
        ]
    elif name.endswith('/raw_cache.py'):
        changes += [
            ['import json\n', 'import json\nimport os\n'],
            ['RAW_CACHE_ROOT = Path.home() / ".cache" / "overwatch" / "raw-v1"',
             'RAW_CACHE_ROOT = Path(os.environ.get("OVERWATCH_RAW_CACHE_ROOT", str(Path.home() / ".cache" / "overwatch" / "raw-v1"))).expanduser()'],
            ['"runs-v1.json"', '"runs-v2.json"'],
            ['<SUFFIX>', '''\n\ndef read_model_runs_cache() -> dict:
    """Cache-only migration read. Never reread producer exports during rendering.

    Existing v1 files are immutable migration inputs; the collector writes v2.
    A corrupt v2 does NOT silently roll the current view back to an old v1.
    """
    path = model_runs_cache_path()
    if not path.exists():
        path = raw_cache_path("global", "kev_laya", "runs-v1.json")
    value = read_json(path)
    if isinstance(value, dict) and isinstance(value.get("schema_version"), str) and value["schema_version"] in {"kev_laya/raw/1", "kev_laya/raw/2"}:
        return value
    warning = [{"code": "model_run_cache_unreadable_or_incompatible", "source": "cache"}] if path.exists() else []
    return {"schema_version": "kev_laya/raw/2", "records": [], "observations": [], "warnings": warning}
'''],
        ]
    elif name.endswith('/cached_metrics.py'):
        changes += [
            ['    from overwatch.raw_cache import model_runs_cache_path\n    from overwatch.kev_laya_adapter import CACHE_VERSION\n\n    return read_json(model_runs_cache_path(), {\n        "schema_version": CACHE_VERSION, "records": [], "warnings": []\n    })',
             '    from overwatch.raw_cache import read_model_runs_cache\n\n    return read_model_runs_cache()'],
        ]
    elif name.endswith('/report_service.py'):
        changes += [
            ['"runs-v1.json"', '"runs-v2.json"'],
            ['        limit=args.limit,', '        limit=args.limit,\n        local_models_only=bool(getattr(args, "local_models_only", False)) or os.environ.get("OVERWATCH_LOCAL_MODELS_ONLY") == "1",'],
        ]
    return [(a, b) for a, b in changes if a != b]


def apply_edits(text: str, changes, *, reverse=False) -> str:
    for old, new in reversed(changes) if reverse else changes:
        if old == '<PREFIX>':
            if reverse:
                if not text.startswith(new): raise ValueError('missing integration prefix')
                text = text[len(new):]
            else: text = new + text
        elif old == '<SUFFIX>':
            if reverse:
                if not text.endswith(new): raise ValueError('missing integration suffix')
                text = text[:-len(new)]
            else: text += new
        else:
            a, b = (new, old) if reverse else (old, new)
            if text.count(a) != 1: raise ValueError(f'patch anchor not unique: {a[:70]!r}')
            text = text.replace(a, b, 1)
    return text


def transform(name: str, text: str) -> str:
    return apply_edits(text, edits(name))


def blob_sha(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def base_text(name: str, raw: bytes) -> str:
    text = raw.replace(b'\r\n', b'\n').decode('utf-8')
    if blob_sha(text.encode()) == BLOBS[name]: return text
    # Exact old integration or exact current integration is upgradable/idempotent.
    for operations in (edits(name), LEGACY[name]):
        try:
            base = apply_edits(text, operations, reverse=True)
            if blob_sha(base.encode()) == BLOBS[name] and apply_edits(base, operations) == text:
                return base
        except ValueError:
            continue
    raise ValueError(f'modified or unexpected target; unchanged: {name}')


def relative_path_key(path, root) -> str:
    """Manifest keys stay POSIX on Windows as well as Linux."""
    return path.relative_to(root).as_posix()


def prepare(root: Path, *, verify_revision=True) -> dict[Path, bytes]:
    root = root.absolute()
    if any(part.casefold() in {'reviews', 'review'} for part in root.parts):
        raise ValueError('review snapshots are immutable; select the development checkout')
    if root.is_symlink() or root.resolve() != root:
        raise ValueError('symlink checkout paths are refused')
    if (root / '.git').is_file():
        raise ValueError('worktree or linked Git directory refused; use the primary checkout')
    if verify_revision:
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                                capture_output=True, text=True, timeout=10)
        if result.stdout.strip() != BASE:
            raise ValueError('checkout revision differs; review/rebase these changes instead of forcing application')
    changes = {}
    def safe_target(path):
        for parent in (path, *path.parents):
            if parent == root.parent: break
            if parent.is_symlink(): raise ValueError(f'symlink target refused: {path}')
    for name in BLOBS:
        path = root / name
        safe_target(path)
        raw = path.read_bytes()
        new = transform(name, base_text(name, raw)).encode('utf-8')
        if b'\r\n' in raw: new = new.replace(b'\n', b'\r\n')
        if new != raw: changes[path] = new
    previous = json.loads((HERE / 'previous-v1-added-hashes.json').read_text())
    for source in sorted((HERE / 'added').rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts: continue
        relative = relative_path_key(source, HERE / 'added')
        target, raw = root / relative, source.read_bytes()
        safe_target(target)
        if target.exists():
            old = target.read_bytes()
            if old.replace(b'\r\n', b'\n') == raw.replace(b'\r\n', b'\n'): continue
            if hashlib.sha256(old.replace(b'\r\n', b'\n')).hexdigest() != previous.get(relative):
                raise ValueError(f'refusing to overwrite existing added file: {target}')
            if b'\r\n' in old: raw = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        changes[target] = raw
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--apply', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.root.absolute()
    changes = prepare(root)
    if not args.apply or not changes:
        print(json.dumps({'status': 'preflight passed' if changes else 'already applied',
                          'base_revision': BASE, 'files': [str(p.relative_to(root)) for p in changes]}, indent=2))
        return 0
    backup = root / ('.kev-laya-stage3-backup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    backup.mkdir()
    existed = {}
    for target in changes:
        rel = target.relative_to(root)
        existed[str(rel)] = target.exists()
        if target.exists():
            copy = backup / rel
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, copy)
    (backup / 'manifest.json').write_text(json.dumps(existed, indent=2))
    written = []
    try:
        for target, raw in changes.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=target.parent, prefix='.kev-laya-')
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                os.replace(temp, target); written.append(target)
            finally:
                if os.path.exists(temp): os.unlink(temp)
    except BaseException:
        for target in written:
            rel = str(target.relative_to(root))
            if existed[rel]: shutil.copy2(backup / rel, target)
            else: target.unlink(missing_ok=True)
        raise
    print(json.dumps({'status': 'applied', 'backup': str(backup), 'files': len(changes)}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
