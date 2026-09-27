"""Read-only byte provenance gate; never imports or runs a model.

Only model.safetensors has an independent upstream digest here. Other files are
checked against the local acquisition manifest, not authenticated as upstream.
This is a cooperative stable-files check, not an adversarial filesystem snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

GATE_VERSION = "pinned-backbone-source/1"
REPOSITORY = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
WEIGHT_BYTES = 988097824
ANCHOR_URL = f"https://huggingface.co/{REPOSITORY}/raw/{REVISION}/model.safetensors"
CHUNK_BYTES = 1024 * 1024
MANIFEST_MAX_BYTES = 64 * 1024
AUXILIARY_LIMITS = {
    "config.json": 64 * 1024,
    "tokenizer.json": 32 * 1024 * 1024,
    "tokenizer_config.json": 256 * 1024,
    "LICENSE": 64 * 1024,
}
REQUIRED_FILES = frozenset((*AUXILIARY_LIMITS, "model.safetensors"))
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class SourceGateError(ValueError):
    """Only controlled codes/fixed filenames may appear in public receipts."""

    def __init__(self, reason: str, filename: str | None = None, *, blocked=False):
        self.reason = reason
        self.filename = filename
        self.blocked = blocked
        super().__init__(reason)


def _stat(path: Path, name: str | None = None):
    try:
        value = path.lstat()
    except FileNotFoundError:
        raise SourceGateError("required_input_missing", name, blocked=True) from None
    except OSError:
        raise SourceGateError("input_unreadable", name, blocked=True) from None
    if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
        # FILE_ATTRIBUTE_REPARSE_POINT covers Windows junctions as well as links.
        raise SourceGateError("links_or_reparse_points_not_supported", name)
    return value


def _stamp(value):
    # Windows ctime is creation time and can differ between lstat and fstat
    # immediately after a new file is written. Identity, size, and mtime remain
    # the useful cross-handle checks there; POSIX ctime also detects metadata edits.
    stamp = (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)
    return stamp if os.name == "nt" else (*stamp, value.st_ctime_ns)


def _directories(root: Path):
    chain = [*reversed(root.parents), root]
    guards = []
    for path in chain:
        value = _stat(path)
        if not stat.S_ISDIR(value.st_mode):
            raise SourceGateError("source_parent_not_directory")
        guards.append((path, (value.st_dev, value.st_ino, value.st_mode)))
    return guards


def _check_directories(guards):
    for path, expected in guards:
        value = _stat(path)
        if (value.st_dev, value.st_ino, value.st_mode) != expected:
            raise SourceGateError("source_directory_changed")


def _regular_file(path: Path, maximum: int, exact: int | None = None):
    value = _stat(path, path.name)
    if not stat.S_ISREG(value.st_mode):
        raise SourceGateError("input_not_regular_file", path.name)
    if value.st_size > maximum:
        raise SourceGateError("input_exceeds_size_limit", path.name)
    if exact is not None and value.st_size != exact:
        raise SourceGateError("upstream_weight_size_mismatch", path.name)
    if value.st_size == 0:
        raise SourceGateError("empty_input", path.name)
    return value


def _read(path: Path, before, accounting: dict, *, retain=False):
    """Hash one opened file in bounded reads, checking descriptor and pathname."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise SourceGateError("input_open_failed", path.name, blocked=True) from None
    accounting["file_opens"] += 1
    pieces = [] if retain else None
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(descriptor, "rb") as stream:
            if _stamp(os.fstat(stream.fileno())) != _stamp(before):
                raise SourceGateError("input_changed_before_read", path.name)
            while True:
                # At most one byte beyond the inspected size detects growth.
                limit = min(CHUNK_BYTES, before.st_size - count + 1)
                raw = stream.read(limit)
                accounting["read_calls"] += 1
                accounting["bytes_read"] += len(raw)
                accounting["largest_read_bytes"] = max(accounting["largest_read_bytes"], len(raw))
                if not raw:
                    break
                count += len(raw)
                if count > before.st_size:
                    raise SourceGateError("input_grew_during_read", path.name)
                digest.update(raw)
                if pieces is not None:
                    pieces.append(raw)
            if count != before.st_size or _stamp(os.fstat(stream.fileno())) != _stamp(before):
                raise SourceGateError("input_changed_during_read", path.name)
    except OSError:
        raise SourceGateError("input_read_failed", path.name, blocked=True) from None
    if _stamp(_stat(path, path.name)) != _stamp(before):
        raise SourceGateError("input_replaced_during_read", path.name)
    return digest.hexdigest(), b"".join(pieces) if pieces is not None else None


def _strict_manifest(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise SourceGateError("duplicate_manifest_key", "source.json")
            result[key] = value
        return result

    def constant(_):
        raise SourceGateError("nonfinite_manifest_value", "source.json")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, SourceGateError):
            raise
        raise SourceGateError("invalid_manifest_json", "source.json") from None
    if not isinstance(value, dict) or set(value) != {"repository", "revision", "license", "sha256"}:
        raise SourceGateError("manifest_fields_mismatch", "source.json")
    if value["repository"] != REPOSITORY or value["revision"] != REVISION:
        raise SourceGateError("repository_or_revision_mismatch", "source.json")
    if value["license"] != "Apache-2.0":
        raise SourceGateError("license_declaration_mismatch", "source.json")
    hashes = value["sha256"]
    if not isinstance(hashes, dict) or set(hashes) != REQUIRED_FILES:
        raise SourceGateError("manifest_file_set_mismatch", "source.json")
    if any(not isinstance(item, str) or not _DIGEST.fullmatch(item) for item in hashes.values()):
        raise SourceGateError("invalid_manifest_digest", "source.json")
    if hashes["model.safetensors"] != WEIGHT_SHA256:
        raise SourceGateError("manifest_weight_differs_from_upstream_anchor", "model.safetensors")
    return value


def verify_source(directory: str | Path | None) -> dict:
    """Return a content-free receipt. No downloads, model parsing, or file writes.

    A verified receipt means the weight bytes match the frozen public LFS pointer
    and all five files match the local manifest. It does NOT authenticate the four
    auxiliary files, execute Safetensors/Qwen, inspect hardware, or attest a run.
    """
    report = {
        "gate_version": GATE_VERSION, "status": "blocked", "reason": None,
        "repository": REPOSITORY, "revision": REVISION,
        "weight_anchor": {"source": ANCHOR_URL, "sha256": WEIGHT_SHA256,
                          "size_bytes": WEIGHT_BYTES, "basis": "pinned_revision_lfs_pointer"},
        "source_manifest_sha256": None, "files": [],
        "accounting": {"file_opens": 0, "read_calls": 0, "bytes_read": 0,
                       "largest_read_bytes": 0, "chunk_limit_bytes": CHUNK_BYTES},
        "scope": "upstream_weight_bytes_and_local_manifest_integrity",
        "pinned_weight_bytes_verified": False, "all_local_file_hashes_verified": False,
        "auxiliary_upstream_origin": "unverified", "upstream_signature": "not_verified",
        "safetensors_structure": "not_tested", "tokenizer_runtime": "not_run",
        "pretrained_initialization": "not_run", "checkpoint_reload": "not_run",
        "serving": "not_run", "cuda": "not_run", "trained_context": "not_run",
        "representative_quality": "unmeasured", "jev_parity": "unknown",
        "native_cases_executed": 0, "network_calls": 0,
        "python_version": ".".join(map(str, sys.version_info[:3])),
    }
    try:
        if directory is None:
            raise SourceGateError("source_directory_required", blocked=True)
        supplied = Path(directory)
        if ".." in supplied.parts:
            raise SourceGateError("parent_traversal_not_supported")
        root = supplied.absolute()
        guards = _directories(root)
        before = {"source.json": _regular_file(root / "source.json", MANIFEST_MAX_BYTES)}
        digest, raw = _read(root / "source.json", before["source.json"], report["accounting"], retain=True)
        report["source_manifest_sha256"] = digest
        manifest = _strict_manifest(raw)
        # Admit every file before a potentially 988 MB read; never parse tensors.
        for name, maximum in AUXILIARY_LIMITS.items():
            before[name] = _regular_file(root / name, maximum)
        before["model.safetensors"] = _regular_file(root / "model.safetensors", WEIGHT_BYTES, WEIGHT_BYTES)
        for name in (*AUXILIARY_LIMITS, "model.safetensors"):
            _check_directories(guards)
            digest, _ = _read(root / name, before[name], report["accounting"])
            if digest != manifest["sha256"][name]:
                raise SourceGateError("file_digest_mismatch", name)
            report["files"].append({"name": name, "sha256": digest, "size_bytes": before[name].st_size,
                                    "basis": "upstream_anchor" if name == "model.safetensors" else "local_manifest_only"})
        _check_directories(guards)
        for name, value in before.items():
            if _stamp(_stat(root / name, name)) != _stamp(value):
                raise SourceGateError("input_changed_before_completion", name)
        report.update(status="verified", pinned_weight_bytes_verified=True,
                      all_local_file_hashes_verified=True)
    except SourceGateError as exc:
        report.update(status="blocked" if exc.blocked else "invalid", reason=exc.reason)
        if exc.filename is not None:
            report["file"] = exc.filename
    except (TypeError, ValueError, OSError):
        # Never print an input value, private pathname, OS error or manifest body.
        report.update(status="invalid", reason="invalid_source_location")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing regular-file snapshot from the pinned download")
    args = parser.parse_args(argv)
    report = verify_source(args.source)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return {"verified": 0, "invalid": 1, "blocked": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
