"""Audit publishable source or a historical ZIP without modifying either.

The default checks INDEX bytes, not potentially different working-tree bytes.
--snapshot explicitly audits a source export without claiming Git verification.
This is a deterministic safety check, not proof that arbitrary secrets cannot
exist. A human review and the repository's normal protection remain required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess
import zipfile

MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 64 * 1024 * 1024

LICENSE_BLOBS = {
    "Kev-LICENSE": "e1ece45b775283628888a36bc75f56078cee9de0",
    "Laya-LICENSE": "d9a10c0d8e868ebf8da0b3dc95bb0be634c34bfe",
}
SKIP_DIRECTORIES = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    "node_modules", "build", "dist", "runs", ".mypy_cache",
}
FORBIDDEN_DIRECTORIES = SKIP_DIRECTORIES | {"backbone", "wandb", "htmlcov"}
FORBIDDEN_EXTENSIONS = {".pt", ".pth", ".safetensors", ".onnx", ".ckpt", ".pyc", ".zip"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA|ENCRYPTED)? ?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def portable_path(name: str) -> bool:
    path = PurePosixPath(name)
    windows = PureWindowsPath(name)
    return bool(name) and not (
        path.is_absolute() or windows.drive or "\\" in name or ":" in name
        or ".." in path.parts or path.as_posix() != name
        or any(p.endswith((".", " ")) for p in path.parts)
        or any(ord(char) < 32 or char in '<>"|?*' for char in name)
        or any(re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", p.split(".")[0], re.I)
               for p in path.parts)
    )


def index_files(root: Path) -> dict[str, bytes]:
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"],
        check=True, capture_output=True, timeout=15,
    ).stdout
    records = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0" or mode not in {b"100644", b"100755"}:
            raise ValueError("unmerged, symbolic-link, or submodule index entry")
        records.append((name.decode("utf-8"), oid))
    if not records:
        raise ValueError("empty Git index; stage the intended source first")
    sizes = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch-check"],
        input=b"".join(oid + b"\n" for _, oid in records),
        check=True, capture_output=True, timeout=15,
    ).stdout.splitlines()
    if len(sizes) != len(records):
        raise ValueError("unexpected Git object size response")
    total = 0
    for (name, oid), line in zip(records, sizes, strict=True):
        actual, kind, size = line.split()
        if actual != oid or kind != b"blob":
            raise ValueError("unexpected Git object size response")
        length = int(size)
        total += length
        if length > MAX_SOURCE_BYTES or total > MAX_TOTAL_SOURCE_BYTES:
            raise ValueError(f"source byte budget exceeded before reading contents: {name}")
    batch = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input=b"".join(oid + b"\n" for _, oid in records),
        check=True, capture_output=True, timeout=30,
    ).stdout
    files = {}
    cursor = 0
    for name, oid in records:
        end = batch.index(b"\n", cursor)
        actual_oid, kind, size = batch[cursor:end].split()
        if actual_oid != oid or kind != b"blob":
            raise ValueError("unexpected Git object response")
        start, length = end + 1, int(size)
        files[name] = batch[start:start + length]
        cursor = start + length + 1
    return files


def snapshot_files(root: Path) -> dict[str, bytes]:
    result = {}
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(p in SKIP_DIRECTORIES or p.endswith(".egg-info") for p in relative.parts):
            continue
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"symbolic link or junction in source: {relative.as_posix()}")
        if path.is_file():
            size = path.stat().st_size
            total += size
            if size > MAX_SOURCE_BYTES or total > MAX_TOTAL_SOURCE_BYTES:
                raise ValueError(f"source byte budget exceeded before reading contents: {relative.as_posix()}")
            result[relative.as_posix()] = path.read_bytes()
    return result


def audit(files: dict[str, bytes]) -> dict:
    errors = []
    folded, modules = {}, {}
    for name, raw in sorted(files.items()):
        if not portable_path(name):
            errors.append(f"nonportable source path: {name}")
        path = PurePosixPath(name)
        if path.name.startswith(".env") and path.name != ".env.example":
            errors.append(f"non-example dotenv file: {name}")
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            errors.append(f"generated/model/archive payload: {name}")
        if any(p in FORBIDDEN_DIRECTORIES or p.endswith(".egg-info") for p in path.parts):
            errors.append(f"generated directory tracked: {name}")
        if len(raw) > MAX_SOURCE_BYTES:
            errors.append(f"source file exceeds 2 MiB review threshold: {name}")
        key = name.casefold()
        if key in folded:
            errors.append(f"case-insensitive path collision: {folded[key]} / {name}")
        folded[key] = name
        if path.suffix in {".ts", ".tsx"} and not name.endswith(".d.ts"):
            module = path.with_suffix("").as_posix().casefold()
            if module in modules:
                errors.append(f"ambiguous TypeScript module stem: {modules[module]} / {name}")
            modules[module] = name
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"binary source requires explicit review: {name}")
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"credential-like content (value withheld): {name}")
        if re.search(r"[A-Za-z]:\\Users\\[^\\]+\\|/home/[A-Za-z0-9_.-]+/", text):
            errors.append(f"personal absolute path: {name}")
        if path.suffix == ".py":
            try:
                compile(text, name, "exec")
            except SyntaxError as exc:
                errors.append(f"Python syntax: {name}:{exc.lineno}")
            if any(line.rstrip(" \t") != line for line in text.splitlines()):
                errors.append(f"Python trailing whitespace: {name}")
        if path.suffix == ".md":
            # Validate relative file links; anchors and external targets are not fetched.
            prose = re.sub(r"```.*?```", "", text, flags=re.S)
            for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)", prose):
                target = target.split("#", 1)[0]
                if not target or ":" in target or target.startswith("//"):
                    continue
                parts = list(path.parent.parts)
                for part in PurePosixPath(target).parts:
                    if part == "..":
                        if parts:
                            parts.pop()
                        else:
                            parts.append("..")
                    elif part != ".":
                        parts.append(part)
                resolved = "/".join(parts).rstrip("/")
                if resolved not in files and not any(k.startswith(resolved + "/") for k in files):
                    errors.append(f"broken relative link: {name} -> {target}")
    for name, expected in LICENSE_BLOBS.items():
        raw = files.get("licenses/" + name)
        if raw is None or git_blob(raw.replace(b"\r\n", b"\n")) != expected:
            errors.append(f"upstream license hash mismatch or absent: {name}")
    if "LICENSE" not in files or "NOTICE" not in files:
        errors.append("root LICENSE and NOTICE must both be present")
    producer = files.get("src/kev_laya/telemetry_contract.py")
    consumer = files.get("integrations/overwatch/added/src/overwatch/kev_laya_contract.py")
    if producer is None or consumer is None or producer.replace(b"\r\n", b"\n") != consumer.replace(b"\r\n", b"\n"):
        errors.append("producer/consumer telemetry contract drift or absence")
    identity = {name: digest(raw.replace(b"\r\n", b"\n")) for name, raw in sorted(files.items())}
    fingerprint = digest(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    return {"passed": not errors, "file_count": len(files), "errors": errors,
            "lf_normalized_tree_sha256": fingerprint,
            "scope": "paths, indexed bytes, credential patterns, syntax, whitespace, relative links, licenses, contract copy",
            "limitations": "Not a complete secret detector, Ruff result, native validation, or approval to publish."}


def verify_archive(path: Path, expected_sha256: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("a trusted lowercase SHA-256 is required")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("archive checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        entries = [i for i in archive.infolist() if not i.is_dir()]
        names = [i.filename for i in entries]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member")
        for info in entries:
            if not portable_path(info.filename) or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("unsafe archive member")
        manifests = [n for n in names if n.endswith("/MANIFEST.json") and n.count("/") == 1]
        if len(manifests) != 1:
            raise ValueError("expected one top-level delivery manifest")
        manifest = json.loads(archive.read(manifests[0]))
        prefix = manifests[0].rsplit("/", 1)[0] + "/"
        payload = manifest["files"]
        if set(names) != {prefix + k for k in payload} | {manifests[0]}:
            raise ValueError("manifest coverage differs from archive members")
        if manifest["file_count"] != len(payload):
            raise ValueError("manifest count differs from payload")
        for name, info in payload.items():
            member = prefix + name
            if archive.getinfo(member).file_size != info["size_bytes"]:
                raise ValueError(f"payload size mismatch: {name}")
            with archive.open(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != info["sha256"]:
                    raise ValueError(f"payload checksum mismatch: {name}")
    return {"passed": True, "sha256": actual, "verified_payload_files": len(payload),
            "archive_modified": False, "extracted": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out and args.out.exists():
        raise FileExistsError("refusing to overwrite evidence")
    try:
        if args.archive:
            result = verify_archive(args.archive, args.archive_sha256 or "")
        else:
            files = snapshot_files(args.root) if args.snapshot else index_files(args.root)
            result = audit(files)
            result["input"] = "source snapshot, NOT a Git index" if args.snapshot else "Git index bytes"
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        result = {"passed": False, "error": str(exc), "error_type": type(exc).__name__}
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
