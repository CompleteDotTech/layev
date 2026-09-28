"""Score private frozen test transcripts offline; never contacts either service."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kev_laya.data import load_suite
from kev_laya.jev_comparison import score_frozen_test
from kev_laya.schema import strict_loads


def _transcripts(path: Path, expected: int) -> tuple[list[dict], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("transcript_must_be_regular_file")
    with path.open("rb") as stream:
        raw = stream.read(128 * 1024 * 1024 + 1)
    if len(raw) > 128 * 1024 * 1024:
        raise ValueError("transcript_byte_budget_exceeded")
    lines = [strict_loads(line) for line in raw.splitlines() if line.strip()]
    if len(lines) != expected:
        raise ValueError("paired_transcript_count_mismatch")
    return lines, hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--layev-transcripts", type=Path, required=True)
    parser.add_argument("--jev-transcripts", type=Path, required=True)
    parser.add_argument("--layev-model", required=True)
    parser.add_argument("--jev-model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    suite, manifest = load_suite(args.suite)
    rows = suite["test"]
    layev, layev_hash = _transcripts(args.layev_transcripts, len(rows))
    jev, jev_hash = _transcripts(args.jev_transcripts, len(rows))
    report = score_frozen_test(rows, layev, jev, layev_model=args.layev_model,
                               jev_model=args.jev_model)
    report["suite_manifest_sha256"] = hashlib.sha256(
        (args.suite / "manifest.json").read_bytes()).hexdigest()
    report["suite_id"] = manifest["id"]
    report["transcript_sha256"] = {"layev": layev_hash, "jev": jev_hash}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
