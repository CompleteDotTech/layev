"""Combine matched frozen evaluation reports across model seeds, offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kev_laya.evaluation import summarize_seed_variation
from kev_laya.schema import strict_loads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=REPORT.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    reports, hashes = {}, {}
    for value in args.run:
        seed_text, separator, file_text = value.partition("=")
        if not separator or not seed_text.isdecimal() or not file_text:
            raise ValueError("run_requires_nonnegative_seed_and_report_path")
        seed = int(seed_text)
        if seed in reports:
            raise ValueError("duplicate_model_seed")
        path = Path(file_text)
        if not path.is_file() or path.is_symlink():
            raise ValueError("report_must_be_regular_file")
        with path.open("rb") as stream:
            raw = stream.read(128 * 1024 * 1024 + 1)
        if len(raw) > 128 * 1024 * 1024:
            raise ValueError("report_byte_budget_exceeded")
        reports[seed] = strict_loads(raw)
        hashes[seed] = hashlib.sha256(raw).hexdigest()
    result = summarize_seed_variation(reports)
    result["input_sha256_by_seed"] = hashes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
