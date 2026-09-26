"""Report group-aware intervals from saved evaluations; no training or selection."""

import argparse
import json
from pathlib import Path

from kev_laya.evaluation import grouped_bootstrap
from kev_laya.io import sha256_file
from kev_laya.schema import strict_loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite evidence")
    actual = sha256_file(args.evaluation)
    if args.expected_sha256 and actual != args.expected_sha256:
        raise ValueError("evaluation checksum mismatch")
    source = strict_loads(args.evaluation.read_bytes())
    report = {"evaluation_sha256": actual, "split": source.get("split"),
              "evidence_class": source.get("evidence_class"),
              "uncertainty": grouped_bootstrap(source["rows"], samples=args.samples, seed=args.seed),
              "general_quality": "not established by this calculation", "jev_parity": "unknown"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
