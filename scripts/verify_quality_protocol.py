"""Read-only, local quality-protocol preflight; does not train or grant acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--data-review", type=Path)
    parser.add_argument("--suite", type=Path)
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from kev_laya.quality_protocol import ProtocolError, PrerequisiteMissing, preflight
    receipt = {"schema_version": "layev-quality-protocol/1", "status": "blocked",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "gate": "quality-protocol-preflight", "quality": "not_measured", "jev_parity": "unknown",
        "native_model": "not_tested", "cuda": "not_tested", "trained_context": "not_tested",
        "regression_outcome": "not_tested", "source_writes": 0, "network_calls": 0,
        "training_calls": 0, "paid_calls": 0,
        "gate_source_sha256": hashlib.sha256((root / "src/kev_laya/quality_protocol.py").read_bytes()).hexdigest()}
    try:
        if None in (args.protocol, args.data_review, args.suite):
            raise PrerequisiteMissing("protocol_data_review_and_suite_required")
        receipt.update(preflight(args.protocol, args.data_review, args.suite))
        code = 0
    except PrerequisiteMissing as exc:
        receipt["reason"] = str(exc)
        code = 2
    except ProtocolError as exc:
        receipt.update(status="rejected", reason=str(exc))
        code = 1
    except Exception as exc:
        receipt.update(status="failed", reason="preflight_execution_failed", error_type=type(exc).__name__)
        code = 1
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
