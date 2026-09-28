"""Check the preserved research matrix and its conservative Layev mapping."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "docs/research/FEATURE_MATRIX.original.json"
MAPPING = ROOT / "docs/research/layev-feature-mapping.json"
EXPECTED_SHA256 = "bc002dbc82b69eb456c1f3d7e11c83d0da5cfe83c699219faa4b88aa54ded405"


def verify() -> None:
    original_bytes = ORIGINAL.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == EXPECTED_SHA256
    original = json.loads(original_bytes)
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    assert mapping["original_matrix_sha256"] == EXPECTED_SHA256
    assert mapping["original_matrix_path"] == "docs/research/FEATURE_MATRIX.original.json"
    assert len(original["features"]) == len(mapping["features"]) == 45
    ids = [feature["id"] for feature in original["features"]]
    assert len(set(ids)) == 45
    assert len(original["projects"]) == 10
    assert all(set(project["cells"]) == set(ids) for project in original["projects"])
    for source, row in zip(original["features"], mapping["features"], strict=True):
        for key in ("id", "title", "definition", "group"):
            assert source[key] == row[key], (source["id"], key)
        assert source["source"] == row["type_safe_primary_source"]
        assert row["classification"] in {"comparison_feature", "Jev_absent_non_gap", "service_policy"}
        assert row["implemented"] in {"source_present", "unknown"}
        assert row["tested"] in {"test_present_not_rerun", "unknown"}
        assert all(row[key] == "unknown" for key in ("native_validated", "quality_measured", "Jev_compared"))
        for key in ("Layev_source_paths", "Layev_test_paths"):
            for relative in row[key]:
                path = Path(relative)
                assert not path.is_absolute() and ".." not in path.parts and (ROOT / path).is_file(), (source["id"], relative)
        assert row["implemented"] == ("source_present" if row["Layev_source_paths"] else "unknown")
        assert row["tested"] == ("test_present_not_rerun" if row["Layev_test_paths"] else "unknown")
        assert all(issue.startswith("https://github.com/CompleteDotTech/layev/issues/") for issue in row["issues"])
    ledger = json.loads((ROOT / "docs/acceptance-status.json").read_text(encoding="utf-8"))
    matrix_status = ledger["original_research_matrix"]
    assert matrix_status["sha256"] == EXPECTED_SHA256
    assert matrix_status["columns"] == ids
    print("Verified preserved matrix SHA-256, 45 mapped features, 450 original cells, and evidence limits")


if __name__ == "__main__":
    verify()
