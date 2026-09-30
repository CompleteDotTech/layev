"""Check the preserved research matrix and its conservative Layev mapping."""
from __future__ import annotations

import hashlib
import json
import re
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
    native_receipt_path = "docs/evidence/native/long-bf16-gradient-10k-20260929.json"
    native_receipt = json.loads((ROOT / native_receipt_path).read_text(encoding="utf-8"))
    assert native_receipt["passed"] is True and native_receipt["precision"] == "bf16"
    assert native_receipt["state_tokens"] == 10520
    for comparison in native_receipt["comparisons"].values():
        assert comparison["passed"] is True
        assert comparison["gradient_count"] == 100
        assert comparison["failed_gradient_count"] == 0
    fp32_receipt_path = "docs/evidence/native/short-fp32-canonical-v10-20260929.json"
    fp32 = json.loads((ROOT / fp32_receipt_path).read_text(encoding="utf-8"))
    assert fp32["precision"] == "fp32"
    assert fp32["derivative_version"] == "cuda-fp32-canonical-v10"
    assert fp32["attention_operator_version"] == "cuda-fp32-absolute-query-bands-fixed256-cpurope-v3"
    assert fp32["cache_plan_version"] == "cuda-double-kv-gqa-storage-v2"
    assert fp32["transformers_version"] == "4.57.1"
    assert fp32["cuda_allocator_fraction"] == 0.7 and fp32["tf32"] is False
    assert fp32["tolerances"] == {"hidden_atol_and_rtol": 1e-4,
                                 "logits_and_probabilities_atol_and_rtol": 1e-5,
                                 "gradients_atol_and_rtol": 2e-5}
    assert fp32["checkpoint_sha256"] == "99f797ffd0521adbee964d130d73eb381c9b6bd3b7896ce61b5007886d19ccf9"
    hashes = [fp32["checkpoint_sha256"], *fp32["source_file_sha256"].values()]
    assert set(fp32["source_file_sha256"]) == {"src/kev_laya/model.py",
        "src/kev_laya/native_gradients.py", "src/kev_laya/training.py", "src/kev_laya/execution.py"}
    assert re.fullmatch(r"[0-9a-f]{40}", fp32["source_parent"])
    hf = fp32["hf_oracle"]
    hashes.extend([hf["private_receipt_sha256"], hf["protocol_script_sha256"]])
    assert hf["gradient_count"] == 96 and hf["all_gradients_passed"] is True
    assert hf["tokens"] == 62 and hf["hidden"]["passed"] is True
    assert 0 <= hf["hidden"]["max_tolerance_ratio"] <= 1
    assert hf["hidden"]["max_absolute"] == 0
    assert 0 <= hf["worst_gradient_tolerance_ratio"] <= 1
    cases = fp32["seam_cases"]
    assert len(cases) == 4
    assert {case["case"] for case in cases} == {"short_full", "short_medium_255",
        "short_medium_256", "medium_257"}
    for case in cases:
        hashes.extend([case["private_receipt_sha256"], case["protocol_script_sha256"]])
        assert len(case["branch_tokens"]) == 2
        assert 1 <= case["state_tokens"] <= 257
        assert all(case["state_tokens"] < length < 8192 for length in case["branch_tokens"])
        if case["case"] == "short_full":
            assert max(case["branch_tokens"]) <= 256
        else:
            assert case["state_tokens"] == {"short_medium_255": 255,
                "short_medium_256": 256, "medium_257": 257}[case["case"]]
        if case["case"].startswith("short_medium"):
            assert min(case["branch_tokens"]) > 256
        assert set(case["comparisons"]) == {"batched_serial", "serial_full", "batched_full"}
        for comparison in case["comparisons"].values():
            assert comparison["passed"] is True
            assert comparison["gradient_count"] == 100
            assert comparison["failed_gradient_count"] == 0 and comparison["failed_gradients"] == {}
            for metric in ("logits", "probabilities", "worst_gradient"):
                assert comparison[metric]["passed"] is True
                assert 0 <= comparison[metric]["max_tolerance_ratio"] <= 1
    extended = fp32["boundary_and_long_cases"]
    assert len(extended) == 3
    assert {case["case"] for case in extended} == {"boundary_8191", "boundary_8192", "long_10520"}
    for case in extended:
        assert case["checkpoint_sha256"] == fp32["checkpoint_sha256"]
        hashes.extend([case["private_receipt_sha256"], case["protocol_script_sha256"], case["source"]["diff_sha256"]])
        assert case["source"]["commit"] == fp32["source_parent"]
        assert case["state_tokens"] == {"boundary_8191": 8191, "boundary_8192": 8192, "long_10520": 10520}[case["case"]]
        assert len(case["branch_tokens"]) == 2
        assert all(case["state_tokens"] < length <= 32768 for length in case["branch_tokens"])
        assert set(case["comparisons"]) == {"batched_serial", "serial_full", "batched_full"}
        for comparison in case["comparisons"].values():
            assert comparison["passed"] is True and comparison["gradient_count"] == 100
            assert comparison["failed_gradient_count"] == 0 and comparison["failed_gradients"] == {}
            for metric in ("logits", "probabilities", "worst_gradient"):
                assert comparison[metric]["passed"] is True
                assert 0 <= comparison[metric]["max_tolerance_ratio"] <= 1
            assert comparison["logits"]["max_absolute"] == comparison["probabilities"]["max_absolute"] == 0
        assert set(case["runs"]) == {"batched", "serial", "full"}
        for run in case["runs"].values():
            assert run["elapsed_seconds"] > 0 and run["peak_allocated_bytes"] > 0
    failure = fp32["exact_limit_resource_failure"]
    assert failure["passed_resource_probe"] is False
    assert failure["stage"] == "forward" and failure["error_type"] == "OutOfMemoryError"
    assert failure["max_branch_tokens"] == 32768 and failure["aggregate_tokens"] == 65536
    assert failure["precision"] == "fp32" and failure["cuda_allocator_fraction"] == 0.7
    assert failure["checkpoint_sha256"] == fp32["checkpoint_sha256"]
    hashes.extend([failure["private_receipt_sha256"], failure["protocol_script_sha256"]])
    assert failure["peak_reserved_bytes"] >= failure["peak_allocated_bytes"] > 0
    assert failure["worker_elapsed_seconds"] > 0
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes)
    for source, row in zip(original["features"], mapping["features"], strict=True):
        for key in ("id", "title", "definition", "group"):
            assert source[key] == row[key], (source["id"], key)
        assert source["source"] == row["type_safe_primary_source"]
        assert row["classification"] in {"comparison_feature", "Jev_absent_non_gap", "service_policy"}
        assert row["implemented"] in {"source_present", "unknown"}
        if row["id"] in {"parallel", "shared"}:
            assert row["tested"].startswith("CPU suite 858 passed on 2026-09-29;")
            assert row["native_validated"].startswith("partial;")
            assert row["evidence_paths"] == [native_receipt_path, fp32_receipt_path]
            assert ("not complete issue 3" in row["limitations"]
                    or "full native" in row["limitations"])
        else:
            assert row["tested"] in {"test_present_not_rerun", "unknown"}
            assert row["native_validated"] == "unknown"
            assert row.get("evidence_paths", []) == []
        assert all(row[key] == "unknown" for key in ("quality_measured", "Jev_compared"))
        for key in ("Layev_source_paths", "Layev_test_paths"):
            for relative in row[key]:
                path = Path(relative)
                assert not path.is_absolute() and ".." not in path.parts and (ROOT / path).is_file(), (source["id"], relative)
        assert row["implemented"] == ("source_present" if row["Layev_source_paths"] else "unknown")
        if row["id"] not in {"parallel", "shared"}:
            assert row["tested"] == ("test_present_not_rerun" if row["Layev_test_paths"] else "unknown")
        assert all(issue.startswith("https://github.com/CompleteDotTech/layev/issues/") for issue in row["issues"])
    ledger = json.loads((ROOT / "docs/acceptance-status.json").read_text(encoding="utf-8"))
    matrix_status = ledger["original_research_matrix"]
    assert matrix_status["sha256"] == EXPECTED_SHA256
    assert matrix_status["columns"] == ids
    print("Verified preserved matrix SHA-256, 45 mapped features, 450 original cells, and evidence limits")


if __name__ == "__main__":
    verify()
