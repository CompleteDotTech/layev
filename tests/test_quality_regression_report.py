"""Software regression receipt tests; no representative-quality evidence."""
from __future__ import annotations

import json

import pytest

from kev_laya.checkpoint import save_checkpoint
from kev_laya.counterfactual import regression_data
from kev_laya.encoding import ByteTokenizer
from kev_laya.io import sha256_file
from scripts.report_quality_regression import CASE, run


@pytest.fixture
def pinned_checkpoint(tmp_path, tiny):
    path = tmp_path / "trained.pt"
    tiny.training_steps = 1
    save_checkpoint(path, tiny, ByteTokenizer(), provenance={"seed": 47, "secret": "never-export-this"})
    return path, sha256_file(path)


def test_preserved_case_has_two_orders_and_three_types_each():
    selected = [datum for datum in regression_data() if datum.request.state == CASE]
    assert len(selected) == 2
    assert {datum.meta["option_order"] for datum in selected} == {"red-first", "blue-first"}
    assert all({question.type for question in datum.request.questions.values()} ==
               {"choice", "noul", "score"} for datum in selected)


def test_pinned_checkpoint_report_retains_all_six_and_provenance(pinned_checkpoint, tmp_path):
    checkpoint, digest = pinned_checkpoint
    output = tmp_path / "regression.json"
    result = run(checkpoint, output, expected_sha256=digest)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == result
    assert result["status"] == "observed_regression_not_heldout"
    assert result["evaluation"]["quality_gate"] == result["status"]
    assert result["checkpoint_sha256"] == digest
    assert result["checkpoint_declarations"]["seed"] == 47
    assert "never-export-this" not in output.read_text(encoding="utf-8")
    assert result["training_exposure_status"] == "not_verified_by_this_report"
    assert result["checkpoint_training_steps_declared"] == 1
    assert len(result["regression_source_sha256"]) == 64
    assert result["tokenizer_identity"] == "utf8-byte-fixture-v1"
    assert result["total"] == 6
    assert result["correct"] == sum(row["correct"] for row in result["evaluation"]["rows"])
    assert ({kind: item["total"] for kind, item in result["by_type"].items()}
            == {"choice": 2, "noul": 2, "score": 2})
    assert ({order: item["total"] for order, item in result["by_option_order"].items()}
            == {"red-first": 3, "blue-first": 3})
    with pytest.raises(FileExistsError):
        run(checkpoint, output, expected_sha256=digest)
    assert json.loads(output.read_text(encoding="utf-8")) == result


@pytest.mark.parametrize("digest", ["0" * 64, "not-a-sha256"])
def test_invalid_checkpoint_pin_fails_before_output(pinned_checkpoint, tmp_path, digest):
    checkpoint, _ = pinned_checkpoint
    output = tmp_path / "should-not-exist.json"
    with pytest.raises(ValueError):
        run(checkpoint, output, expected_sha256=digest)
    assert not output.exists()


def test_untrained_checkpoint_refused(tmp_path, tiny):
    checkpoint = tmp_path / "init.pt"
    save_checkpoint(checkpoint, tiny, ByteTokenizer())
    output = tmp_path / "should-not-exist.json"
    with pytest.raises(ValueError, match="declares_no_training_steps"):
        run(checkpoint, output, expected_sha256=sha256_file(checkpoint))
    assert not output.exists()


def test_incomplete_or_bad_regression_result_refused(pinned_checkpoint, tmp_path, monkeypatch):
    from scripts import report_quality_regression as module

    checkpoint, digest = pinned_checkpoint
    original = module.evaluate

    def incomplete(*args, **kwargs):
        report = original(*args, **kwargs)
        report["rows"].pop()
        return report

    monkeypatch.setattr(module, "evaluate", incomplete)
    output = tmp_path / "should-not-exist.json"
    with pytest.raises(ValueError, match="readout_incomplete"):
        run(checkpoint, output, expected_sha256=digest)
    assert not output.exists()


def test_all_six_rule_retains_type_and_order_failures(pinned_checkpoint, tmp_path, monkeypatch):
    from scripts import report_quality_regression as module

    checkpoint, digest = pinned_checkpoint
    original = module.evaluate
    wrong = [False]

    def controlled(*args, **kwargs):
        report = original(*args, **kwargs)
        for row in report["rows"]:
            row["correct"] = 1.0
        if wrong[0]:
            row = next(row for row in report["rows"] if row["type"] == "score"
                       and row["option_order"] == "blue-first")
            row["correct"] = 0.0
        return report

    monkeypatch.setattr(module, "evaluate", controlled)
    passed = run(checkpoint, tmp_path / "pass.json", expected_sha256=digest)
    assert passed["all_six_correct"] is True and passed["correct"] == 6
    wrong[0] = True
    failed = run(checkpoint, tmp_path / "fail.json", expected_sha256=digest)
    assert failed["all_six_correct"] is False and failed["correct"] == 5
    assert failed["by_type"]["score"] == {"correct": 1, "total": 2}
    assert failed["by_option_order"]["blue-first"] == {"correct": 2, "total": 3}
    assert len(failed["evaluation"]["rows"]) == 6


def test_existing_output_is_not_replaced_even_for_wrong_checkpoint_pin(
        pinned_checkpoint, tmp_path):
    checkpoint, _ = pinned_checkpoint
    output = tmp_path / "existing.json"
    output.write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(checkpoint, output, expected_sha256="0" * 64)
    assert output.read_text(encoding="utf-8") == "preserved"


def test_receipt_filters_untrusted_checkpoint_metadata(tmp_path, tiny):
    sentinel = "never-export-this"
    checkpoint = tmp_path / "trained.pt"
    tiny.training_steps = 1
    tiny.loaded_backbone_sha256 = sentinel
    save_checkpoint(checkpoint, tiny, ByteTokenizer(), parent_sha256=sentinel,
                    provenance={"seed": 47, "secret": sentinel, "commit": "a" * 40,
                                "data_sha256": "b" * 64})
    output = tmp_path / "regression.json"
    result = run(checkpoint, output, expected_sha256=sha256_file(checkpoint))
    assert sentinel not in output.read_text(encoding="utf-8")
    assert result["parent_sha256"] is None
    assert result["loaded_backbone_sha256"] is None
    assert result["checkpoint_declarations"] == {
        "seed": 47, "commit": "a" * 40, "data_sha256": "b" * 64}


def test_invalid_native_flag_cannot_make_a_pretrained_receipt(tmp_path, tiny):
    checkpoint = tmp_path / "trained.pt"
    tiny.training_steps = 1
    tiny.native_weights_loaded = "never-export-this"
    save_checkpoint(checkpoint, tiny, ByteTokenizer())
    output = tmp_path / "regression.json"
    with pytest.raises(ValueError, match="checkpoint_native_flag_invalid"):
        run(checkpoint, output, expected_sha256=sha256_file(checkpoint))
    assert not output.exists()
