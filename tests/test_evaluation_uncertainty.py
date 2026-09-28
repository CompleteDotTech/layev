"""Uncertainty reports do not guess groups or certify independent natural data."""

import copy
import random

import pytest
import torch

from kev_laya.encoding import ByteTokenizer, Limits
from kev_laya.evaluation import calibrate, evaluate, grouped_bootstrap, summarize


def rows():
    return [
        {"group": group, "correct": correct, "answer_probability": 0.8,
         "nll": 0.3, "brier": 0.1, "ordinal_mae": None}
        for group, correct in (("a", 0.0), ("b", 1.0), ("c", 1.0))
        for _ in range(3)
    ]


def test_group_bootstrap_is_reproducible_and_preserves_rng():
    state = random.getstate()
    result = grouped_bootstrap(rows(), samples=100, seed=123)
    assert result == grouped_bootstrap(rows(), samples=100, seed=123)
    assert random.getstate() == state
    assert result["groups"] == 3 and result["status"] == "measured"
    assert result["intervals"]["ordinal_mae"] is None
    interval = result["intervals"]["accuracy"]
    assert 0 <= interval["lower"] <= 2 / 3 <= interval["upper"] <= 1


def test_duplicate_questions_do_not_create_independent_groups():
    original = grouped_bootstrap(rows(), samples=100, seed=9)
    repeated = grouped_bootstrap(rows() * 5, samples=100, seed=9)
    assert repeated["groups"] == 3
    for key in ("accuracy", "nll", "brier", "ece"):
        assert repeated["intervals"][key]["lower"] == pytest.approx(original["intervals"][key]["lower"])
        assert repeated["intervals"][key]["upper"] == pytest.approx(original["intervals"][key]["upper"])


@pytest.mark.parametrize("data", [[], [{"record_id": "not-a-group"}], rows()[:3]])
def test_missing_or_insufficient_groups_remain_unknown(data):
    result = grouped_bootstrap(data)
    assert result["status"] == "unknown"
    assert all(value is None for value in result["intervals"].values())


@pytest.mark.parametrize("bins", [0, -1, True, 0.5])
def test_invalid_bin_count_rejected(bins):
    with pytest.raises(ValueError):
        summarize(rows(), bins=bins)


def test_evaluation_retains_group_identity(tiny, suite):
    data, _ = suite
    report = evaluate(tiny, data["development"][:2], ByteTokenizer(), Limits(512, 8192), split="development")
    assert {row["group"] for row in report["rows"]} == {d.meta["group"] for d in data["development"][:2]}


def test_declared_variation_slices_keep_missing_values_unknown(tiny, suite):
    from kev_laya.data import Datum

    data, _ = suite
    first, second = data["development"][:2]
    first_meta = {**first.meta, "option_order": "reversed",
                  "variations": {"colors": "red", "wording": "short"}}
    selected = [Datum(first.request, first.targets, first_meta), second]
    report = evaluate(tiny, selected, ByteTokenizer(), Limits(512, 8192), split="development")
    assert report["by"]["option_order"]["reversed"]["count"] == len(first.targets)
    assert report["by"]["option_order"]["unknown"]["count"] == len(second.targets)
    assert report["variation_by"]["colors"]["red"]["count"] == len(first.targets)
    assert report["variation_by"]["colors"]["unknown"]["count"] == len(second.targets)
    assert report["variation_by"]["wording"]["short"]["count"] == len(first.targets)
    assert report["variation_by"]["levels"]["unknown"]["count"] == len(report["rows"])


def test_calibration_without_type_does_not_retain_unreported_old_temperature(tiny, suite):
    data, manifest = suite
    datum = copy.deepcopy(data["calibration"][0])
    key = next(k for k, q in datum.request.questions.items() if q.type == "choice")
    index = list(datum.request.questions).index(key)
    from kev_laya.data import Datum
    choice_only = Datum(datum.request.model_copy(update={"questions": {key: datum.request.questions[key]}}),
                        [datum.targets[index]], datum.meta)
    tiny.temperatures.update(noul=2.0, score=3.0)
    before = copy.deepcopy(tiny.state_dict())
    result = calibrate(tiny, [choice_only], ByteTokenizer(), Limits(512, 8192),
                       split="calibration", split_sha256=manifest["partitions"]["calibration"]["sha256"], iterations=2)
    assert tiny.temperatures["noul"] == result["fits"]["noul"]["temperature"] == 1.0
    assert tiny.temperatures["score"] == result["fits"]["score"]["temperature"] == 1.0
    for name, tensor in tiny.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
