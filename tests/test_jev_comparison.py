"""Offline contract checks do not stand in for an authorized Jev run."""
import copy
import json

import pytest

from kev_laya.data import freeze_smoke, load_suite, parse_datum
from kev_laya.jev_comparison import score_frozen_test, score_pair
from scripts.score_jev_comparison import main as score_cli


@pytest.fixture
def pair():
    datum = parse_datum({
        "state": {"case": "approved-example"},
        "questions": {
            "choice": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}},
            "score": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]},
            "noul": {"type": "noul", "instructions": "Yes?"}},
        "gold": {"choice": "a", "score": 1, "noul": True},
        "meta": {"id": "example-1", "group": "example", "domain": "fixture", "language": "en"}})
    answers = {"choice": {"type": "choice", "choice": "a", "confidence": 0.8,
                          "probabilities": {"a": 0.8, "b": 0.2}},
               "score": {"type": "score", "score": 0.7, "confidence": 0.6,
                         "legend": {"0": "low", "1": "high"},
                         "probabilities": {"0": 0.3, "1": 0.7}},
               "noul": {"type": "noul", "noul": 0.9}}

    def transcript(model):
        return {"request": {"state": datum.request.state,
                             "questions": {key: question.model_dump()
                                           for key, question in datum.request.questions.items()},
                             "model": model},
                "response": {"model": model, "answers": copy.deepcopy(answers)}}

    return datum, transcript("kev-laya-preview"), transcript("jev-1.13.0")


def test_paired_scoring_uses_identical_request_and_rules(pair):
    datum, layev, jev = pair
    result = score_pair(datum, layev, jev, layev_model="kev-laya-preview", jev_model="jev-1.13.0")
    assert result["question_count"] == 3
    assert result["layev"]["summary"] == result["jev"]["summary"]
    assert result["layev"]["summary"]["accuracy"] == 1
    assert result["by"]["type"]["jev"]["score"]["count"] == 1
    assert result["actual_billed_cost"] == "not_verified"


@pytest.mark.parametrize("mutation,reason", [
    (lambda t: t["request"].update(state={"case": "changed"}), "frozen_request_mismatch"),
    (lambda t: t["response"].update(model="jev-latest"), "response_model_mismatch"),
    (lambda t: t["response"]["answers"].pop("noul"), "answer_ids_mismatch"),
    (lambda t: t["response"]["answers"]["choice"]["probabilities"].update(a=0.9),
     "invalid_probability_distribution"),
    (lambda t: t["response"]["answers"]["score"].update(type="choice"), "answer_type_mismatch"),
])
def test_pair_rejects_mismatched_or_invalid_response(pair, mutation, reason):
    datum, layev, jev = pair
    mutation(jev)
    with pytest.raises(ValueError, match=reason):
        score_pair(datum, layev, jev, layev_model="kev-laya-preview", jev_model="jev-1.13.0")


def test_pair_refuses_moving_alias(pair):
    datum, layev, jev = pair
    with pytest.raises(ValueError, match="comparison_model_version_not_pinned"):
        score_pair(datum, layev, jev, layev_model="kev-laya-preview", jev_model="jev-latest")


def test_offline_aggregate_keeps_paired_counts_and_unknown_cost(pair):
    datum, layev, jev = pair
    report = score_frozen_test([datum], [layev], [jev],
                               layev_model="kev-laya-preview", jev_model="jev-1.13.0")
    assert report["sample_records"] == 1
    assert report["sample_questions"] == 3
    assert report["summary"]["layev"] == report["summary"]["jev"]
    assert report["pairs"][0]["request_sha256"]
    assert report["actual_billed_cost"] == "not_verified"
    with pytest.raises(ValueError, match="paired_transcript_count_mismatch"):
        score_frozen_test([datum], [layev], [],
                          layev_model="kev-laya-preview", jev_model="jev-1.13.0")


def test_offline_cli_reads_frozen_synthetic_transcripts(tmp_path):
    suite_path = tmp_path / "suite"
    freeze_smoke(suite_path)
    suite, _ = load_suite(suite_path)

    def transcript(datum, model):
        answers = {}
        for (key, question), target in zip(datum.request.questions.items(), datum.targets, strict=True):
            options = [name for name, _ in question.options()]
            chosen = target.index(1.0)
            if question.type == "noul":
                answers[key] = {"type": "noul", "noul": 0.9 if chosen else 0.1}
            else:
                probabilities = {name: 0.9 if i == chosen else 0.1 / (len(options) - 1)
                                 for i, name in enumerate(options)}
                answers[key] = {"type": question.type, "probabilities": probabilities}
                if question.type == "choice":
                    answers[key]["choice"] = options[chosen]
                else:
                    answers[key]["score"] = sum(i * p for i, p in enumerate(probabilities.values()))
        return {"request": {"state": datum.request.state,
                            "questions": {key: q.model_dump() for key, q in datum.request.questions.items()},
                            "model": model}, "response": {"model": model, "answers": answers}}

    files = {}
    for arm, model in (("layev", "kev-laya-preview"), ("jev", "jev-1.13.0")):
        path = tmp_path / f"{arm}.jsonl"
        path.write_text("".join(json.dumps(transcript(datum, model)) + "\n"
                                for datum in suite["test"]), encoding="utf-8")
        files[arm] = path
    output = tmp_path / "report.json"
    assert score_cli(["--suite", str(suite_path), "--layev-transcripts", str(files["layev"]),
                      "--jev-transcripts", str(files["jev"]), "--layev-model", "kev-laya-preview",
                      "--jev-model", "jev-1.13.0", "--out", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["sample_records"] == len(suite["test"])
    assert report["summary"]["layev"] == report["summary"]["jev"]
    assert report["actual_billed_cost"] == "not_verified"
    with pytest.raises(FileExistsError):
        score_cli(["--suite", str(suite_path), "--layev-transcripts", str(files["layev"]),
                   "--jev-transcripts", str(files["jev"]), "--layev-model", "kev-laya-preview",
                   "--jev-model", "jev-1.13.0", "--out", str(output)])
