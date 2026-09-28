"""Offline paired scoring for frozen System One responses; no network calls.

The caller keeps raw requests and responses in approved private storage. This
module verifies a common request/answer contract and scores both arms with the
same rules. Billing and request authorization require separate live receipts.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re

from .data import Datum
from .evaluation import summarize
from .schema import SystemOneRequest, canonical


def request_sha256(request: SystemOneRequest) -> str:
    """Hash the semantic request shared by both models, excluding model name."""
    payload = {"state": request.state,
               "questions": {key: question.model_dump() for key, question in request.questions.items()}}
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def _probabilities(answer: dict, question) -> list[float]:
    if not isinstance(answer, dict) or answer.get("type") != question.type:
        raise ValueError("answer_type_mismatch")
    keys = [key for key, _ in question.options()]
    if question.type == "noul":
        value = answer.get("noul")
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid_noul_probability")
        return [1 - float(value), float(value)]
    values = answer.get("probabilities")
    if not isinstance(values, dict) or set(values) != set(keys):
        raise ValueError("probability_keys_mismatch")
    probabilities = [values[key] for key in keys]
    if any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
           for p in probabilities) or abs(sum(probabilities) - 1) > 1e-5:
        raise ValueError("invalid_probability_distribution")
    if question.type == "choice":
        chosen = answer.get("choice")
        if chosen not in values or values[chosen] < max(probabilities) - 1e-5:
            raise ValueError("choice_probability_mismatch")
    else:
        score = answer.get("score")
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= len(keys) - 1:
            raise ValueError("invalid_score")
    return [float(p) for p in probabilities]


def _score(datum: Datum, body: dict, expected_model: str) -> list[dict]:
    if not isinstance(body, dict) or body.get("model") != expected_model:
        raise ValueError("response_model_mismatch")
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(datum.request.questions):
        raise ValueError("answer_ids_mismatch")
    rows = []
    for (question_id, question), target in zip(datum.request.questions.items(), datum.targets, strict=True):
        probabilities = _probabilities(answers[question_id], question)
        if len(probabilities) != len(target):
            raise ValueError("target_shape_mismatch")
        chosen = max(range(len(probabilities)), key=probabilities.__getitem__)
        nll = -sum(y * math.log(max(p, 1e-12)) for p, y in zip(probabilities, target, strict=True))
        ordinal = (abs(sum(i * (p - y) for i, (p, y) in enumerate(zip(probabilities, target, strict=True))))
                   if question.type == "score" else None)
        rows.append({"record_id": datum.meta["id"], "group": datum.meta["group"],
                     "question_id": question_id, "type": question.type,
                     "domain": datum.meta["domain"], "language": datum.meta["language"],
                     "option_order": datum.meta.get("option_order", "unknown"),
                     "variations": datum.meta.get("variations", {}), "option_count": len(probabilities),
                     "target": target, "probabilities": probabilities,
                     "answer_probability": max(probabilities), "correct": target[chosen],
                     "nll": nll, "brier": sum((p - y) ** 2 for p, y in zip(probabilities, target, strict=True)),
                     "ordinal_mae": ordinal})
    return rows


def score_pair(datum: Datum, layev: dict, jev: dict, *, layev_model: str,
               jev_model: str) -> dict:
    """Verify one frozen labeled request and score both supplied response bodies."""
    if not re.fullmatch(r"jev-\d+\.\d+\.\d+", jev_model):
        raise ValueError("comparison_model_version_not_pinned")
    expected_hash = request_sha256(datum.request)
    for transcript, model in ((layev, layev_model), (jev, jev_model)):
        if not isinstance(transcript, dict) or set(transcript) != {"request", "response"}:
            raise ValueError("transcript_fields_invalid")
        request = SystemOneRequest.model_validate(transcript["request"])
        if request.model != model or request_sha256(request) != expected_hash:
            raise ValueError("frozen_request_mismatch")
    layev_rows = _score(datum, layev["response"], layev_model)
    jev_rows = _score(datum, jev["response"], jev_model)
    by = {}
    for field in ("type", "domain", "language", "option_order", "option_count"):
        by[field] = {}
        for arm, rows in (("layev", layev_rows), ("jev", jev_rows)):
            groups = defaultdict(list)
            for row in rows:
                groups[str(row[field])].append(row)
            by[field][arm] = {key: summarize(group) for key, group in sorted(groups.items())}
    return {"request_sha256": expected_hash, "question_count": len(layev_rows),
            "layev_response_sha256": hashlib.sha256(canonical(layev["response"]).encode()).hexdigest(),
            "jev_response_sha256": hashlib.sha256(canonical(jev["response"]).encode()).hexdigest(),
            "layev": {"summary": summarize(layev_rows), "rows": layev_rows},
            "jev": {"summary": summarize(jev_rows), "rows": jev_rows}, "by": by,
            "latency": "not_measured", "actual_billed_cost": "not_verified",
            "remote_authorization": "not_verified", "jev_parity": "unknown"}


def score_frozen_test(data: list[Datum], layev: list[dict], jev: list[dict], *,
                      layev_model: str, jev_model: str) -> dict:
    """Score equal-length frozen transcript sequences without dropping failures."""
    if not data or len(data) != len(layev) or len(data) != len(jev):
        raise ValueError("paired_transcript_count_mismatch")
    scored = [score_pair(datum, local, remote, layev_model=layev_model, jev_model=jev_model)
              for datum, local, remote in zip(data, layev, jev, strict=True)]
    rows = {arm: [row for pair in scored for row in pair[arm]["rows"]]
            for arm in ("layev", "jev")}
    by = {}
    for field in ("type", "domain", "language", "option_order", "option_count"):
        by[field] = {}
        for arm in ("layev", "jev"):
            groups = defaultdict(list)
            for row in rows[arm]:
                groups[str(row[field])].append(row)
            by[field][arm] = {key: summarize(group) for key, group in sorted(groups.items())}
    return {"schema_version": "layev-jev-offline-comparison/1",
            "scope": "offline paired scoring only; no remote call, latency or billed-cost verification",
            "sample_records": len(data), "sample_questions": len(rows["layev"]),
            "models": {"layev": layev_model, "jev": jev_model},
            "pairs": [{key: pair[key] for key in ("request_sha256", "question_count",
                                            "layev_response_sha256", "jev_response_sha256")}
                      for pair in scored],
            "summary": {arm: summarize(rows[arm]) for arm in rows}, "by": by,
            "latency": "not_measured", "actual_billed_cost": "not_verified",
            "remote_authorization": "not_verified", "jev_parity": "unknown"}
