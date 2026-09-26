"""Held-out evaluation and calibration; no test-set fitting path."""
from __future__ import annotations
from collections import defaultdict
import math
import torch
from .data import Datum
from .encoding import Tokenizer, Limits
from .model import DecisionEngine


def summarize(rows: list[dict], bins: int = 10) -> dict:
    if not rows:
        return {"count": 0, "nll": None, "brier": None, "ece": None, "accuracy": None,
                "ordinal_mae": None, "reliability": [], "risk_coverage": []}
    reliability = []
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        chosen = [r for r in rows if lo <= r["answer_probability"] and (r["answer_probability"] < hi or i == bins - 1)]
        if chosen:
            confidence = sum(r["answer_probability"] for r in chosen) / len(chosen)
            accuracy = sum(r["correct"] for r in chosen) / len(chosen)
            ece += len(chosen) / len(rows) * abs(confidence - accuracy)
            reliability.append({"lower": lo, "upper": hi, "count": len(chosen), "confidence": confidence, "accuracy": accuracy})
        else:
            reliability.append({"lower": lo, "upper": hi, "count": 0, "confidence": None, "accuracy": None})
    ordered = sorted(rows, key=lambda r: r["answer_probability"], reverse=True)
    curve, mistakes = [], 0.0
    for index, row in enumerate(ordered, start=1):
        mistakes += 1 - row["correct"]
        # Only evaluate boundaries between tied confidences, avoiding arbitrary within-tie selection.
        if index == len(ordered) or row["answer_probability"] != ordered[index]["answer_probability"]:
            curve.append({"coverage": index / len(rows), "risk": mistakes / index, "threshold": row["answer_probability"]})
    ordinal = [r["ordinal_mae"] for r in rows if r["ordinal_mae"] is not None]
    return {"count": len(rows), "nll": sum(r["nll"] for r in rows) / len(rows),
            "brier": sum(r["brier"] for r in rows) / len(rows), "ece": ece,
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "ordinal_mae": sum(ordinal) / len(ordinal) if ordinal else None,
            "reliability": reliability, "risk_coverage": curve}


@torch.no_grad()
def evaluate(model: DecisionEngine, data: list[Datum], tokenizer: Tokenizer, limits: Limits, *, split: str) -> dict:
    model.eval()
    rows = []
    for datum in data:
        enc = datum.encode(tokenizer, limits)
        logits, _ = model(enc)
        for branch, z, target in zip(enc.branches, logits, datum.targets, strict=True):
            kind = branch.question.type
            lp = (z.double() / model.temperatures[kind]).log_softmax(-1)
            p = lp.exp()
            y = torch.tensor(target, device=p.device, dtype=torch.double)
            chosen = int(p.argmax())
            levels = torch.arange(len(p), device=p.device, dtype=torch.double)
            context_length = len(enc.state) + len(branch.ids)
            rows.append({"record_id": datum.meta["id"], "question_id": branch.question_id,
                         "type": kind, "domain": datum.meta["domain"], "language": datum.meta["language"],
                         "option_count": len(p), "context_length": context_length,
                         "length_bucket": next((n for n in (512, 2048, 8192, 16384, 32768) if context_length <= n), "over32k"),
                         "probabilities": p.tolist(), "target": target, "answer_probability": float(p.max()),
                         "correct": float(y[chosen]), "nll": float(-(lp * y).sum()), "brier": float((p - y).square().sum()),
                         "ordinal_mae": float(((p * levels).sum() - (y * levels).sum()).abs()) if kind == "score" else None})
    breakdowns = {}
    for feature in ("type", "domain", "language", "option_count", "length_bucket"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row[feature])].append(row)
        breakdowns[feature] = {name: summarize(values) for name, values in sorted(grouped.items())}
    return {"split": split, "evidence_class": "pretrained-backbone" if model.native_weights_loaded else "tiny-synthetic-fixture",
            "confidence_evaluated": "max probability, not public entropy concentration", "summary": summarize(rows),
            "by": breakdowns, "rows": rows, "temperatures": dict(model.temperatures)}


@torch.no_grad()
def collect_calibration_logits(model, data, tokenizer, limits):
    model.eval()
    rows = []
    for datum in data:
        enc = datum.encode(tokenizer, limits)
        logits, _ = model(enc)
        rows.extend((branch.question.type, z.detach().cpu().double(), torch.tensor(y, dtype=torch.double))
                    for branch, z, y in zip(enc.branches, logits, datum.targets, strict=True))
    return rows


def calibrate(model: DecisionEngine, data: list[Datum], tokenizer: Tokenizer, limits: Limits,
              *, split: str, split_sha256: str, iterations: int = 60) -> dict:
    if split != "calibration":
        raise ValueError("calibration may only fit the frozen calibration partition")
    if not data:
        raise ValueError("empty calibration partition")
    rows = collect_calibration_logits(model, data, tokenizer, limits)
    fits = {}
    for kind in ("choice", "score", "noul"):
        selected = [(z, y) for t, z, y in rows if t == kind]
        if not selected:
            fits[kind] = {"temperature": 1.0, "status": "unfitted-no-samples", "count": 0}
            continue
        log_t = torch.zeros((), dtype=torch.double, requires_grad=True)
        opt = torch.optim.Adam([log_t], lr=0.05)
        def value():
            return torch.stack([-(y * (z / log_t.exp()).log_softmax(-1)).sum() for z, y in selected]).mean()
        initial = float(value().detach())
        best_loss, best_t = initial, 1.0
        for _ in range(iterations):
            opt.zero_grad()
            loss = value()
            loss.backward()
            opt.step()
            with torch.no_grad():
                log_t.clamp_(math.log(0.2), math.log(5.0))
                nll = float(value())
                if nll < best_loss:
                    best_loss, best_t = nll, float(log_t.exp())
        model.temperatures[kind] = best_t
        fits[kind] = {"temperature": best_t, "count": len(selected), "status": "fitted-on-calibration",
                      "before_nll": initial, "after_nll": best_loss}
    model.calibration_provenance = {"status": "fitted-held-out", "partition": split, "sha256": split_sha256,
                                    "fits": fits, "method": "per-type-temperature-nll-v1", "bounds": [0.2, 5.0]}
    return model.calibration_provenance
