"""Held-out evaluation and calibration; no test-set fitting path."""
from __future__ import annotations
from collections import defaultdict
import math
import random
import re
import torch
from .data import Datum
from .encoding import Tokenizer, Limits
from .model import DecisionEngine


def summarize(rows: list[dict], bins: int = 10) -> dict:
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
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
def evaluate(model: DecisionEngine, data: list[Datum], tokenizer: Tokenizer, limits: Limits, *,
             split: str, test_claim=None, test_readout_kind: str | None = None,
             checkpoint_sha256: str | None = None, diagnostic: bool = False) -> dict:
    if split == "test":
        from .quality_readout import QualityReadoutLedger
        if diagnostic and any(value is not None for value in (test_claim, test_readout_kind, checkpoint_sha256)):
            raise ValueError("diagnostic test cannot use a quality claim")
        if not diagnostic:
            if not isinstance(test_claim, QualityReadoutLedger):
                raise ValueError("untouched test requires a durable readout claim")
            test_claim.consume_test_readout(test_readout_kind, checkpoint_sha256)
    elif diagnostic or any(value is not None for value in (test_claim, test_readout_kind, checkpoint_sha256)):
        raise ValueError("test access arguments require the test split")
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
            rows.append({"record_id": datum.meta["id"], "group": datum.meta["group"], "question_id": branch.question_id,
                         "type": kind, "domain": datum.meta["domain"], "language": datum.meta["language"],
                         "option_order": datum.meta.get("option_order", "unknown"),
                         "variations": datum.meta.get("variations", {}),
                         "option_count": len(p), "context_length": context_length,
                         "length_bucket": next((n for n in (512, 2048, 8192, 16384, 32768) if context_length <= n), "over32k"),
                         "probabilities": p.tolist(), "target": target, "answer_probability": float(p.max()),
                         "correct": float(y[chosen]), "nll": float(-(lp * y).sum()), "brier": float((p - y).square().sum()),
                         "ordinal_mae": float(((p * levels).sum() - (y * levels).sum()).abs()) if kind == "score" else None})
    breakdowns = {}
    for feature in ("type", "domain", "language", "option_order", "option_count", "length_bucket"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row[feature])].append(row)
        breakdowns[feature] = {name: summarize(values) for name, values in sorted(grouped.items())}
    variation_breakdowns = {}
    for feature in ("colors", "levels", "wording", "question_ids", "domains", "languages", "context_lengths"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["variations"].get(feature, "unknown")].append(row)
        variation_breakdowns[feature] = {name: summarize(values) for name, values in sorted(grouped.items())}
    return {"split": split,
            "quality_gate": ("untracked_diagnostic_no_representative_claim" if diagnostic else
                             "claimed_paired_test" if split == "test" else "development_only"),
            "evidence_class": "pretrained-backbone" if model.native_weights_loaded else "tiny-synthetic-fixture",
            "confidence_evaluated": "max probability, not public entropy concentration", "summary": summarize(rows),
            "by": breakdowns, "variation_by": variation_breakdowns,
            "rows": rows, "temperatures": dict(model.temperatures)}


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
            # A previous calibration must not survive under an unfitted=1.0 receipt.
            model.temperatures[kind] = 1.0
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
    status = ("fitted-held-out" if all(row["status"] == "fitted-on-calibration"
                                   for row in fits.values()) else "partial-held-out")
    model.calibration_provenance = {"status": status, "partition": split, "sha256": split_sha256,
                                    "fits": fits, "method": "per-type-temperature-nll-v1", "bounds": [0.2, 5.0]}
    return model.calibration_provenance


def grouped_bootstrap(rows: list[dict], *, samples: int = 1000, seed: int = 42,
                      confidence: float = 0.95) -> dict:
    """Percentile intervals resampling source groups, never individual questions.

    This describes variation across the supplied source groups, not across model
    seeds or deployments. It does not establish semantic independence of those
    groups, repair evaluation leakage, or turn synthetic data into quality proof.
    Historical reports lacking explicit groups remain unknown; IDs are not used
    to guess a grouping. Soft-target correctness remains fractional accuracy.
    """
    if type(samples) is not int or samples < 2:
        raise ValueError("at least two bootstrap samples are required")
    if type(seed) is not int or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap seed or confidence")
    fields = ("accuracy", "nll", "brier", "ece", "ordinal_mae")
    result = {"method": "source-group-percentile-bootstrap-v1", "samples": samples,
              "seed": seed, "confidence": confidence, "groups": 0,
              "status": "unknown", "intervals": {key: None for key in fields}}
    grouped = defaultdict(list)
    for row in rows:
        if not isinstance(row.get("group"), str) or not row["group"]:
            result["reason"] = "explicit source-group metadata absent"
            return result
        grouped[row["group"]].append(row)
    result["groups"] = len(grouped)
    if len(grouped) < 2:
        result["reason"] = "fewer than two source groups"
        return result
    groups = [grouped[key] for key in sorted(grouped)]
    rng = random.Random(seed)  # Do not mutate training/evaluation global RNG.
    values = {key: [] for key in fields}
    for _ in range(samples):
        selected = [row for _ in groups for row in groups[rng.randrange(len(groups))]]
        metrics = summarize(selected)
        for key in fields:
            value = metrics[key]
            if value is not None:
                values[key].append(value)
    alpha = (1 - confidence) / 2
    for key, observations in values.items():
        if not observations:
            continue
        observations.sort()

        def quantile(q):
            index = (len(observations) - 1) * q
            lower = math.floor(index)
            upper = math.ceil(index)
            return observations[lower] + (observations[upper] - observations[lower]) * (index - lower)

        result["intervals"][key] = {"lower": quantile(alpha), "upper": quantile(1 - alpha),
                                    "valid_resamples": len(observations)}
    result["status"] = "measured"
    result["scope"] = "supplied source groups only; not across model seeds or deployments"
    return result


def summarize_seed_variation(reports: dict[int, dict]) -> dict:
    """Report observed variation across matched, declared model seeds.

    Source-group uncertainty remains separate. Distinct checkpoint hashes
    prevent direct reuse, but do not authenticate independent training.
    """
    if not isinstance(reports, dict) or len(reports) < 3 or any(
        type(seed) is not int or seed < 0 for seed in reports
    ):
        raise ValueError("at_least_three_distinct_nonnegative_seeds_required")
    fields = ("accuracy", "nll", "brier", "ece", "ordinal_mae")
    slices = ("type", "domain", "language", "option_order", "option_count", "length_bucket")
    variation_slices = ("colors", "levels", "wording", "question_ids", "domains",
                        "languages", "context_lengths")
    signatures = None
    split = None
    split_sha256 = None
    checkpoints = {}
    per_seed = {}
    per_slice = {field: {} for field in (*slices, *variation_slices)}

    def spread(values):
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("seed_metric_missing_in_some_runs")
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in values):
            raise ValueError("nonfinite_seed_metric")
        mean = sum(values) / len(values)
        return {"mean": mean, "sample_sd": math.sqrt(sum((value - mean) ** 2 for value in values)
                                                     / (len(values) - 1)),
                "minimum": min(values), "maximum": max(values), "values": list(values)}

    for seed, report in sorted(reports.items()):
        if not isinstance(report, dict) or not isinstance(report.get("split"), str) or not report["split"]:
            raise ValueError("seed_report_split_missing")
        if split is None:
            split = report["split"]
        elif report["split"] != split:
            raise ValueError("seed_report_split_mismatch")
        checkpoint = report.get("checkpoint_sha256")
        partition = report.get("split_sha256")
        if not isinstance(checkpoint, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint):
            raise ValueError("seed_checkpoint_identity_missing")
        if not isinstance(partition, str) or not re.fullmatch(r"[0-9a-f]{64}", partition):
            raise ValueError("seed_split_identity_missing")
        if checkpoint in checkpoints.values():
            raise ValueError("seed_checkpoint_reused")
        checkpoints[seed] = checkpoint
        if split_sha256 is None:
            split_sha256 = partition
        elif partition != split_sha256:
            raise ValueError("seed_split_identity_mismatch")
        rows = report.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("seed_report_rows_missing")
        identity = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("seed_report_row_invalid")
            key = (row.get("record_id"), row.get("question_id"))
            if any(not isinstance(part, str) or not part for part in key) or key in identity:
                raise ValueError("seed_report_row_identity_invalid")
            try:
                identity[key] = (row["group"], row["type"], row["domain"], row["language"],
                                 row["option_order"], row["option_count"], row["length_bucket"],
                                 tuple(row["target"]), tuple(sorted(row.get("variations", {}).items())))
            except (KeyError, TypeError, AttributeError):
                raise ValueError("seed_report_row_metadata_invalid") from None
        if signatures is None:
            signatures = identity
        elif identity != signatures:
            raise ValueError("seed_reports_not_same_frozen_rows_and_targets")
        per_seed[seed] = summarize(rows)
        for field in per_slice:
            grouped = defaultdict(list)
            for row in rows:
                value = row.get(field, "unknown") if field in slices else row.get("variations", {}).get(field, "unknown")
                grouped[str(value)].append(row)
            for name, group in grouped.items():
                per_slice[field].setdefault(name, {})[seed] = summarize(group)

    def metrics(summaries):
        return {field: spread([summaries[seed][field] for seed in sorted(reports)]) for field in fields}

    return {"method": "matched-model-seed-descriptive-v1", "split": split,
            "split_sha256": split_sha256, "checkpoint_sha256_by_seed": checkpoints,
            "seeds": sorted(reports), "seed_count": len(reports), "row_count_per_seed": len(signatures),
            "per_seed": per_seed, "between_seed": metrics(per_seed),
            "by": {field: {name: metrics(summaries) for name, summaries in sorted(groups.items())}
                   for field, groups in per_slice.items()},
            "scope": "observed variation across supplied model seeds only; source-group bootstrap separate"}
