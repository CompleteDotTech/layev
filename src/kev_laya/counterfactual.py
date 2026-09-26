"""Frozen balanced marker diagnostics, not representative natural-language data.

Every source-case group contains every color x ordinal-level combination. Case,
language marker, distractor, and option order are fixed within a group and cannot
predict its answers. All six siblings stay in one split. This probes shortcuts;
it is not a multilingual-comprehension or Jev benchmark.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import random
from .data import PARTITIONS, load_suite, parse_datum
from .io import atomic_json, sha256_file
from .schema import canonical

PROTOCOL = {
    "id": "counterfactual-markers-v1", "seed": 42, "source_groups": 128,
    "counterfactuals_per_group": 6, "sft_steps": 160, "matched_extension_steps": 40,
    "learning_rate": .003, "extension_learning_rate": .003, "accumulation": 2,
    "choice_permutation": True, "ordinal_coefficient": .1, "pgps_coefficient": .1,
    "temperature_iterations": 60, "accuracy_threshold": .70,
    "selection": "none; fixed terminal checkpoint of each arm",
    "arms": ["supervised-160", "supervised-200", "pgps-200", "pgps-200-calibrated"],
    "readout": "test only after all weight-training arms finish; calibration split only for temperature",
    "factors": {"color": ["red", "blue"], "level": [0, 1, 2], "language_marker": ["en", "es", "fr", "de"]},
    "grouping": "all six counterfactual siblings stay in the same partition; exact-state duplicate rejection",
    "scope": "synthetic marker software diagnostic; not semantic-near-duplicate or natural-language generalization evidence",
    "known_regression": "color=red; level=1; case=99999 (excluded from all four suite partitions)",
    "prior_failure_preserved": {"fixture": "original parallel short smoke", "accuracy": .5, "threshold": .7},
}


def definitions(labels=("red", "blue")):
    return {
        "color": {"type": "choice", "instructions": "color?", "criteria": {k: None for k in labels}},
        "is_red": {"type": "noul", "instructions": "color=red?", "criteria": {"false": "blue", "true": "red"}},
        "level": {"type": "score", "instructions": "level?", "criteria": ["0", "1", "2"]},
    }


def freeze_counterfactual(directory: Path) -> dict:
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("refusing to replace a frozen counterfactual suite")
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(PROTOCOL["seed"])
    cases = rng.sample(range(10000, 99999), PROTOCOL["source_groups"])
    partitions = {s: [] for s in PARTITIONS}
    for i, case in enumerate(cases):
        group = f"counterfactual-source-{case}"
        bucket = int(hashlib.sha256((PROTOCOL["id"] + '/' + group).encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 60 else "development" if bucket < 73 else "calibration" if bucket < 86 else "test"
        language = ("en", "es", "fr", "de")[i % 4]
        decoy = rng.choice(("red", "blue"))
        labels = ("red", "blue") if rng.randrange(2) else ("blue", "red")
        for color in ("red", "blue"):
            for level in (0, 1, 2):
                row = {
                    "state": f"color={color}; level={level}; case={case}; language={language}; distractor={decoy}",
                    "questions": definitions(labels),
                    "gold": {"color": color, "is_red": color == "red", "level": level},
                    "meta": {"id": f"{group}/{color}/{level}", "group": group,
                             "domain": "synthetic-counterfactual-markers", "language": language},
                }
                partitions[split].append(row)
    manifest = {
        "schema_version": 1, "id": PROTOCOL["id"], "license": "Apache-2.0",
        "provenance": "Generated locally by kev_laya.counterfactual.freeze_counterfactual; no user or served data",
        "grouping": PROTOCOL["grouping"], "protocol": copy.deepcopy(PROTOCOL),
        "generator_sha256": sha256_file(Path(__file__)), "partitions": {},
    }
    for split, rows in partitions.items():
        path = directory / f'{split}.jsonl'
        path.write_text(''.join(canonical(row) + '\n' for row in rows), encoding='utf-8')
        manifest["partitions"][split] = {"file": path.name, "rows": len(rows), "sha256": sha256_file(path)}
    atomic_json(directory / 'manifest.json', manifest)
    load_suite(directory)  # Enforce the normal grouping/hash/duplicate contract.
    return manifest


def regression_data():
    """Previously specified regression plus paired distractor/case/option-order probes.

    Kept separate from all untouched evaluation partitions and never used for fit.
    """
    rows = []
    for case in (99999, 100000, 1234567):
        for level in (0, 1, 2):
            for color in ("red", "blue"):
                for order in (("red", "blue"), ("blue", "red")):
                    row = {"state": f"color={color}; level={level}; case={case}",
                           "questions": definitions(order),
                           "gold": {"color": color, "is_red": color == "red", "level": level},
                           "meta": {"id": f"known-regression/{case}/{color}/{level}/{order[0]}",
                                    "group": f"known-regression/{case}", "language": "en",
                                    "domain": "observed-regression-not-held-out-quality"}}
                    rows.append(parse_datum(row))
    return rows
