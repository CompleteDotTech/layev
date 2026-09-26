"""Explicit labeled datasets only. Serving never invokes this module."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import random
from .encoding import Encoding, Limits, Tokenizer, encode_request
from .schema import SystemOneRequest, canonical, strict_loads
from .io import atomic_json, sha256_file

PARTITIONS = ("train", "development", "calibration", "test")


@dataclass(frozen=True)
class Datum:
    request: SystemOneRequest
    targets: list[list[float]]
    meta: dict
    def encode(self, tokenizer: Tokenizer, limits: Limits) -> Encoding:
        return encode_request(self.request, tokenizer, limits)


def parse_datum(obj: dict) -> Datum:
    if set(obj) != {"state", "questions", "gold", "meta"}:
        raise ValueError("dataset row requires exactly state/questions/gold/meta")
    request = SystemOneRequest.model_validate({"state": obj["state"], "questions": obj["questions"], "model": "training"})
    if set(obj["gold"]) != set(request.questions):
        raise ValueError("gold IDs must match question IDs")
    meta = obj["meta"]
    if not all(isinstance(meta.get(k), str) and meta[k] for k in ("id", "group", "domain", "language")):
        raise ValueError("dataset requires id/group/domain/language provenance")
    targets = []
    for key, question in request.questions.items():
        keys = [key for key, _ in question.options()]
        target = obj["gold"][key]
        if isinstance(target, list):
            if len(target) != len(keys) or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in target) or abs(sum(target) - 1) > 1e-6:
                raise ValueError("invalid soft target")
            targets.append([float(v) for v in target])
        else:
            label = str(target).lower() if isinstance(target, bool) else str(target)
            if label not in keys:
                raise ValueError("hard target is not an option")
            targets.append([float(k == label) for k in keys])
    return Datum(request, targets, dict(meta))


def load_suite(directory: Path) -> tuple[dict[str, list[Datum]], dict]:
    directory = Path(directory)
    manifest = strict_loads((directory / "manifest.json").read_bytes())
    if manifest.get("schema_version") != 1 or set(manifest["partitions"]) != set(PARTITIONS):
        raise ValueError("invalid frozen suite manifest")
    suite, groups, ids, normalized = {}, {}, set(), {}
    for split in PARTITIONS:
        info = manifest["partitions"][split]
        name = info["file"]
        if Path(name).name != name:
            raise ValueError("partition file must be a basename")
        path = directory / name
        if sha256_file(path) != info["sha256"]:
            raise ValueError("partition checksum mismatch: " + split)
        data = [parse_datum(strict_loads(line)) for line in path.read_bytes().splitlines() if line.strip()]
        if not data or len(data) != info["rows"]:
            raise ValueError("partition count mismatch or empty partition")
        for item in data:
            if item.meta["id"] in ids:
                raise ValueError("duplicate source record ID")
            ids.add(item.meta["id"])
            group = item.meta["group"]
            if group in groups and groups[group] != split:
                raise ValueError("group leakage across splits")
            groups[group] = split
            norm = " ".join(canonical(item.request.state).casefold().split())
            state_hash = hashlib.sha256(norm.encode()).hexdigest()
            if state_hash in normalized and normalized[state_hash] != split:
                raise ValueError("normalized-state duplicate leakage")
            normalized[state_hash] = split
        suite[split] = data
    return suite, manifest


def freeze_smoke(directory: Path, count: int = 256) -> dict:
    """Generated software fixture, not a scientific quality benchmark or a parent baseline."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("refusing to replace existing suite")
    directory.mkdir(parents=True, exist_ok=True)
    partitions = {name: [] for name in PARTITIONS}
    colors = ("red", "blue")
    texts = {"en": "color", "es": "color", "fr": "couleur", "de": "Farbe"}
    # Source-case group variants can be added without changing this stable split assignment.
    for i in range(count):
        group = f"smoke-case-{i:05d}"
        bucket = int(hashlib.sha256(("smoke-v1/" + group).encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 60 else "development" if bucket < 73 else "calibration" if bucket < 86 else "test"
        color = colors[i % 2]
        level = i % 3
        language = list(texts)[(i // 2) % len(texts)]
        row = {
            "state": f"{texts[language]}={color}; level={level}; case={i:05d}",
            "questions": {
                "color": {"type": "choice", "instructions": "color?", "criteria": {"red": None, "blue": None}},
                "is_red": {"type": "noul", "instructions": "color=red?", "criteria": {"false": "blue", "true": "red"}},
                "level": {"type": "score", "instructions": "level?", "criteria": ["0", "1", "2"]}
            },
            "gold": {"color": color, "is_red": color == "red", "level": level},
            "meta": {"id": group, "group": group, "language": language, "domain": "synthetic-markers"}
        }
        partitions[split].append(row)
    manifest = {"schema_version": 1, "id": "smoke-v1", "license": "Apache-2.0",
                "provenance": "Generated by kev_laya.data.freeze_smoke; not external data or genuine multilingual understanding",
                "grouping": "source-case groups plus normalized-state duplicate rejection; near-duplicate semantic grouping needs curated real data",
                "partitions": {}}
    for name, rows in partitions.items():
        path = directory / (name + ".jsonl")
        path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
        manifest["partitions"][name] = {"file": path.name, "sha256": sha256_file(path), "rows": len(rows)}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


class Sampler:
    """Explicit permutation, cursor, epoch, and independent RNG for exact continuation."""
    def __init__(self, count: int, seed: int):
        if count <= 0:
            raise ValueError("empty training partition")
        self.rng = random.Random(seed)
        self.order = list(range(count))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.epoch = 0
    def next(self) -> int:
        if self.cursor == len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
            self.epoch += 1
        result = self.order[self.cursor]
        self.cursor += 1
        return result
    def state_dict(self) -> dict:
        return {"order": self.order[:], "cursor": self.cursor, "epoch": self.epoch, "rng": self.rng.getstate()}
    def load_state_dict(self, state: dict):
        if sorted(state["order"]) != list(range(len(self.order))) or not 0 <= state["cursor"] <= len(self.order):
            raise ValueError("sampler state does not match dataset")
        self.order, self.cursor, self.epoch = state["order"][:], state["cursor"], state["epoch"]
        self.rng.setstate(state["rng"])


def permute_choices(datum: Datum, rng) -> Datum:
    """Augment only unordered Choice sets, preserving each label's exact soft target.

    Score level order and Noul semantics never change. RNG state belongs to the
    trainer checkpoint, so interrupted training follows the identical permutations.
    """
    questions={};targets=[]
    for (key,question),target in zip(datum.request.questions.items(),datum.targets,strict=True):
        if question.type=="choice":
            labels=list(question.criteria)
            order=list(range(len(labels)));rng.shuffle(order)
            criteria={labels[i]:question.criteria[labels[i]] for i in order}
            questions[key]=question.model_copy(update={"criteria":criteria})
            targets.append([target[i] for i in order])
        else:
            questions[key]=question
            targets.append(target[:])
    return Datum(datum.request.model_copy(update={"questions":questions}),targets,dict(datum.meta))
