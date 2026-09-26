"""Deterministic, request-local question microbatch planning (no model imports).

SPDX-License-Identifier: Apache-2.0
Right-padded causal rows adapt Kev's cached-branch scheduling and Laya's row
collation. All budgets are checked before the first model pass. No truncation,
no implicit singleton fallback, and no dependence on question IDs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from .encoding import Encoding

EXECUTION_VERSION = "parallel-questions-v1"


@dataclass(frozen=True)
class BatchPolicy:
    max_branches: int = 16
    max_padded_tokens: int = 65536
    max_cache_bytes: int = 536870912
    sort_by_length: bool = True

    def __post_init__(self):
        for name in ("max_branches", "max_padded_tokens", "max_cache_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.sort_by_length) is not bool:
            raise ValueError("sort_by_length must be a boolean")

    def to_dict(self) -> dict:
        return {"version": EXECUTION_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> BatchPolicy:
        data = dict(data)
        if data.pop("version", EXECUTION_VERSION) != EXECUTION_VERSION:
            raise ValueError("unsupported question execution version")
        return cls(**data)


class BatchBudgetExceeded(ValueError):
    def __init__(self, limit: str, actual: int, maximum: int, question_id: str):
        self.detail = {"code": "question_batch_budget_exceeded", "limit": limit,
                       "actual": actual, "maximum": maximum, "question_id": question_id}
        super().__init__(f"question batch {limit}: {actual} exceeds {maximum}")


@dataclass(frozen=True)
class BranchBatch:
    indices: tuple[int, ...]
    lengths: tuple[int, ...]
    padded_length: int
    padded_tokens_with_prefix: int
    estimated_cache_bytes: int


def plan_batches(encoding: Encoding, policy: BatchPolicy, *, kv_bytes_per_token: int,
                 gqa_bytes_per_token: int) -> tuple[BranchBatch, ...]:
    """Budget parent KV + per-row concatenated KV + materialized GQA K/V.

    Byte rates include all layers and use parameter element size (conservative
    under autocast). This is not a total process/activation memory bound. During
    training autograd can retain tensors across multiple microbatches.
    """
    if not encoding.state or not encoding.branches:
        raise ValueError("nonempty state and questions required")
    if kv_bytes_per_token < 1 or gqa_bytes_per_token < 1:
        raise ValueError("positive cache byte rates required")
    state_length = len(encoding.state)

    def costs(indices):
        longest = max(len(encoding.branches[i].ids) for i in indices)
        tokens = len(indices) * (state_length + longest)
        cache = state_length * kv_bytes_per_token + tokens * (kv_bytes_per_token + gqa_bytes_per_token)
        return longest, tokens, cache

    order = list(range(len(encoding.branches)))
    if policy.sort_by_length:
        order.sort(key=lambda i: len(encoding.branches[i].ids))
    # Validate all singleton rows up front: no partial model work before failure.
    for i in order:
        branch = encoding.branches[i]
        if not branch.ids or not branch.option_ends or any(p < 0 or p >= len(branch.ids) - 1 for p in branch.option_ends):
            raise ValueError("invalid branch readout positions")
        _, tokens, cache = costs([i])
        for key, actual, maximum in (("padded_tokens", tokens, policy.max_padded_tokens),
                                     ("cache_bytes", cache, policy.max_cache_bytes)):
            if actual > maximum:
                raise BatchBudgetExceeded(key, actual, maximum, branch.question_id)

    result, current = [], []

    def finish():
        longest, tokens, cache = costs(current)
        result.append(BranchBatch(tuple(current), tuple(len(encoding.branches[i].ids) for i in current),
                                  longest, tokens, cache))

    for i in order:
        trial = [*current, i]
        _, tokens, cache = costs(trial)
        if current and (len(trial) > policy.max_branches or tokens > policy.max_padded_tokens
                        or cache > policy.max_cache_bytes):
            finish()
            current = []
        current.append(i)
    finish()
    return tuple(result)
