"""Frozen System One wire contract, independent of the model runtime."""
from __future__ import annotations
import json
import math
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def validate_json(value: Any, depth: int = 0) -> Any:
    if depth > 32:
        raise ValueError("JSON nesting exceeds 32")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, list):
        for child in value:
            validate_json(child, depth + 1)
        return value
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for child in value.values():
            validate_json(child, depth + 1)
        return value
    raise ValueError("not a JSON value with string keys")


def entry(value: Any) -> Any:
    if value is not None and not isinstance(value, (str, dict, list)):
        raise ValueError("entry must be string, object, array, or null")
    return validate_json(value)


def canonical(value: Any) -> str:
    validate_json(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def render(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Question(StrictModel):
    type: Literal["choice", "score", "noul"]
    instructions: Any
    criteria: Any = None

    @field_validator("instructions")
    @classmethod
    def valid_instructions(cls, value: Any) -> Any:
        return entry(value)

    @model_validator(mode="after")
    def valid_criteria(self) -> Question:
        c = self.criteria
        if self.type == "choice":
            if not isinstance(c, dict) or not 1 <= len(c) <= 255:
                raise ValueError("Choice requires 1..255 options")
            if not all(isinstance(k, str) and 1 <= len(k) <= 512 for k in c):
                raise ValueError("option keys must be nonempty strings of at most 512 characters")
            for v in c.values():
                entry(v)
        elif self.type == "score":
            if not isinstance(c, list) or not 2 <= len(c) <= 10:
                raise ValueError("Score requires 2..10 levels")
            for v in c:
                entry(v)
        elif c is not None:
            if not isinstance(c, dict) or set(c) - {"true", "false"}:
                raise ValueError("Noul criteria may contain only true and false")
            for v in c.values():
                entry(v)
        return self

    def options(self) -> list[tuple[str, Any]]:
        if self.type == "choice":
            return list(self.criteria.items())
        if self.type == "score":
            return [(str(i), v) for i, v in enumerate(self.criteria)]
        criteria = self.criteria or {}
        return [("false", criteria["false"] if criteria.get("false") is not None else "No: the statement does not hold"),
                ("true", criteria["true"] if criteria.get("true") is not None else "Yes: the statement holds")]


class SystemOneRequest(StrictModel):
    state: Any
    model: str = Field(min_length=1, max_length=200)
    questions: dict[str, Question] = Field(min_length=1, max_length=1024)

    @field_validator("state")
    @classmethod
    def valid_state(cls, value: Any) -> Any:
        if not isinstance(value, (str, dict, list)):
            raise ValueError("state must be string, object, or array")
        return validate_json(value)

    @field_validator("questions")
    @classmethod
    def valid_ids(cls, value: dict[str, Question]) -> dict[str, Question]:
        if any(not key or len(key) > 512 for key in value):
            raise ValueError("question IDs must contain 1..512 characters")
        return value


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float = Field(ge=0, le=1)
    legend: dict[str, str]
    probabilities: dict[str, float]


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1)


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    forward_tokens: int = Field(ge=0)
    state_tokens: int = Field(ge=0)
    branch_tokens: list[int]
    generated_tokens: Literal[0] = 0


class SystemOneResponse(StrictModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    confidence_definition: Literal["entropy-concentration-v1"] = "entropy-concentration-v1"
    calibration_status: str


def strict_loads(raw: bytes | str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result
    def constant(_: str) -> Any:
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
