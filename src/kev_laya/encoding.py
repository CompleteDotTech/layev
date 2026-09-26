"""Lossless, versioned serialization with explicit logical token budgets."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import hashlib
import re
from .schema import Question, SystemOneRequest, canonical, render

SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
SERIALIZATION = "kev-laya-prefix-v1"
NATIVE_SERIALIZATION = "kev-laya-prefix-v2"
LEGACY_ESCAPE = "special-spelling-v1"
LITERAL_ENCODING = "literal-special-v2"


class Tokenizer(Protocol):
    special: tuple[int, ...]
    identity: str
    def encode(self, text: str) -> list[int]: ...
    def metadata(self) -> dict: ...


class ByteTokenizer:
    """UTF-8 fixture tokenizer, deliberately NOT the pretrained model's tokenizer."""
    special = (256, 257, 258, 259, 260)
    identity = "utf8-byte-fixture-v1"
    serialization = SERIALIZATION
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))
    def decode_literal(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="strict")
    def metadata(self) -> dict:
        return {"kind": "byte-fixture", "identity": self.identity, "vocab_size": 261,
                "serialization": self.serialization, "literal_encoding": "utf8-bytes-v1"}


class QwenTokenizer:
    """Literal content is ordinary BPE text, never an added special token.

    v2 does not escape/replace caller characters. The pinned Rust tokenizer is
    configured once to bypass added-special recognition, and every result is
    checked against its decoder. This left-inverse check establishes injectivity
    for accepted inputs, including arbitrary Unicode. A lossy tokenizer fails
    explicitly instead of changing content. No per-call tokenizer mutations.

    v1 is retained ONLY to interpret checkpoints explicitly recording the legacy
    lossy spelling rule. Loading old weights never silently selects v2.
    """
    def __init__(self, path: str | Path, *, literal_encoding: str = LITERAL_ENCODING):
        from tokenizers import Tokenizer as RustTokenizer
        if literal_encoding not in {LITERAL_ENCODING, LEGACY_ESCAPE}:
            raise ValueError("unsupported literal-tokenization version")
        self.path = Path(path)
        self._tokenizer = RustTokenizer.from_file(str(self.path))
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()
        ids = tuple(self._tokenizer.token_to_id(token) for token in SPECIAL)
        if None in ids or len(set(ids)) != 5:
            raise ValueError("pinned tokenizer lacks distinct required delimiter tokens")
        self.special = ids
        self.identity = "qwen-tokenizer-sha256:" + hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.literal_encoding = literal_encoding
        self.serialization = NATIVE_SERIALIZATION if literal_encoding == LITERAL_ENCODING else SERIALIZATION
        self._special_ids = set(ids)
        if literal_encoding == LITERAL_ENCODING:
            # tokenizers 0.22.1: encode_special_tokens=True means special spellings
            # pass through the ordinary model, rather than added-token extraction.
            if not hasattr(self._tokenizer, "encode_special_tokens"):
                raise ValueError("tokenizer backend lacks literal-special support; use tokenizers==0.22.1")
            self._special_ids.update(i for i, t in self._tokenizer.get_added_tokens_decoder().items() if t.special)
            self._tokenizer.encode_special_tokens = True
            # Check the actual tokenizer/decoder, not a regex-only surrogate.
            for literal in (*SPECIAL, "<¦fim_prefix¦>", "\\<|fim_prefix|>\\", "é中文🙂\n"):
                self.encode(literal)

    def encode(self, text: str) -> list[int]:
        if self.literal_encoding == LEGACY_ESCAPE:
            safe = re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text)
            ids = self._tokenizer.encode(safe, add_special_tokens=False).ids
        else:
            ids = self._tokenizer.encode(text, add_special_tokens=False).ids
            if self.decode_literal(ids) != text:
                raise ValueError("tokenizer is not reversible for this literal input; no lossy fallback")
        if self._special_ids.intersection(ids):
            raise ValueError("control token detected in user input")
        return ids

    def decode_literal(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def encode_output(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def metadata(self) -> dict:
        return {"kind": "qwen", "identity": self.identity, "path": str(self.path),
                "escape": self.literal_encoding, "literal_encoding": self.literal_encoding,
                "serialization": self.serialization, "vocab_size": self._tokenizer.get_vocab_size()}


def preprocessing_identity(tokenizer) -> dict:
    """Stable semantic identity, excluding a machine-specific tokenizer file path."""
    meta = tokenizer.metadata()
    return {"tokenizer": tokenizer.identity,
            "serialization": meta.get("serialization", SERIALIZATION),
            "literal_encoding": meta.get("literal_encoding", meta.get("escape", "utf8-bytes-v1"))}


def serialize_state(state, tokenizer) -> str:
    if getattr(tokenizer, "serialization", SERIALIZATION) == NATIVE_SERIALIZATION:
        # Unlike legacy render(), preserve the difference between a JSON object
        # and a string that happens to spell that object. JSON escaping is reversible.
        return canonical({"state": state})
    return render(state)


@dataclass(frozen=True)
class Limits:
    branch: int = 32768
    aggregate: int = 65536
    max_questions: int = 1024
    def __post_init__(self):
        if self.branch < 8 or self.aggregate < self.branch or self.max_questions < 1:
            raise ValueError("invalid context limits")


class ContextOverflow(ValueError):
    def __init__(self, limit: str, actual: int, maximum: int, question_id: str | None = None):
        self.detail = {"code": "context_overflow", "limit": limit, "actual": actual,
                       "maximum": maximum, "question_id": question_id}
        super().__init__(f"{limit}: {actual} exceeds {maximum}")


@dataclass(frozen=True)
class Branch:
    question_id: str
    question: Question
    ids: tuple[int, ...]
    option_ends: tuple[int, ...]


@dataclass(frozen=True)
class Encoding:
    state: tuple[int, ...]
    branches: tuple[Branch, ...]
    logical_tokens: int


def encode_request(request: SystemOneRequest, tokenizer: Tokenizer, limits: Limits) -> Encoding:
    if len(request.questions) > limits.max_questions:
        raise ContextOverflow("questions", len(request.questions), limits.max_questions)
    s, q, o, e, d = tokenizer.special
    state = (s, *tokenizer.encode(serialize_state(request.state, tokenizer)))
    if len(state) >= limits.branch:
        raise ContextOverflow("state_plus_branch", len(state), limits.branch)
    branches = []
    total = len(state)
    for key, question in request.questions.items():
        ids = [q, *tokenizer.encode(canonical({"type": question.type, "instructions": question.instructions}))]
        ends = []
        for label, rubric in question.options():
            ids.extend((o, *tokenizer.encode(canonical({"label": label, "criterion": rubric})), e))
            ends.append(len(ids) - 1)
        ids.append(d)
        if len(state) + len(ids) > limits.branch:
            raise ContextOverflow("state_plus_branch", len(state) + len(ids), limits.branch, key)
        total += len(ids)
        if total > limits.aggregate:
            raise ContextOverflow("aggregate_input", total, limits.aggregate)
        branches.append(Branch(key, question, tuple(ids), tuple(ends)))
    return Encoding(state, tuple(branches), total)
