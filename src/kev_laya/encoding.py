"""Lossless, versioned serialization with explicit logical token budgets."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import hashlib
import json
import re
from .schema import Question, SystemOneRequest, canonical, render, strict_loads

SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
SERIALIZATION = "kev-laya-prefix-v1"
NATIVE_SERIALIZATION = "kev-laya-prefix-v2"
LEGACY_ESCAPE = "special-spelling-v1"
LEGACY_LITERAL_ENCODING = "literal-special-v2"
LITERAL_ENCODING = "literal-bpe-v4"
OUTPUT_ENCODING = "source-json-output-v1"
AMBIGUOUS_LITERAL_ENCODINGS = frozenset({"literal-bpe-v3", "literal-special-v3"})


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


def literal_backend_json(raw: bytes) -> str:
    """Derive deterministic ordinary BPE text from the fingerprinted source bytes.

    No vocabulary, merges, pre-tokenizer or decoder replacement is permitted.
    Exact round trips remain mandatory even for a structurally accepted backend.
    """
    payload = strict_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tokenizer source must be a JSON object")
    model = payload.get("model")
    decoder = payload.get("decoder")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise ValueError("literal backend requires a BPE model")
    if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
        raise ValueError("literal backend requires the original ByteLevel decoder")
    dropout = model.get("dropout")
    if dropout is not None and (type(dropout) not in (int, float) or dropout != 0):
        raise ValueError("stochastic BPE dropout is not supported")
    if not isinstance(payload.get("added_tokens"), list):
        raise ValueError("tokenizer source must declare added_tokens")
    for field in ("normalizer", "post_processor", "truncation", "padding"):
        payload[field] = None
    payload["added_tokens"] = []
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


class QwenTokenizer:
    """Versioned lossless literal input with explicitly separate output counting.

    v4 chooses the source JSON output backend, without padding/truncation or
    automatically inserted special tokens. That policy may normalize output text
    and match added-token spellings; it is NOT the caller-input encoding policy.
    The derived literal backend and output policy both enter checkpoint identity.

    Prior unpublished v3 implementations collided on identity while differing in
    output accounting. Both tags are refused. Explicit v1/v2 retain historical
    behavior and limitations; no checkpoint is silently upgraded.
    """
    def __init__(self, path: str | Path, *, literal_encoding: str = LITERAL_ENCODING):
        if not isinstance(literal_encoding, str):
            raise ValueError("unsupported literal-tokenization version; explicit migration required")
        if literal_encoding in AMBIGUOUS_LITERAL_ENCODINGS:
            raise ValueError("ambiguous unpublished v3 preprocessing version; "
                             "reinitialize, retrain and recalibrate under literal-bpe-v4")
        from tokenizers import Tokenizer as RustTokenizer
        if literal_encoding not in {LITERAL_ENCODING, LEGACY_LITERAL_ENCODING, LEGACY_ESCAPE}:
            raise ValueError("unsupported literal-tokenization version; explicit migration required")
        self.path = Path(path)
        # Parse exactly the bytes being fingerprinted; do not reread a changed file.
        raw = self.path.read_bytes()
        self._tokenizer = RustTokenizer.from_str(raw.decode("utf-8"))
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()
        self._output_tokenizer = self._tokenizer
        ids = tuple(self._tokenizer.token_to_id(token) for token in SPECIAL)
        if None in ids or len(set(ids)) != 5:
            raise ValueError("pinned tokenizer lacks distinct required delimiter tokens")
        self.special = ids
        self.identity = "qwen-tokenizer-sha256:" + hashlib.sha256(raw).hexdigest()
        self.literal_encoding = literal_encoding
        self.serialization = SERIALIZATION if literal_encoding == LEGACY_ESCAPE else NATIVE_SERIALIZATION
        self._special_ids = set(ids)
        if literal_encoding == LITERAL_ENCODING:
            # The flag encode_special_tokens alone does not bypass non-special
            # added tokens. Derive a literal backend with NO added-token matcher.
            self._special_ids.update(self._tokenizer.get_added_tokens_decoder())
            literal_json = literal_backend_json(raw)
            self.literal_backend_sha256 = hashlib.sha256(literal_json.encode("utf-8")).hexdigest()
            self.output_encoding = OUTPUT_ENCODING
            self._tokenizer = RustTokenizer.from_str(literal_json)
            self._tokenizer.no_padding()
            self._tokenizer.no_truncation()
            if self._tokenizer.get_added_tokens_decoder():
                raise ValueError("literal backend retained added-token extraction")
        elif literal_encoding == LEGACY_LITERAL_ENCODING:
            if not hasattr(self._tokenizer, "encode_special_tokens"):
                raise ValueError("tokenizer backend lacks literal-special support; use tokenizers==0.22.1")
            self._special_ids.update(i for i, t in self._tokenizer.get_added_tokens_decoder().items() if t.special)
            self._tokenizer.encode_special_tokens = True
        if literal_encoding != LEGACY_ESCAPE:
            probes = (*SPECIAL, "<¦fim_prefix¦>", "\\<|fim_prefix|>\\", "é中文🙂\n")
            if literal_encoding == LITERAL_ENCODING:
                probes += ("e\u0301", "\x00", "<|im_start|>")
            for text in probes:
                self.encode(text)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("literal input must be a string")
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ValueError("unpaired Unicode surrogates are unsupported; no silent replacement") from None
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
        if not isinstance(text, str):
            raise TypeError("output text must be a string")
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ValueError("unpaired Unicode surrogates are unsupported; no silent replacement") from None
        return self._output_tokenizer.encode(text, add_special_tokens=False).ids

    def metadata(self) -> dict:
        meta = {"kind": "qwen", "identity": self.identity, "path": str(self.path),
                "escape": self.literal_encoding, "literal_encoding": self.literal_encoding,
                "serialization": self.serialization, "vocab_size": self._output_tokenizer.get_vocab_size()}
        if self.literal_encoding == LITERAL_ENCODING:
            meta.update(literal_backend_sha256=self.literal_backend_sha256,
                        output_encoding=self.output_encoding)
        return meta


def preprocessing_identity(tokenizer) -> dict:
    """Stable semantic identity, excluding a machine-specific tokenizer file path."""
    meta = tokenizer.metadata()
    identity = {"tokenizer": tokenizer.identity,
                "serialization": meta.get("serialization", SERIALIZATION),
                "literal_encoding": meta.get("literal_encoding", meta.get("escape", "utf8-bytes-v1"))}
    if identity["literal_encoding"] == LITERAL_ENCODING:
        # Telemetry v2 requires exactly these THREE string fields and requires
        # tokenizer to equal the original source identity. Bind extra semantics
        # into the literal identity string rather than changing the wire shape.
        specification = {"version": LITERAL_ENCODING,
                         "literal_backend_sha256": meta["literal_backend_sha256"],
                         "output_encoding": meta["output_encoding"]}
        digest = hashlib.sha256(canonical(specification).encode("utf-8")).hexdigest()
        identity["literal_encoding"] = LITERAL_ENCODING + ":" + digest
    return identity


def validate_preprocessing_metadata(tokenizer, meta: dict, recorded: dict | None) -> None:
    """Reject omitted, conflicting or relabeled v4 checkpoint identity.

    The loader supplies the historical default for legacy omission only. v4
    never existed without its full identity, so omission is not a valid default.
    """
    expected = preprocessing_identity(tokenizer)
    if tokenizer.metadata().get("literal_encoding") == LITERAL_ENCODING:
        if not isinstance(recorded, dict):
            raise ValueError("v4 checkpoint requires explicit preprocessing identity")
        actual = tokenizer.metadata()
        for key in ("literal_encoding", "escape", "serialization",
                    "literal_backend_sha256", "output_encoding"):
            if meta.get(key) != actual[key]:
                raise ValueError("checkpoint tokenizer metadata mismatch: " + key)
    if recorded != expected:
        raise ValueError("checkpoint preprocessing identity mismatch")


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
