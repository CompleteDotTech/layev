"""Exact context construction against the reviewed Qwen tokenizer bytes."""

import hashlib
import os
from pathlib import Path

import pytest

from kev_laya.encoding import Limits, QwenTokenizer, encode_request
from kev_laya.native_validation import make_case, verify_context_boundaries


PINNED_TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"


def test_six_native_context_strata_construct_exact_boundaries():
    path_value = os.environ.get("KEV_LAYA_TEST_TOKENIZER_JSON")
    if not path_value:
        pytest.skip("reviewed Qwen tokenizer path not supplied")
    path = Path(path_value)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINNED_TOKENIZER_SHA256
    tokenizer = QwenTokenizer(path)
    for position in ("beginning", "middle", "end"):
        for language in ("en", "es"):
            request = make_case(tokenizer, position, language)
            encoded = encode_request(request, tokenizer, Limits())
            assert len(encoded.branches) == 19
            assert max(len(encoded.state) + len(branch.ids) for branch in encoded.branches) == 32768
            assert encoded.logical_tokens == 65536
            assert max(len(branch.question.options()) for branch in encoded.branches
                       if branch.question.type == "choice") == 255
            assert max(len(branch.question.options()) for branch in encoded.branches
                       if branch.question.type == "score") == 10
            overflow = verify_context_boundaries(request, tokenizer, encoded)
            assert set(overflow) == {"aggregate_input", "state_plus_branch"}
