#!/usr/bin/env python3
"""Offline, fail-closed pinned-tokenizer gate. Not a model/CUDA/quality gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys

PINNED_REPOSITORY = 'Qwen/Qwen2.5-0.5B'
PINNED_REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
PINNED_RUNTIME = '0.22.1'


class GateBlocked(ValueError):
    """A missing/mismatched prerequisite, not a passing skipped test."""


def inspect_source(source_dir: str | None, expected_sha256: str | None) -> dict:
    """Verify existing files against a separately reviewed hash; never download."""
    if not source_dir:
        raise GateBlocked('source_directory_missing')
    if not expected_sha256 or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise GateBlocked('reviewed_tokenizer_sha256_missing_or_invalid')
    root = Path(source_dir)
    try:
        manifest_raw = (root / 'source.json').read_bytes()
        manifest = json.loads(manifest_raw)
        raw = (root / 'tokenizer.json').read_bytes()
    except (OSError, ValueError):
        raise GateBlocked('source_files_missing_or_invalid') from None
    if not isinstance(manifest, dict) or manifest.get('revision') != PINNED_REVISION:
        raise GateBlocked('source_revision_mismatch')
    hashes = manifest.get('sha256')
    digest = hashlib.sha256(raw).hexdigest()
    if not isinstance(hashes, dict) or hashes.get('tokenizer.json') != digest:
        raise GateBlocked('source_manifest_tokenizer_hash_mismatch')
    if digest != expected_sha256:
        raise GateBlocked('reviewed_tokenizer_hash_mismatch')
    return {'tokenizer_sha256': digest,
            'source_manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
            'tokenizer_bytes': len(raw)}


def exercise_tokenizer(source_dir: str, source_receipt: dict) -> dict:
    """Execute the real locally installed tokenizer; no injected transport/backend."""
    try:
        runtime = metadata.version('tokenizers')
    except metadata.PackageNotFoundError:
        raise GateBlocked('tokenizers_runtime_missing') from None
    if runtime != PINNED_RUNTIME:
        raise GateBlocked('tokenizers_runtime_version_mismatch')
    # Source checkout wins over an unrelated installed kev_laya distribution.
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / 'src'))
    from kev_laya.encoding import (
        ContextOverflow, LEGACY_LITERAL_ENCODING, LITERAL_ENCODING,
        Limits, QwenTokenizer, SPECIAL, encode_request, preprocessing_identity,
        validate_preprocessing_metadata,
    )
    from kev_laya.schema import SystemOneRequest, canonical
    from tokenizers import Tokenizer as RustTokenizer

    path = Path(source_dir) / 'tokenizer.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source_receipt['tokenizer_sha256']:
        raise GateBlocked('tokenizer_changed_after_source_verification')
    original = RustTokenizer.from_str(raw.decode('utf-8'))
    original.no_padding()
    original.no_truncation()
    added = original.get_added_tokens_decoder()
    tok = QwenTokenizer(path)
    if tok.identity != 'qwen-tokenizer-sha256:' + source_receipt['tokenizer_sha256']:
        raise GateBlocked('tokenizer_identity_changed_during_initialization')
    corpus = list(dict.fromkeys([
        *SPECIAL, *(t.content for t in added.values()), '',
        '<¦fim_prefix¦>', '\\<|fim_prefix|>\\', r'\u003c|fim_prefix|>',
        'e\u0301', 'é', '가', '\x00', '\r\n\t', '中文🙂', '\u2028\u2029',
        '<|fim_prefix|><|fim_prefix|>', 'before <|fim_prefix|> after',
        canonical({'nested': ['<|fim_prefix|>', 'e\u0301', {'escaped': '\\'}]}),
    ]))
    forbidden = set(added) | set(tok.special)
    for text in corpus:
        ids = tok.encode(text)
        if forbidden.intersection(ids) or tok.decode_literal(ids) != text:
            raise ValueError('literal_round_trip_or_control_id_failure')
    for text in corpus:
        if tok.encode_output(text) != original.encode(text, add_special_tokens=False).ids:
            raise ValueError('declared_output_policy_mismatch')
    if tok.encode('é') == tok.encode('e\u0301'):
        raise ValueError('composed_decomposed_unicode_collision')
    if tok.encode(SPECIAL[0]) == tok.encode('<¦fim_prefix¦>'):
        raise ValueError('literal_spelling_collision')
    req = SystemOneRequest(state={'nested': corpus}, model='tokenizer-gate', questions={
        'metadata-only-id': {'type': 'choice', 'instructions': {'x': ['e\u0301', SPECIAL[0]]},
                             'criteria': {SPECIAL[1]: [SPECIAL[2]], 'other': None}},
        'score': {'type': 'score', 'instructions': 'level', 'criteria': ['low', {'high': '🙂'}]},
        'noul': {'type': 'noul', 'instructions': 'e\u0301'},
    })
    enc = encode_request(req, tok, Limits())
    if json.loads(tok.decode_literal(list(enc.state[1:]))) != {'state': req.state}:
        raise ValueError('structured_state_round_trip_failure')
    if enc.logical_tokens != len(enc.state) + sum(len(b.ids) for b in enc.branches):
        raise ValueError('logical_token_accounting_failure')
    renamed = req.model_copy(update={'questions': {f'q{i}': q for i, q in enumerate(req.questions.values())}})
    again = encode_request(renamed, tok, Limits())
    if enc.state != again.state or [b.ids for b in enc.branches] != [b.ids for b in again.branches]:
        raise ValueError('question_id_leaked_into_model_tokens')
    branch_limit = len(enc.state) + max(len(b.ids) for b in enc.branches)
    if encode_request(req, tok, Limits(branch_limit, enc.logical_tokens)) != enc:
        raise ValueError('exact_budget_failure')
    for limits, expected in [
        (Limits(branch_limit - 1, enc.logical_tokens), 'state_plus_branch'),
        (Limits(branch_limit, enc.logical_tokens - 1), 'aggregate_input'),
    ]:
        try:
            encode_request(req, tok, limits)
        except ContextOverflow as exc:
            if exc.detail['limit'] != expected:
                raise ValueError('wrong_overflow_category') from None
        else:
            raise ValueError('over_budget_input_accepted')
    # This is tokenizer-metadata reload, NOT a trained-model checkpoint reload.
    reloaded = QwenTokenizer(path, literal_encoding=tok.metadata()['literal_encoding'])
    validate_preprocessing_metadata(reloaded, tok.metadata(), preprocessing_identity(tok))
    if preprocessing_identity(tok) != preprocessing_identity(reloaded):
        raise ValueError('tokenizer_reload_identity_mismatch')
    for text in corpus:
        if tok.encode(text) != reloaded.encode(text):
            raise ValueError('tokenizer_reload_token_mismatch')
    for version in ('literal-bpe-v3', 'literal-special-v3'):
        try:
            QwenTokenizer(path, literal_encoding=version)
        except ValueError:
            pass
        else:
            raise ValueError('ambiguous_v3_was_accepted')
    try:
        old = QwenTokenizer(path, literal_encoding=LEGACY_LITERAL_ENCODING)
        for text in corpus:
            old.encode(text)
        legacy = 'accepted_same_literal_corpus'
    except ValueError:
        legacy = 'explicit_legacy_refusal_preserved'
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_receipt['tokenizer_sha256']:
        raise ValueError('original_tokenizer_file_changed')
    return {'runtime': runtime, 'literal_encoding': LITERAL_ENCODING,
            'preprocessing': preprocessing_identity(tok),
            'literal_backend_sha256': tok.metadata()['literal_backend_sha256'],
            'output_encoding': tok.metadata()['output_encoding'], 'literal_cases': len(corpus),
            'logical_tokens': enc.logical_tokens, 'branch_limit_tested': branch_limit,
            'reserved_ids': len(forbidden), 'vocab_size': tok.metadata()['vocab_size'],
            'legacy_v2_result': legacy, 'tokenizer_metadata_reload': 'passed',
            'output_policy_comparison': 'passed', 'ambiguous_v3_refusal': 'passed',
            'encoding_source_sha256': hashlib.sha256((repo / 'src/kev_laya/encoding.py').read_bytes()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', default=os.environ.get('KEV_LAYA_QWEN_DIR'))
    parser.add_argument('--expected-tokenizer-sha256', help='Separately reviewed pinned-upstream hash, not an arbitrary local hash')
    args = parser.parse_args(argv)
    receipt = {
        'schema_version': 1, 'gate': 'pinned-qwen-tokenizer', 'status': 'blocked',
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'repository': PINNED_REPOSITORY, 'revision': PINNED_REVISION,
        'python': platform.python_version(), 'platform': platform.system(),
        'hash_authority': 'caller-supplied separately reviewed pinned-source hash',
        'native_model': 'not_tested', 'cuda': 'not_tested',
        'checkpoint_and_serving': 'not_tested', 'trained_context': 'not_tested',
        'quality': 'not_tested', 'jev_parity': 'unknown',
        'network_calls': 0, 'source_writes': 0, 'literal_cases_executed': 0,
    }
    try:
        source = inspect_source(args.source_dir, args.expected_tokenizer_sha256)
        receipt.update(source)
        receipt['literal_cases_executed'] = None  # Unknown until the full gate completes.
        result = exercise_tokenizer(args.source_dir, source)
        receipt.update(result, status='passed', literal_cases_executed=result['literal_cases'])
        result_code = 0
    except GateBlocked as exc:
        receipt['reason'] = str(exc)
        result_code = 2
    except Exception as exc:
        # Do not echo raw file paths, tokenizer content or arbitrary exception text.
        receipt.update(status='failed', reason='tokenizer_gate_failed', error_type=type(exc).__name__)
        result_code = 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return result_code


if __name__ == '__main__':
    raise SystemExit(main())
