"""Source contract tests. The backend double is NOT Rust, Qwen or native evidence."""
import hashlib
import json
import sys

import pytest

from kev_laya import encoding
from test_literal_v4 import config, source  # noqa: F401


@pytest.mark.parametrize('mode', ['literal-bpe-v3', 'literal-special-v3'])
def test_unpublished_v3_is_refused_before_loading_any_backend(tmp_path, monkeypatch, mode):
    monkeypatch.setitem(sys.modules, 'tokenizers', None)
    with pytest.raises(ValueError, match='ambiguous unpublished v3'):
        encoding.QwenTokenizer(tmp_path / 'absent.json', literal_encoding=mode)


def test_v4_has_full_semantic_identity(source):
    tok = encoding.QwenTokenizer(source)
    meta = tok.metadata()
    identity = encoding.preprocessing_identity(tok)
    assert identity['literal_encoding'].startswith('literal-bpe-v4:')
    assert meta['output_encoding'] == 'source-json-output-v1'
    assert meta['literal_backend_sha256'] == hashlib.sha256(
        encoding.literal_backend_json(source.read_bytes()).encode()).hexdigest()
    assert 'path' not in identity
    encoding.validate_preprocessing_metadata(tok, meta, identity)
    assert tok.encode_output(encoding.SPECIAL[0]) == [256]
    assert tok.encode_output('e\u0301') == list('é'.encode())
    assert tok.encode('e\u0301') == list('e\u0301'.encode())


def test_metadata_is_fresh_and_path_does_not_change_semantic_identity(source, tmp_path):
    tok = encoding.QwenTokenizer(source)
    path = tmp_path / 'relocated.json'
    path.write_bytes(source.read_bytes())
    other = encoding.QwenTokenizer(path)
    assert tok.metadata()['path'] != other.metadata()['path']
    assert encoding.preprocessing_identity(tok) == encoding.preprocessing_identity(other)
    meta = tok.metadata()
    meta['output_encoding'] = 'mutated-by-caller'
    assert tok.metadata()['output_encoding'] == 'source-json-output-v1'


@pytest.mark.parametrize('recorded', [None, {}, [], 'identity', {'literal_encoding': 'literal-bpe-v4'}])
def test_incomplete_recorded_v4_identity_is_refused(source, recorded):
    tok = encoding.QwenTokenizer(source)
    with pytest.raises(ValueError, match='identity'):
        encoding.validate_preprocessing_metadata(tok, tok.metadata(), recorded)


@pytest.mark.parametrize('key', ['literal_backend_sha256', 'output_encoding', 'serialization',
                                'literal_encoding', 'escape'])
@pytest.mark.parametrize('mutation', ['missing', 'altered'])
def test_each_v4_metadata_field_is_required_and_bound(source, key, mutation):
    tok = encoding.QwenTokenizer(source)
    meta = tok.metadata()
    if mutation == 'missing':
        del meta[key]
    else:
        meta[key] = 'other-policy'
    with pytest.raises(ValueError, match='metadata mismatch'):
        encoding.validate_preprocessing_metadata(tok, meta, encoding.preprocessing_identity(tok))


@pytest.mark.parametrize('key', ['tokenizer', 'serialization', 'literal_encoding'])
def test_recorded_identity_tampering_is_refused(source, key):
    tok = encoding.QwenTokenizer(source)
    identity = encoding.preprocessing_identity(tok)
    identity[key] = 'other'
    with pytest.raises(ValueError, match='identity mismatch'):
        encoding.validate_preprocessing_metadata(tok, tok.metadata(), identity)


def test_unexpected_identity_fields_are_not_ignored(source):
    tok = encoding.QwenTokenizer(source)
    identity = encoding.preprocessing_identity(tok) | {'undeclared_semantics': True}
    with pytest.raises(ValueError, match='identity mismatch'):
        encoding.validate_preprocessing_metadata(tok, tok.metadata(), identity)


def test_byte_fixture_identity_does_not_gain_native_claims():
    tok = encoding.ByteTokenizer()
    expected = {'tokenizer': 'utf8-byte-fixture-v1', 'serialization': 'kev-laya-prefix-v1',
                'literal_encoding': 'utf8-bytes-v1'}
    assert encoding.preprocessing_identity(tok) == expected
    encoding.validate_preprocessing_metadata(tok, tok.metadata(), expected)


@pytest.mark.parametrize('value', ['\ud800', '\udfff', 'before\ud800after'])
@pytest.mark.parametrize('method', ['encode', 'encode_output'])
def test_surrogates_are_refused_without_replacement(source, value, method):
    tok = encoding.QwenTokenizer(source)
    with pytest.raises(ValueError, match='no silent replacement'):
        getattr(tok, method)(value)


@pytest.mark.parametrize('value', [None, b'bytes', 3, [], {}])
@pytest.mark.parametrize('method', ['encode', 'encode_output'])
def test_nonstring_input_is_not_silently_coerced(source, value, method):
    tok = encoding.QwenTokenizer(source)
    with pytest.raises(TypeError, match='must be a string'):
        getattr(tok, method)(value)


@pytest.mark.parametrize('dropout', [0.1, 1, -0.1, True, '0', [], {}])
def test_stochastic_or_malformed_dropout_is_refused(dropout):
    payload = config()
    payload['model']['dropout'] = dropout
    with pytest.raises(ValueError, match='dropout'):
        encoding.literal_backend_json(json.dumps(payload).encode())


@pytest.mark.parametrize('dropout', [None, 0, 0.0])
def test_deterministic_dropout_is_accepted(dropout):
    payload = config()
    payload['model']['dropout'] = dropout
    literal = json.loads(encoding.literal_backend_json(json.dumps(payload).encode()))
    assert literal['model'] == payload['model']


@pytest.mark.parametrize('field, value', [('model', {'type': 'WordPiece'}), ('model', None),
                                        ('decoder', {'type': 'WordPiece'}), ('decoder', []),
                                        ('added_tokens', None), ('added_tokens', {})])
def test_unverified_tokenizer_families_and_matcher_shapes_are_refused(field, value):
    payload = config()
    payload[field] = value
    with pytest.raises(ValueError):
        encoding.literal_backend_json(json.dumps(payload).encode())


def test_derivation_retains_only_chosen_components_and_does_not_modify_source(source):
    raw = source.read_bytes()
    payload = json.loads(raw)
    literal = json.loads(encoding.literal_backend_json(raw))
    for key in ('model', 'pre_tokenizer', 'decoder'):
        assert literal[key] == payload[key]
    assert literal['added_tokens'] == []
    for key in ('normalizer', 'post_processor', 'padding', 'truncation'):
        assert literal[key] is None
    assert source.read_bytes() == raw


@pytest.mark.parametrize('raw', [b'[]', b'null', b'{"model":null,"model":null}',
                                b'{"not_finite":NaN}', b'not-json'])
def test_ambiguous_or_nonobject_json_is_refused(raw):
    with pytest.raises(ValueError):
        encoding.literal_backend_json(raw)


def test_v4_identity_preserves_existing_telemetry_v2_projection(source):
    # Exact shape/string/source-equality invariants read from the live unchanged
    # telemetry_contract.py validate_extensions. Not a complete snapshot test.
    tok = encoding.QwenTokenizer(source)
    identity = encoding.preprocessing_identity(tok)
    assert set(identity) == {'tokenizer', 'serialization', 'literal_encoding'}
    assert all(isinstance(value, str) and len(value) <= 240 for value in identity.values())
    assert identity['tokenizer'] == tok.identity


@pytest.mark.parametrize('field', ['output_encoding', 'literal_backend_sha256'])
def test_each_semantic_component_changes_the_fingerprint_without_wire_shape_change(source, field):
    tok = encoding.QwenTokenizer(source)
    before = encoding.preprocessing_identity(tok)
    # Deliberate mutation simulates a DIFFERENT implementation policy. It is
    # not how production requests are processed.
    setattr(tok, field, 'different-semantics')
    after = encoding.preprocessing_identity(tok)
    assert before['literal_encoding'] != after['literal_encoding']
    assert set(before) == set(after)
    assert before['tokenizer'] == after['tokenizer']


def test_semantic_fingerprint_can_be_recomputed_from_checkpoint_metadata(source):
    tok = encoding.QwenTokenizer(source)
    meta = tok.metadata()
    spec = {'version': meta['literal_encoding'], 'output_encoding': meta['output_encoding'],
            'literal_backend_sha256': meta['literal_backend_sha256']}
    expected = 'literal-bpe-v4:' + hashlib.sha256(encoding.canonical(spec).encode()).hexdigest()
    assert encoding.preprocessing_identity(tok)['literal_encoding'] == expected
