"""Encoding contract doubles: deliberately NOT real Qwen/native evidence."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unicodedata

import pytest

from kev_laya.encoding import (
    ByteTokenizer, ContextOverflow, LEGACY_ESCAPE, LITERAL_ENCODING,
    Limits, NATIVE_SERIALIZATION, QwenTokenizer, SPECIAL,
    encode_request, preprocessing_identity,
)
from kev_laya.schema import SystemOneRequest, canonical

LEGACY_V2 = "literal-special-v2"


class AddedVocabularyDouble:
    """A UTF-8 contract double with added-token extraction and NFC.

    This is not BPE and does not validate the Rust library or pretrained model.
    It reproduces the relevant independent pipeline rules: special=False added
    tokens are extracted even with encode_special_tokens=True, and NFC is lossy
    for exact decomposed-Unicode round trips.
    """
    instances = []

    def __init__(self, config):
        self.config = config
        self.added = {row['id']: SimpleNamespace(**row) for row in config.get('added_tokens', [])}
        self.encode_special_tokens = False
        self.padding_disabled = False
        self.truncation_disabled = False
        type(self).instances.append(self)

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def from_str(cls, text):
        return cls(json.loads(text))

    def no_padding(self):
        self.padding_disabled = True

    def no_truncation(self):
        self.truncation_disabled = True

    def get_added_tokens_decoder(self):
        return dict(self.added)

    def get_vocab_size(self):
        return 256 + len(self.added)

    def token_to_id(self, text):
        return next((i for i, token in self.added.items() if token.content == text), None)

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        if self.config.get('normalizer'):
            text = unicodedata.normalize('NFC', text)
        candidates = [(i, t) for i, t in self.added.items()
                      if not (t.special and self.encode_special_tokens)]
        output = []
        pos = 0
        while pos < len(text):
            matches = [(i, t) for i, t in candidates if text.startswith(t.content, pos)]
            if matches:
                i, token = max(matches, key=lambda pair: len(pair[1].content))
                output.append(i)
                pos += len(token.content)
            else:
                output.extend(text[pos].encode('utf-8'))
                pos += 1
        if not self.truncation_disabled and self.config.get('truncation'):
            output = output[:self.config['truncation']['max_length']]
        if not self.padding_disabled and self.config.get('padding'):
            output += [0] * max(0, self.config['padding']['length'] - len(output))
        return SimpleNamespace(ids=output)

    def decode(self, ids, *, skip_special_tokens=False):
        assert not skip_special_tokens
        chunks, pending = [], []
        for i in ids:
            if i in self.added:
                chunks.append(bytes(pending).decode('utf-8'))
                pending = []
                chunks.append(self.added[i].content)
            else:
                pending.append(i)
        chunks.append(bytes(pending).decode('utf-8'))
        return ''.join(chunks)


def config(*, all_special=False, normalizer=True):
    spellings = (*SPECIAL, '<|im_start|>', '<|im_end|>', '<|extra_control|>')
    return {
        'model': {'type': 'BPE', 'contract_double': True},
        'pre_tokenizer': {'type': 'ByteLevel'},
        'decoder': {'type': 'ByteLevel'},
        'added_tokens': [{'id': 256 + i, 'content': text,
                          'special': all_special or i % 2 == 1}
                         for i, text in enumerate(spellings)],
        'normalizer': {'type': 'NFC'} if normalizer else None,
        'truncation': {'max_length': 1}, 'padding': {'length': 100},
    }


@pytest.fixture
def source(tmp_path, monkeypatch):
    AddedVocabularyDouble.instances = []
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(Tokenizer=AddedVocabularyDouble))
    path = tmp_path / 'tokenizer.json'
    path.write_text(json.dumps(config()))
    return path


LITERALS = [*SPECIAL, '<|im_start|>', '<|extra_control|>', '<¦fim_prefix¦>',
            'prefix <|fim_prefix|> suffix', '<|fim_prefix|><|fim_prefix|>',
            '\\<|fim_prefix|>\\', r'\u003c|fim_prefix|>', '\x00', '\r\n\t',
            'é', 'e\u0301', '中文🙂', '가', '𝓽𝓮𝔁𝔱', '\u2028\u2029', '',
            'A' * 257, json.dumps({'nested': ['<|fim_prefix|>', 'e\u0301', {'x': '\\'}]})]


@pytest.mark.parametrize('text', LITERALS)
def test_default_literal_preserves_exact_input(source, text):
    tok = QwenTokenizer(source)
    ids = tok.encode(text)
    assert tok.decode_literal(ids) == text
    assert set(ids).isdisjoint(tok._special_ids)


def test_v4_preserves_original_source_and_output_vocabulary(source):
    before = source.read_bytes()
    tok = QwenTokenizer(source)
    assert tok.metadata()['literal_encoding'] == 'literal-bpe-v4'
    assert tok.metadata()['serialization'] == NATIVE_SERIALIZATION
    assert tok.metadata()['vocab_size'] == 264
    assert tok.special == tuple(range(256, 261))
    assert tok.identity == 'qwen-tokenizer-sha256:' + hashlib.sha256(before).hexdigest()
    assert tok.encode_output(SPECIAL[0]) == [256]
    assert tok.encode_output('e\u0301') == list('é'.encode())
    assert tok.encode('e\u0301') != tok.encode('é')
    assert tok._tokenizer is not tok._output_tokenizer
    assert not tok._tokenizer.get_added_tokens_decoder()
    assert tok._output_tokenizer.config['normalizer'] == {'type': 'NFC'}
    assert tok._tokenizer.config['normalizer'] is None
    for name in ('model', 'pre_tokenizer', 'decoder'):
        assert tok._tokenizer.config[name] == tok._output_tokenizer.config[name]
    assert source.read_bytes() == before
    assert all(t.padding_disabled and t.truncation_disabled for t in AddedVocabularyDouble.instances)


def test_v2_regression_remains_explicit_and_is_not_auto_upgraded(source):
    with pytest.raises(ValueError, match='control token'):
        QwenTokenizer(source, literal_encoding=LEGACY_V2)


def test_supported_v2_stays_v2_and_retains_unicode_refusal(source):
    source.write_text(json.dumps(config(all_special=True)))
    tok = QwenTokenizer(source, literal_encoding=LEGACY_V2)
    assert tok.literal_encoding == LEGACY_V2
    assert tok.decode_literal(tok.encode(SPECIAL[0])) == SPECIAL[0]
    with pytest.raises(ValueError, match='not reversible'):
        tok.encode('e\u0301')
    assert tok.encode_output(SPECIAL[0]) == list(SPECIAL[0].encode())


def test_legacy_v1_spelling_rule_remains_distinct(source):
    old = QwenTokenizer(source, literal_encoding=LEGACY_ESCAPE)
    new = QwenTokenizer(source)
    assert old.encode(SPECIAL[0]) == old.encode('<¦fim_prefix¦>')
    assert new.encode(SPECIAL[0]) != new.encode('<¦fim_prefix¦>')
    assert preprocessing_identity(old) != preprocessing_identity(new)


@pytest.mark.parametrize('mode', ['', 'literal-bpe-v99', None, 'literal-special-v1'])
def test_unknown_versions_are_refused(source, mode):
    with pytest.raises(ValueError, match='version'):
        QwenTokenizer(source, literal_encoding=mode)


def test_decoder_failure_has_no_lossy_fallback(source, monkeypatch):
    tok = QwenTokenizer(source)
    monkeypatch.setattr(tok._tokenizer, 'decode', lambda *a, **kw: 'changed')
    with pytest.raises(ValueError, match='not reversible'):
        tok.encode('original')


def test_control_id_is_rejected_even_if_decoder_roundtrip_matches(source, monkeypatch):
    tok = QwenTokenizer(source)
    monkeypatch.setattr(tok._tokenizer, 'encode', lambda *a, **kw: SimpleNamespace(ids=[263]))
    monkeypatch.setattr(tok._tokenizer, 'decode', lambda *a, **kw: 'same')
    with pytest.raises(ValueError, match='control token'):
        tok.encode('same')


def test_missing_control_delimiter_is_refused(source):
    original = config()
    original['added_tokens'] = original['added_tokens'][1:]
    source.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='distinct required delimiter'):
        QwenTokenizer(source)


def test_request_mixed_structures_budgets_and_metadata_only_ids(source):
    tok = QwenTokenizer(source)
    req = SystemOneRequest(state={'nested': ['e\u0301', SPECIAL[0], {'literal': '{}'}]},
        model='contract-test', questions={
            'id-not-content': {'type': 'choice', 'instructions': {'x': [SPECIAL[1], 'e\u0301']},
                               'criteria': {SPECIAL[2]: ['🙂'], 'second': None}},
            'score': {'type': 'score', 'instructions': ['level'], 'criteria': ['low', {'x': 'high'}]},
            'noul': {'type': 'noul', 'instructions': 'e\u0301', 'criteria': {'true': 'yes', 'false': 'no'}},
        })
    before = req.model_dump()
    enc = encode_request(req, tok, Limits(4096, 8192))
    assert json.loads(tok.decode_literal(list(enc.state[1:]))) == {'state': req.state}
    assert enc.logical_tokens == len(enc.state) + sum(len(b.ids) for b in enc.branches)
    for branch in enc.branches:
        assert branch.ids[0] == tok.special[1] and branch.ids[-1] == tok.special[4]
        assert sum(i in tok._special_ids for i in branch.ids) == 2 + 2 * len(branch.question.options())
    renamed = req.model_copy(update={'questions': {f'renamed-{i}': q for i, q in enumerate(req.questions.values())}})
    other = encode_request(renamed, tok, Limits(4096, 8192))
    assert enc.state == other.state
    assert [b.ids for b in enc.branches] == [b.ids for b in other.branches]
    branch_limit = len(enc.state) + max(len(b.ids) for b in enc.branches)
    assert encode_request(req, tok, Limits(branch_limit, enc.logical_tokens)) == enc
    with pytest.raises(ContextOverflow) as exc:
        encode_request(req, tok, Limits(branch_limit - 1, enc.logical_tokens))
    assert exc.value.detail['limit'] == 'state_plus_branch'
    with pytest.raises(ContextOverflow) as exc:
        encode_request(req, tok, Limits(branch_limit, enc.logical_tokens - 1))
    assert exc.value.detail['limit'] == 'aggregate_input'
    assert req.model_dump() == before
    text = req.model_copy(update={'state': '{}'})
    obj = req.model_copy(update={'state': {}})
    assert encode_request(text, tok, Limits(4096, 8192)).state != encode_request(obj, tok, Limits(4096, 8192)).state


def test_tokenizer_metadata_reload_is_path_independent(source, tmp_path):
    original = QwenTokenizer(source)
    metadata = original.metadata()
    copied = tmp_path / 'checkpoint-local' / 'tokenizer.json'
    copied.parent.mkdir()
    copied.write_bytes(source.read_bytes())
    reloaded = QwenTokenizer(copied, literal_encoding=metadata['literal_encoding'])
    assert preprocessing_identity(original) == preprocessing_identity(reloaded)
    assert original.encode(canonical({'x': LITERALS})) == reloaded.encode(canonical({'x': LITERALS}))


def test_shared_literal_backend_has_no_per_call_mutations(source):
    tok = QwenTokenizer(source)
    expected = [tok.encode(t) for t in LITERALS]
    configs = [json.dumps(t.config, sort_keys=True) for t in AddedVocabularyDouble.instances]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(tok.encode, LITERALS * 3))
    assert actual == expected * 3
    assert configs == [json.dumps(t.config, sort_keys=True) for t in AddedVocabularyDouble.instances]


def test_byte_fixture_identity_unchanged():
    assert preprocessing_identity(ByteTokenizer()) == {
        'tokenizer': 'utf8-byte-fixture-v1', 'serialization': 'kev-laya-prefix-v1',
        'literal_encoding': 'utf8-bytes-v1'}
