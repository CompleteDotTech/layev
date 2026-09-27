"""Checkpoint function boundaries with explicit collaborators, NOT model evidence.

Compile the repository's actual save/load functions without importing the model
stack. The fake model and backend isolate metadata-validation ordering only.
The full repository suite and native checkpoint gate remain independently required.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from kev_laya import encoding
from test_literal_v4 import source  # noqa: F401


class ModelDouble:
    def __init__(self, cfg, policy):
        self.cfg = SimpleNamespace(vocab_size=264)
        self.moved = False
    def load_state_dict(self, state, strict):
        assert strict is True
    def to(self, device):
        self.moved = True


def functions(payload, writes=None):
    code = Path(__file__).resolve().parents[1] / 'src/kev_laya/checkpoint.py'
    tree = ast.parse(code.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {'load_checkpoint', 'save_checkpoint'}]
    assert len(selected) == 2
    globals_ = {name: getattr(encoding, name) for name in (
        'QwenTokenizer', 'ByteTokenizer', 'SERIALIZATION', 'LITERAL_ENCODING',
        'preprocessing_identity', 'validate_preprocessing_metadata')}
    globals_.update(Path=Path, hashlib=hashlib, FORMAT='kev-laya-checkpoint/1',
        DecisionEngine=ModelDouble, BackboneConfig=lambda **kw: kw,
        BatchPolicy=SimpleNamespace(from_dict=lambda x: x),
        sha256_file=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        torch=SimpleNamespace(load=lambda *a, **kw: deepcopy(payload)),
        atomic_file=lambda *args: writes.append(args) if writes is not None else None)
    # Function annotations reference only collaborator names supplied above.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(code), 'exec'), globals_)
    return globals_


def payload_for(tok):
    return {'format': 'kev-laya-checkpoint/1', 'config': {}, 'model': {},
            'temperatures': {'choice': 1.0}, 'calibration': {},
            'native_weights_loaded': False, 'training_steps': 0,
            'tokenizer': tok.metadata() | {'path': 'tokenizer.json'},
            'preprocessing': encoding.preprocessing_identity(tok)}


def test_actual_load_function_accepts_complete_v4_metadata(source, tmp_path):
    tok = encoding.QwenTokenizer(source)
    path = tmp_path / 'fixture.pt'; path.write_bytes(b'not-real-weights')
    payload = payload_for(tok)
    model, reloaded, receipt = functions(payload)['load_checkpoint'](path)
    assert model.moved
    assert encoding.preprocessing_identity(reloaded) == payload['preprocessing']
    assert receipt['native_weights_loaded'] is False


@pytest.mark.parametrize('mutation', ['missing', 'null', 'empty', 'output', 'derived', 'metadata'])
def test_actual_load_function_rejects_incomplete_or_changed_identity(source, tmp_path, mutation):
    tok = encoding.QwenTokenizer(source)
    path = tmp_path / 'fixture.pt'; path.write_bytes(b'not-real-weights')
    payload = payload_for(tok)
    if mutation == 'missing': del payload['preprocessing']
    elif mutation == 'null': payload['preprocessing'] = None
    elif mutation == 'empty': payload['preprocessing'] = {}
    elif mutation == 'output': payload['preprocessing']['output_encoding'] = 'literal-output'
    elif mutation == 'derived': payload['preprocessing']['literal_backend_sha256'] = '0' * 64
    elif mutation == 'metadata': del payload['tokenizer']['output_encoding']
    with pytest.raises(ValueError, match='preprocessing|metadata'):
        functions(payload)['load_checkpoint'](path)


@pytest.mark.parametrize('version', ['literal-bpe-v3', 'literal-special-v3'])
def test_actual_load_function_refuses_old_candidate_tags(source, tmp_path, version):
    tok = encoding.QwenTokenizer(source)
    path = tmp_path / 'fixture.pt'; path.write_bytes(b'not-real-weights')
    payload = payload_for(tok)
    payload['tokenizer'].update(literal_encoding=version, escape=version)
    with pytest.raises(ValueError, match='ambiguous unpublished v3'):
        functions(payload)['load_checkpoint'](path)


def test_changed_tokenizer_source_refuses_export_before_any_write(source, tmp_path):
    tok = encoding.QwenTokenizer(source)
    source.write_text('changed-after-initialization')
    target = tmp_path / 'new-checkpoint' / 'fixture.pt'
    writes = []
    with pytest.raises(ValueError, match='source changed since initialization'):
        functions({}, writes)['save_checkpoint'](target, ModelDouble({}, {}), tok)
    assert writes == []
    assert not target.exists()
    assert not (target.parent / 'tokenizer.json').exists()


def test_byte_legacy_omission_still_loads_but_explicit_null_is_rejected(tmp_path):
    tok = encoding.ByteTokenizer()
    path = tmp_path / 'fixture.pt'; path.write_bytes(b'not-real-weights')
    payload = payload_for(tok)
    del payload['preprocessing']
    assert functions(payload)['load_checkpoint'](path)[0].moved
    payload['preprocessing'] = None
    with pytest.raises(ValueError, match='identity mismatch'):
        functions(payload)['load_checkpoint'](path)
