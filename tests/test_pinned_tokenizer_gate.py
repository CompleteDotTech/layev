"""Offline preflight tests, never a claimed native-tokenizer pass."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_pinned_tokenizer.py'
_spec = importlib.util.spec_from_file_location('pinned_tokenizer_gate', SCRIPT)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def write_source(tmp_path):
    raw = b'{"synthetic":"preflight-only"}'
    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / 'tokenizer.json').write_bytes(raw)
    (tmp_path / 'source.json').write_text(json.dumps({
        'revision': gate.PINNED_REVISION, 'sha256': {'tokenizer.json': digest}}))
    return digest


@pytest.mark.parametrize('value', [None, '', 'not-a-hash', 'g' * 64, 'a' * 63, 'a' * 65])
def test_absent_or_invalid_reviewed_hash_blocks(tmp_path, value):
    with pytest.raises(gate.GateBlocked, match='reviewed_tokenizer_sha256'):
        gate.inspect_source(str(tmp_path), value)


def test_missing_directory_blocks():
    with pytest.raises(gate.GateBlocked, match='source_directory_missing'):
        gate.inspect_source(None, 'a' * 64)


def test_missing_files_block(tmp_path):
    with pytest.raises(gate.GateBlocked, match='source_files_missing'):
        gate.inspect_source(str(tmp_path), 'a' * 64)


def test_matching_preflight_does_not_assert_native_success(tmp_path):
    digest = write_source(tmp_path)
    result = gate.inspect_source(str(tmp_path), digest)
    assert result['tokenizer_sha256'] == digest
    assert 'status' not in result and 'native_model' not in result


def test_independent_hash_mismatch_blocks(tmp_path):
    write_source(tmp_path)
    with pytest.raises(gate.GateBlocked, match='reviewed_tokenizer_hash_mismatch'):
        gate.inspect_source(str(tmp_path), 'a' * 64)


def test_mutated_tokenizer_blocks(tmp_path):
    digest = write_source(tmp_path)
    (tmp_path / 'tokenizer.json').write_bytes(b'tampered')
    with pytest.raises(gate.GateBlocked, match='manifest_tokenizer_hash_mismatch'):
        gate.inspect_source(str(tmp_path), digest)


@pytest.mark.parametrize('manifest', [None, [], {}, {'revision': 'main'}])
def test_bad_or_unpinned_manifest_blocks(tmp_path, manifest):
    digest = write_source(tmp_path)
    (tmp_path / 'source.json').write_text(json.dumps(manifest))
    with pytest.raises(gate.GateBlocked, match='source_revision_mismatch'):
        gate.inspect_source(str(tmp_path), digest)


def test_bad_hash_container_blocks(tmp_path):
    digest = write_source(tmp_path)
    (tmp_path / 'source.json').write_text(json.dumps({'revision': gate.PINNED_REVISION, 'sha256': []}))
    with pytest.raises(gate.GateBlocked, match='manifest_tokenizer_hash_mismatch'):
        gate.inspect_source(str(tmp_path), digest)


def test_missing_prerequisite_cli_has_nonzero_receipt(tmp_path):
    run = subprocess.run([sys.executable, str(SCRIPT), '--source-dir', str(tmp_path)],
                         text=True, capture_output=True, timeout=15, check=False)
    assert run.returncode == 2
    receipt = json.loads(run.stdout)
    assert receipt['status'] == 'blocked'
    assert receipt['network_calls'] == receipt['source_writes'] == receipt['literal_cases_executed'] == 0
    assert receipt['native_model'] == receipt['cuda'] == receipt['quality'] == 'not_tested'
    assert receipt['jev_parity'] == 'unknown'
    assert str(tmp_path) not in run.stdout
