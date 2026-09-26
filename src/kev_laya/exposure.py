"""Immutable observation provenance, separate from resumable optimizer state.

These records attest what this trusted local trainer observed, not remotely signed
proof of training. Checksums detect artifact substitution/corruption; an attacker
who can rewrite every checkpoint and all expected hashes is outside this trust model.
"""
from __future__ import annotations
import copy
import pickle
from pathlib import Path
import torch
from .encoding import preprocessing_identity
from .io import sha256_file

FORMAT = 'kev-laya-training-exposure/1'
FIELDS = ('maximum_branch_tokens', 'maximum_aggregate_tokens', 'examples',
          'branch_32768_examples', 'aggregate_65536_examples', 'optimizer_steps')


def start_exposure(model, tokenizer) -> dict:
    identity = preprocessing_identity(tokenizer) | {'backbone': model.cfg.source, 'backbone_revision': model.cfg.revision}
    return {'format': FORMAT, 'identity': identity,
            'evidence_class': 'pretrained-backbone' if model.native_weights_loaded else 'tiny-synthetic-fixture',
            'observed': dict.fromkeys(FIELDS, 0), 'operation': 'training', 'parent': None}


def observe(exposure: dict, encoded) -> None:
    longest = len(encoded.state) + max(len(b.ids) for b in encoded.branches)
    counts = exposure['observed']
    counts['maximum_branch_tokens'] = max(counts['maximum_branch_tokens'], longest)
    counts['maximum_aggregate_tokens'] = max(counts['maximum_aggregate_tokens'], encoded.logical_tokens)
    counts['examples'] += 1
    counts['branch_32768_examples'] += int(longest >= 32768)
    counts['aggregate_65536_examples'] += int(encoded.logical_tokens >= 65536)


def snapshot_exposure(model, tokenizer, *, training: bool, parent_checkpoint: Path | None = None,
                      operation='export', parent_sha256=None) -> dict | None:
    existing = getattr(model, 'training_exposure', None)
    if existing is None:
        return None  # Legacy/unknown counters are never manufactured from configuration.
    out = copy.deepcopy(existing)
    expected = preprocessing_identity(tokenizer)
    if any(out['identity'].get(k) != v for k, v in expected.items()):
        raise ValueError('training exposure preprocessing mismatch')
    if training:
        out['operation'] = 'training'
        return out
    out['operation'] = operation
    if parent_checkpoint is None:
        out['parent'] = None  # Missing derivation proof is explicit and fails verification.
    else:
        parent_checkpoint = Path(parent_checkpoint)
        digest = sha256_file(parent_checkpoint)
        if digest != parent_sha256:
            raise ValueError('training exposure parent checksum mismatch')
        out['parent'] = {'file': parent_checkpoint.name, 'sha256': digest}
    return out


def verify_exposure(checkpoint: Path, *, parents: list[Path] = (), expected_sha256=None) -> dict:
    """Verify lineage in checkpoint-local files or explicitly supplied parent files.

    Never follows arbitrary checkpoint-provided filesystem paths or remote URIs.
    Calibrated/exported checkpoints must carry exactly their parent's exposure.
    Training roots must carry actual trainer observation records, not config limits.
    """
    from .encoding import SERIALIZATION
    explicit = {sha256_file(Path(p)): Path(p) for p in parents if Path(p).is_file()}
    visited = set()
    chain = []

    def unknown(reason):
        return {'status': 'unknown', 'reason': reason, 'native_32k_64k': False, 'chain': list(chain)}

    def read(path: Path, expected=None):
        if len(visited) >= 64:
            return unknown('derivation depth limit')
        if not path.is_file():
            return unknown('missing parent checkpoint')
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            return unknown('checkpoint or parent hash mismatch')
        if digest in visited:
            return unknown('derivation cycle')
        visited.add(digest)
        manifest_file = path.with_suffix('.manifest.json')
        if manifest_file.is_file():
            from .schema import strict_loads
            manifest = strict_loads(manifest_file.read_bytes())
            if manifest.get('sha256') != digest:
                return unknown('checkpoint manifest hash mismatch')
        point = torch.load(path, map_location='cpu', weights_only=True)
        exposure = point.get('training_exposure')
        if not isinstance(exposure, dict) or exposure.get('format') != FORMAT:
            return unknown('legacy or missing immutable training-exposure evidence')
        meta = point.get('tokenizer', {})
        identity = point.get('preprocessing', {'tokenizer': meta.get('identity'),
                    'serialization': meta.get('serialization', SERIALIZATION),
                    'literal_encoding': meta.get('literal_encoding', meta.get('escape', 'utf8-bytes-v1'))})
        if any(exposure.get('identity', {}).get(k) != v for k, v in identity.items()):
            return unknown('tokenizer/serialization mismatch')
        cfg = point.get('config', {})
        if exposure['identity'].get('backbone') != cfg.get('source') or exposure['identity'].get('backbone_revision') != cfg.get('revision'):
            return unknown('backbone identity mismatch')
        counts = exposure.get('observed', {})
        if set(counts) != set(FIELDS) or any(type(v) is not int or v < 0 for v in counts.values()):
            return unknown('invalid exposure counters')
        if any(counts[k] > counts['examples'] for k in ('branch_32768_examples', 'aggregate_65536_examples')):
            return unknown('inconsistent exposure counts')
        chain.append({'sha256': digest, 'operation': exposure.get('operation')})
        op = exposure.get('operation')
        if op in {'calibration', 'export'}:
            parent = exposure.get('parent')
            if not isinstance(parent, dict) or parent.get('sha256') != point.get('parent_sha256'):
                return unknown('missing or inconsistent derivation parent')
            name = parent.get('file')
            if not isinstance(name, str) or Path(name).name != name:
                return unknown('unsafe parent filename')
            target = explicit.get(parent['sha256'], path.parent / name)
            result = read(target, parent['sha256'])
            if result['status'] != 'verified':
                return result
            if counts != result['observed'] or exposure['identity'] != result['identity'] or exposure['evidence_class'] != result['evidence_class']:
                return unknown('derived artifact changed parent exposure')
            result['chain'] = list(chain)
            return result
        if op != 'training':
            return unknown('unknown exposure operation')
        # Only the trainer supplies this root; inference/calibration does not gain
        # exposure by copying optimizer counters or declared context limits.
        state = point.get('training_state')
        if not isinstance(state, dict) or state.get('exposure_observed') != counts:
            return unknown('training root lacks matching observed optimization-boundary record')
        if counts['optimizer_steps'] == 0 or counts['examples'] == 0:
            return unknown('no observed optimization')
        base = point.get('backbone_source') or {}
        required_files = {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'LICENSE'}
        hashes = base.get('sha256', {})
        native_source = (base.get('repository') == 'Qwen/Qwen2.5-0.5B'
                         and base.get('revision') == '060db6499f32faf8b98477b0a26969ef7d8b9987'
                         and isinstance(hashes, dict) and set(hashes) == required_files
                         and all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in hashes.values())
                         and hashes.get('model.safetensors') == point.get('loaded_backbone_sha256')
                         and meta.get('identity') == 'qwen-tokenizer-sha256:' + str(hashes.get('tokenizer.json')))
        native = (native_source and point.get('native_weights_loaded') is True and meta.get('kind') == 'qwen'
                  and exposure['evidence_class'] == 'pretrained-backbone'
                  and cfg.get('source') == 'Qwen/Qwen2.5-0.5B'
                  and cfg.get('revision') == '060db6499f32faf8b98477b0a26969ef7d8b9987')
        return {'status': 'verified', 'identity': exposure['identity'], 'observed': dict(counts),
                'evidence_class': exposure['evidence_class'], 'chain': list(chain),
                'native_32k_64k': bool(native and counts['maximum_branch_tokens'] >= 32768
                                     and counts['maximum_aggregate_tokens'] >= 65536
                                     and counts['branch_32768_examples'] > 0 and counts['aggregate_65536_examples'] > 0)}
    try:
        return read(Path(checkpoint), expected_sha256)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, AttributeError, pickle.UnpicklingError):
        return unknown('unreadable or malformed checkpoint evidence')
