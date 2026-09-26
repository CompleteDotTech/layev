"""Durable bounded attempt receipts; independent of optimizer checkpoint frequency."""
from __future__ import annotations
import copy
from pathlib import Path
import uuid
from .io import atomic_json
from .schema import strict_loads
from .telemetry_contract import MAX_ATTEMPTS, attempt_digest, validate_attempt_records


def begin_attempt(output: Path, state: dict, *, resumed: bool) -> list[dict]:
    """Write the new receipt before exporting its snapshot.

    The last 64 consecutive receipts accompany every v2 snapshot. This bridges
    missed polls without guessing an index from the most recent optimizer save.
    Beyond the retained window collectors expose an unverified observation.
    """
    path = Path(output) / 'attempt-lineage.json'
    records = copy.deepcopy(state.get('attempt_lineage', [])) if resumed else []
    validate_attempt_records(records, state['experiment_id'], state['run_id'])
    if records and (records[-1]['attempt_id'] != state['attempt_id'] or
                    records[-1]['attempt_index'] != state['attempt_index']):
        raise ValueError('checkpoint attempt identity disagrees with its receipts')
    read_failed = False
    raw = None
    if path.exists():
        try:
            with path.open('rb') as f:
                raw = f.read(65537)
        except OSError:
            read_failed = True
    if raw is not None:
        if len(raw) > 65536:
            raise ValueError('attempt receipt file exceeds bound')
        saved = strict_loads(raw)
        if set(saved) != {'format', 'records'} or saved['format'] != 'kev-laya-attempts/1':
            raise ValueError('unsupported attempt receipts')
        validate_attempt_records(saved['records'], state['experiment_id'], state['run_id'])
        if not resumed:
            raise ValueError('existing run receipts require resume')
        durable = saved['records']
        anchors = {r['attempt_index']: r for r in records}
        for receipt in durable:
            anchor = anchors.get(receipt['attempt_index'])
            if anchor is not None and receipt != anchor:
                raise ValueError('durable receipt conflicts with checkpoint lineage')
            if receipt['attempt_index'] == state['attempt_index'] and (
                receipt['attempt_id'] != state['attempt_id'] or
                receipt['parent_attempt_id'] != state['parent_attempt_id']):
                raise ValueError('durable receipt conflicts with checkpoint attempt identity')
        # A checkpoint can be ahead of a failed ledger write. Prefer whichever
        # verified local chain is newer, never roll the attempt number backwards.
        if durable and (not records or durable[-1]['attempt_index'] >= records[-1]['attempt_index']):
            records = durable
    if not records and resumed:
        # An older checkpoint provides one known anchor, not invented intervening attempts.
        anchor = {'experiment_id': state['experiment_id'], 'run_id': state['run_id'],
                  'attempt_id': state['attempt_id'], 'attempt_index': state['attempt_index'],
                  'parent_attempt_id': state['parent_attempt_id'], 'previous_sha256': None}
        if anchor['attempt_index'] != 0:
            # No receipt hash for the missing predecessor: retain no fabricated chain.
            records = []
        else:
            anchor['sha256'] = attempt_digest(anchor)
            records = [anchor]
    previous = records[-1] if records else None
    if resumed:
        index = (previous['attempt_index'] if previous else state['attempt_index']) + 1
        parent_id = previous['attempt_id'] if previous else state['attempt_id']
        state.update(attempt_index=index, parent_attempt_id=parent_id, attempt_id=uuid.uuid4().hex)
    record = {'experiment_id': state['experiment_id'], 'run_id': state['run_id'],
              'attempt_id': state['attempt_id'], 'attempt_index': state['attempt_index'],
              'parent_attempt_id': state['parent_attempt_id'],
              'previous_sha256': previous['sha256'] if previous else None}
    record['sha256'] = attempt_digest(record)
    # A legacy non-root anchor without its predecessor is intentionally unproven.
    if record['attempt_index'] > 0 and record['previous_sha256'] is None:
        records = []
    else:
        records = (records + [record])[-MAX_ATTEMPTS:]
        validate_attempt_records(records, state['experiment_id'], state['run_id'])
    state['attempt_lineage'] = records
    try:
        atomic_json(path, {'format': 'kev-laya-attempts/1', 'records': records})
        state['lineage_durable'] = not read_failed
    except OSError:
        # The new receipt remains in the optimizer checkpoint and export, but a
        # crash before either is durable can leave a later attempt unverified.
        # Monitoring storage failure must not abort model optimization.
        state['lineage_durable'] = False
    return records
