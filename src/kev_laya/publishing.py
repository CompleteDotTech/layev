"""Explicit compact publishing, never a default side effect.

W&B publication updates an EXISTING public-API run. The training/monitoring
publisher is not that run's lifecycle owner and never init()/finish()s it.
Trainer phase is in the snapshot; W&B provider state remains independently owned.
No request/response content, code bundle, or checkpoint is uploaded.
"""
from __future__ import annotations
from pathlib import Path
from .telemetry_contract import decode_snapshot, identifier, MAX_SNAPSHOT_BYTES, validate_snapshot


class WandbPublisher:
    def __init__(self, identity: dict, *, enabled=False, sdk=None):
        self.identity, self.enabled, self.sdk, self.run = dict(identity), enabled, sdk, None
        if set(identity) != {'entity', 'project', 'run_id'}:
            raise ValueError('complete W&B identity required')
        for value in identity.values():
            identifier(value)
            if '/' in value:
                raise ValueError('invalid W&B component')

    def publish(self, snapshot: dict) -> bool:
        validate_snapshot(snapshot)
        if snapshot['wandb'] != self.identity:
            raise ValueError('snapshot W&B identity differs from publishing target')
        if not self.enabled:
            return False
        run = self.run
        if run is None:
            sdk = self.sdk
            if sdk is None:
                import wandb as sdk
            api = sdk.Api(timeout=10)
            run = api.run('/'.join(self.identity[k] for k in ('entity', 'project', 'run_id')))
        # A failed lookup/setup must never cache a target that bypasses this
        # policy on retry. Also recheck the available config of a cached run;
        # this is not a remote refresh or an atomic cross-writer guarantee.
        config = dict(run.config)
        if config.get('framework') not in (None, 'kev_laya'):
            raise ValueError('refusing to relabel a different framework W&B run')
        if self.run is None:
            run.config.update({'framework': 'kev_laya', 'telemetry_schema_version': snapshot['schema_version']})
            run.update()
            # Commit the cached handle only after validation and setup succeed.
            # A remote failure can have an uncertain outcome; no rollback or
            # implicit retry is claimed here. The caller owns retry scheduling.
            self.run = run
        run.summary['kev_laya/snapshot'] = snapshot
        run.summary.update()
        return True


def publish_wandb(snapshot_path: Path, *, entity: str, project: str, sdk=None):
    with Path(snapshot_path).open('rb') as stream:
        snapshot = decode_snapshot(stream.read(MAX_SNAPSHOT_BYTES + 1))
    identity = snapshot['wandb']
    if not identity or identity['entity'] != entity or identity['project'] != project:
        raise ValueError('snapshot W&B identity must match the explicit destination')
    return WandbPublisher(identity, enabled=True, sdk=sdk).publish(snapshot)


def publish_s3(snapshot_path: Path, *, bucket: str, key: str, client=None):
    with Path(snapshot_path).open('rb') as stream:
        raw = stream.read(MAX_SNAPSHOT_BYTES + 1)
    decode_snapshot(raw)
    if not bucket or not key or '?' in key or any(ord(c) < 32 for c in bucket + key):
        raise ValueError('invalid explicit object destination')
    if client is None:
        import boto3
        client = boto3.client('s3')
    client.put_object(Bucket=bucket, Key=key, Body=raw, ContentType='application/json', ServerSideEncryption='AES256')
