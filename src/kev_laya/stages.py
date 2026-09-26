"""Structured telemetry for isolated calibration, evaluation and serving processes."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import uuid
from .encoding import Limits
from .schema import canonical
from .telemetry import TelemetryWriter, artifact, new_snapshot


def stage_writer(path: Path, phase: str, model, tokenizer, point: dict, limits: Limits, *, manifest=None,
                 experiment_id='local-stages', run_id=None) -> TelemetryWriter:
    if path.exists():
        raise FileExistsError('stage telemetry destination exists; choose a new run path')
    old=point.get('provenance',{})
    from .provenance import capture_source, extensions, ResourceSampler, hardware_identity
    source=capture_source()
    provenance={
        'model':point['model_id'],'backbone':model.cfg.source,'backbone_revision':model.cfg.revision,
        'tokenizer':tokenizer.identity,'config_sha256':old.get('config_sha256',hashlib.sha256(canonical(model.config_dict()).encode()).hexdigest()),
        'data_sha256':old.get('data_sha256'),'split_hashes':old.get('split_hashes',{}),'seed':old.get('seed',0),
        'repository':source['repository'],'commit':source['commit'],'precision':'fp32',
        'hardware':hardware_identity(next(model.parameters()).device),'context_limits':{'branch':limits.branch,'aggregate':limits.aggregate},
        'evidence_class':'pretrained-backbone' if model.native_weights_loaded else 'tiny-synthetic-fixture'}
    if manifest:
        provenance['data_sha256']=hashlib.sha256(canonical(manifest).encode()).hexdigest()
        provenance['split_hashes']={k:v['sha256'] for k,v in manifest['partitions'].items()}
    snapshot=new_snapshot(experiment_id,run_id or uuid.uuid4().hex,uuid.uuid4().hex,provenance, schema_version=2,
        extensions=extensions(tokenizer,source,resources=ResourceSampler(device=next(model.parameters()).device).sample(),
                              calibration=model.calibration_provenance))
    writer=TelemetryWriter(path,snapshot)
    writer.update(phase=phase)
    writer.start_heartbeat()
    return writer
