"""Extra boundary regressions; optional REAL tokenizer tests are separately skipped."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from kev_laya.exposure import verify_exposure
from kev_laya.lineage import begin_attempt
from kev_laya.telemetry_contract import attempt_digest
from kev_laya.provenance import ResourceSampler
from kev_laya.training import train, TrainSettings
from kev_laya.objectives import ObjectiveConfig
from kev_laya.encoding import ByteTokenizer, Limits, QwenTokenizer, SPECIAL
from test_stage3_defects import snapshot, collection, write_registry, S3Double, WbPagesDouble
from overwatch.kev_laya_adapter import merge_snapshots, presentation
from overwatch.providers.kev_laya import collect_snapshots, CollectionLimits


def load_installer():
    path = Path(__file__).resolve().parents[1]/'integrations/overwatch/apply.py'
    sp=importlib.util.spec_from_file_location('stage3_installer',path)
    m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def test_overwatch_upgrade_reverse_and_idempotence(tmp_path,monkeypatch):
    m=load_installer();name='src/overwatch/constants.py'
    base=b'# exact anchor fixture, not the original application\n'
    path=tmp_path/name;path.parent.mkdir(parents=True);path.write_bytes(base)
    monkeypatch.setattr(m,'BLOBS',{name:m.blob_sha(base)})
    # Simulate the already-applied, known prior delta on this explicit fixture.
    old=m.apply_edits(base.decode(),m.LEGACY[name]);path.write_text(old)
    changes=m.prepare(tmp_path,verify_revision=False)
    for target,raw in changes.items():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    assert m.prepare(tmp_path,verify_revision=False)=={}
    assert path.read_text()==m.transform(name,base.decode())
    path.write_text(path.read_text()+'# unrelated user change\n')
    with pytest.raises(ValueError,match='modified or unexpected'):m.prepare(tmp_path,verify_revision=False)


def test_overwatch_revision_checked_operations_reverse_all_targets():
    m=load_installer()
    # Pure transformation verification. This is NOT native Overwatch execution.
    for name in m.BLOBS:
        # Include each original anchor needed by both stages. The old new-string
        # anchors introduced by earlier substitutions must not appear twice.
        legacy=m.LEGACY[name]
        seed='\n'.join(a for a,b in legacy if a not in {'<PREFIX>','<SUFFIX>'})
        for a,b in m.edits(name)[len(legacy):]:
            if a.startswith('<'):continue
            if a not in seed and not any(a in old_new for _,old_new in legacy):
                # Anchors may be introduced by a previous added edit.
                if not any(a in v for _,v in m.edits(name)[:m.edits(name).index((a,b))]):seed+='\n'+a
        changed=m.transform(name,seed)
        assert m.apply_edits(changed,m.edits(name),reverse=True)==seed
        assert m.apply_edits(m.apply_edits(seed,legacy),legacy,reverse=True)==seed


def test_overwatch_review_and_symlink_targets_refused(tmp_path):
    m=load_installer()
    with pytest.raises(ValueError,match='review'):m.prepare(tmp_path/'reviews'/'snapshot',verify_revision=False)
    target=tmp_path/'target';target.mkdir();link=tmp_path/'linked'
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if os.name == 'nt' and getattr(exc, 'winerror', None) == 1314:
            pytest.skip('Windows symlink creation privilege is unavailable')
        raise
    with pytest.raises(ValueError,match='symlink'):m.prepare(link,verify_revision=False)


def test_missed_many_attempts_window_bound_is_explicit(tmp_path):
    state={'experiment_id':'e','run_id':'r','attempt_id':'a0','attempt_index':0,'parent_attempt_id':None}
    begin_attempt(tmp_path,state,resumed=False)
    original=merge_snapshots(None,collection(snapshot(state,v2=True)))
    for _ in range(20):begin_attempt(tmp_path,state,resumed=True)
    recovered=merge_snapshots(json.loads(json.dumps(original)),collection(snapshot(state,v2=True)))
    assert recovered['records'][0]['snapshot']['attempt_index']==20
    for _ in range(70):begin_attempt(tmp_path,state,resumed=True)
    assert len(state['attempt_lineage'])==64
    missing=merge_snapshots(recovered,collection(snapshot(state,v2=True)))
    assert missing['records'][0]['snapshot']['attempt_index']==20
    view=presentation(missing)['runs'][0]
    assert view['verification']=='last_verified_with_newer_uncertainty'
    assert view['unverified_observations'][0]['attempt_index']==90


def test_rehashed_conflicting_parent_receipt_not_accepted(tmp_path):
    state={'experiment_id':'e','run_id':'r','attempt_id':'a0','attempt_index':0,'parent_attempt_id':None}
    begin_attempt(tmp_path,state,resumed=False)
    old=merge_snapshots(None,collection(snapshot(state,v2=True)))
    begin_attempt(tmp_path,state,resumed=True);new=snapshot(state,v2=True)
    rows=new['extensions']['attempt_lineage']
    rows[0]['attempt_id']='someone-else';rows[0]['sha256']=attempt_digest(rows[0])
    rows[1]['previous_sha256']=rows[0]['sha256'];rows[1]['parent_attempt_id']='someone-else'
    rows[1]['sha256']=attempt_digest(rows[1]);new['parent_attempt_id']='someone-else'
    out=merge_snapshots(old,collection(new))
    assert out['records'][0]['snapshot']['attempt_index']==0 and out['observations']


def test_budget_skipped_source_keeps_last_success_timestamp(tmp_path):
    export=tmp_path/'export.json';export.write_text(json.dumps(snapshot()))
    reg=tmp_path/'reg.json';write_registry(reg,[{'id':'s','transport':'local','path':str(export)}])
    first=merge_snapshots(None,collect_snapshots(reg))
    stamp=first['collection_sources'][0]['last_successful_refresh_at']
    export.unlink()
    after=merge_snapshots(json.loads(json.dumps(first)),collect_snapshots(reg,limits=CollectionLimits(seconds=0)))
    assert after['collection_sources'][0]['last_successful_refresh_at']==stamp
    assert after['collection_sources'][0]['refreshed_at'] is None
    assert after['records'][0]['received_at']==first['records'][0]['received_at']


def test_mixed_transports_share_a_single_pre_read_payload_budget(tmp_path):
    raw=json.dumps(snapshot()).encode()
    local=tmp_path/'local.json';local.write_bytes(raw)
    reg=tmp_path/'reg.json';write_registry(reg,[
        {'id':'local','transport':'local','path':str(local)},
        {'id':'s3','transport':'s3','bucket':'b','key':'k'},
        {'id':'wb','transport':'wandb','entity':'e','project':'p','limit':10}])
    s3=S3Double(raw);wb=WbPagesDouble(snapshot())
    out=collect_snapshots(reg,s3_client=s3,wandb_api=wb,limits=CollectionLimits(payload_bytes=2*len(raw)))
    assert out['accounting']['payload_bytes']==2*len(raw)
    assert s3.fetches==1 and s3.bytes==len(raw) and wb.calls==0
    assert out['sources'][2]['status']=='skipped_payload_budget'


def test_resource_sampling_is_measured_and_interval_bounded(monkeypatch):
    now=[0.];sampler=ResourceSampler(interval=2,clock=lambda:now[0])
    first=sampler.sample();second=sampler.sample()
    assert first==second and first['units']=='bytes'
    assert first['rss_bytes'] is None or first['rss_bytes']>0
    assert first['gpu_allocated_bytes'] is None
    now[0]=3.;third=sampler.sample()
    assert third['sampled_at']>=first['sampled_at']
    assert third['interval_seconds']==2


def test_monitoring_ledger_write_failure_does_not_abort_training(tmp_path,monkeypatch,tiny,suite):
    from kev_laya import lineage
    def broken(*args,**kwargs):raise OSError('simulated monitoring filesystem failure')
    monkeypatch.setattr(lineage,'atomic_json',broken)
    data,manifest=suite
    result=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',
       TrainSettings(steps=2,save_every=1),ObjectiveConfig(),Limits(512,8192))
    point=torch.load(result['checkpoint'],weights_only=True)
    assert point['training_state']['step']==2
    export=json.loads(Path(result['snapshot']).read_text())
    assert export['monitoring_export_failures']>=1
    assert export['extensions']['lineage_durable'] is False
    assert verify_exposure(Path(result['checkpoint']))['status']=='verified'


def test_unreadable_exposure_is_unknown_not_a_crash(tmp_path):
    path=tmp_path/'bad.pt';path.write_bytes(b'not a torch archive')
    assert verify_exposure(path)['status']=='unknown'


def test_real_rust_bytelevel_tokenizer_round_trip(tmp_path):
    """Actual tokenizers implementation, not the recording double (optional dependency)."""
    tokenizers=pytest.importorskip('tokenizers',reason='real Rust tokenizer library unavailable; contract double is not native evidence')
    from tokenizers import models, pre_tokenizers, decoders, trainers
    backend=tokenizers.Tokenizer(models.BPE())
    backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=400,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=list(SPECIAL))
    examples=['<|fim_prefix|>','<¦fim_prefix¦>',r'\\<|fim_prefix|>',r'\u003c|fim_prefix|>','é e\u0301 中文🙂\n\x00', '<|im_start|>']
    backend.train_from_iterator(examples,trainer=trainer)
    file=tmp_path/'tokenizer.json';backend.save(str(file))
    tok=QwenTokenizer(file)
    for text in examples:
        ids=tok.encode(text)
        assert not set(ids)&set(tok.special) and tok.decode_literal(ids)==text
    assert tok.encode(examples[0])!=tok.encode(examples[1])


@pytest.mark.native
def test_real_pinned_qwen_tokenizer_literals():
    pytest.importorskip('tokenizers',reason='native tokenizer runtime unavailable')
    directory=os.environ.get('KEV_LAYA_QWEN_DIR')
    if not directory:pytest.skip('set KEV_LAYA_QWEN_DIR to verified pinned Qwen snapshot')
    root=Path(directory)
    manifest=json.loads((root/'source.json').read_text())
    assert manifest['revision']=='060db6499f32faf8b98477b0a26969ef7d8b9987'
    assert manifest['sha256']['tokenizer.json']==hashlib.sha256((root/'tokenizer.json').read_bytes()).hexdigest()
    tok=QwenTokenizer(root/'tokenizer.json')
    for text in ['<|fim_prefix|>','<¦fim_prefix¦>','<|box_start|>','e\u0301\n中文🙂',r'\\\\<|fim_prefix|>',json.dumps({'nested':['<|fim_prefix|>',r'\u003c']})]:
        assert tok.decode_literal(tok.encode(text))==text


def test_bad_cached_collection_and_observation_shapes_warn_not_crash():
    s=snapshot();base=merge_snapshots(None,collection(s))
    base['collection_sources']=None
    base['observations']=[{'snapshot':s}]
    after=merge_snapshots(base,collection(s))
    assert after['records'] and after['warnings']
    view=presentation(base)
    assert len(view['runs'])==1 and view['warnings']
    base['records']=None
    assert presentation(base)['runs']==[]


def test_collection_completeness_not_confused_with_valid_registry(tmp_path):
    raw=tmp_path/'raw.json';raw.write_text(json.dumps(snapshot()))
    reg=tmp_path/'registry.json';write_registry(reg,[{'id':'s','transport':'local','path':str(raw)}])
    assert collect_snapshots(reg)['complete'] is True
    skipped=collect_snapshots(reg,limits=CollectionLimits(seconds=0))
    assert skipped['registry_complete'] is True and skipped['complete'] is False


def test_explicit_resume_experiment_conflict_rejected_without_export_write(tmp_path,tiny,suite):
    data,manifest=suite;settings=TrainSettings(steps=2,save_every=1)
    result=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',settings,
        ObjectiveConfig(),Limits(512,8192),experiment_id='original',stop_after=1)
    before=Path(result['snapshot']).read_bytes()
    with pytest.raises(ValueError,match='experiment identity'):
        train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',settings,
            ObjectiveConfig(),Limits(512,8192),resume=Path(result['checkpoint']),experiment_id='changed')
    assert Path(result['snapshot']).read_bytes()==before


def test_initial_resumed_export_has_restored_progress(tmp_path,tiny,suite,monkeypatch):
    from kev_laya.telemetry import TelemetryWriter
    data,manifest=suite;settings=TrainSettings(steps=3,save_every=1)
    result=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',settings,
        ObjectiveConfig(),Limits(512,8192),stop_after=1)
    captured=[];original=TelemetryWriter.export
    def capture(self):
        captured.append(copy.deepcopy(self.snapshot))
        return original(self)
    monkeypatch.setattr(TelemetryWriter,'export',capture)
    train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',settings,
        ObjectiveConfig(),Limits(512,8192),resume=Path(result['checkpoint']),stop_after=2)
    assert captured[0]['attempt_index']==1 and captured[0]['progress']['optimizer_steps']==1
    assert captured[0]['progress']['forward_tokens']==result['state']['forward_tokens']
    assert captured[0]['extensions']['execution']['prefix_passes']==result['state']['microbatches']


def test_integration_previous_version_manifest_keys_are_portable():
    from pathlib import PureWindowsPath, PurePosixPath
    m=load_installer()
    for cls in (PureWindowsPath,PurePosixPath):
        root=cls('delivery/added');file=root/'src/overwatch/kev_laya_adapter.py'
        assert m.relative_path_key(file,root)=='src/overwatch/kev_laya_adapter.py'


def test_raw_cache_reader_unhashable_version_is_warning(tmp_path):
    m=load_installer()
    # Execute the actual added cache-read function, not a replacement application.
    source=next(new for old,new in m.edits('src/overwatch/raw_cache.py') if old=='<SUFFIX>' and 'def read_model_runs_cache' in new)
    path=tmp_path/'runs-v2.json';path.write_text('{}')
    namespace={'model_runs_cache_path':lambda:path,'raw_cache_path':lambda *args:path,
               'read_json':lambda p:{'schema_version':[]}}
    exec(source,namespace)
    result=namespace['read_model_runs_cache']()
    assert not result['records'] and result['warnings'][0]['code']=='model_run_cache_unreadable_or_incompatible'
