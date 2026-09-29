import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import builtins
import importlib.util
import io
import json
from types import SimpleNamespace
import pytest
from kev_laya.telemetry import new_snapshot, TelemetryWriter
from kev_laya.telemetry_contract import validate_snapshot, decode_snapshot, TelemetryError
from kev_laya.io import atomic_json
from kev_laya.registration import register
from overwatch.providers.kev_laya import collect_snapshots
from overwatch.kev_laya_adapter import merge_snapshots, presentation

@pytest.fixture
def snapshot():
    p={'model':'kev-laya-fixture','backbone':'fixture','backbone_revision':'fixture-v1','tokenizer':'bytes',
       'config_sha256':'1'*64,'data_sha256':'2'*64,'split_hashes':{},'seed':42,'repository':'https://github.com/example/model',
       'commit':'a'*40,'precision':'fp32','hardware':'cpu','context_limits':{'branch':512,'aggregate':8192},'evidence_class':'tiny-synthetic-fixture'}
    s=new_snapshot('experiment','run','attempt-0',p);s['phase']='train';return s

def collected(*snapshots,ids=None):
    ids=ids or [f'source-{i}' for i in range(len(snapshots))]
    return {'records':[{'snapshot':s,'source_id':id,'transport':'local','provider_status':None} for id,s in zip(ids,snapshots)],
            'warnings':[],'configured_ids':ids}

def test_contract_copies_identical():
    import kev_laya.telemetry_contract as a
    import overwatch.kev_laya_contract as b
    assert Path(a.__file__).read_bytes()==Path(b.__file__).read_bytes()

@pytest.mark.parametrize('change',[{'framework':'TrainConfig'},{'schema_version':2},{'state':'private'},{'sequence':-1},
    {'phase':'running'},{'attempt_index':1},{'history':[{}]*129},{'metrics':{'unexpected':1}},
    {'resume':{'capable':True,'checkpoint_sha256':None}}])
def test_contract_rejects_bad_snapshots(snapshot,change):
    snapshot.update(change)
    with pytest.raises(ValueError):validate_snapshot(snapshot)

def test_no_credential_artifact_uri(snapshot):
    snapshot['artifacts']=[{'kind':'evaluation','uri':'https://example.test/file?token=secret', 'sha256':'a'*64,
                            'size_bytes':1,'parent_sha256':None,'resumable':False}]
    with pytest.raises(ValueError):validate_snapshot(snapshot)


def test_streamed_training_metrics_are_v2_only(snapshot):
    additions={'numerics/recomputed_logits_max_abs':0.0,
               'execution/streamed_recompute_compute_tokens':66624.0}
    snapshot['metrics']=additions
    with pytest.raises(TelemetryError):
        validate_snapshot(snapshot)
    snapshot['schema_version']=2
    snapshot['framework_version']='0.1.0+stage3'
    snapshot['extensions']={'serialization':{'tokenizer':'bytes','serialization':'fixture','literal_encoding':'fixture'},
                            'source':{'source_tree_sha256':None,'archive_sha256':None,'git_dirty':None,
                                      'git_status':'unavailable','package_version':'fixture'},
                            'attempt_lineage':[],'lineage_durable':None,'execution':None,'resources':None,
                            'calibration':None}
    snapshot['history']=[{'step':0,'phase':'train','metrics':additions}]
    validate_snapshot(snapshot)

def test_writer_bounded_atomic_and_failure_isolation(tmp_path,monkeypatch,snapshot):
    path=tmp_path/'snapshot.json';writer=TelemetryWriter(path,snapshot)
    for i in range(135):
        progress=copy.deepcopy(writer.snapshot['progress']);progress['optimizer_steps']=i
        assert writer.update(progress=progress,metrics={'loss/ce':1.0/(i+1)})
    good=path.read_bytes();data=decode_snapshot(good)
    assert len(data['history'])==128 and data['sequence']==135
    def fail(*a,**kw):raise OSError('SECRET-CREDENTIAL-MUST-NOT-LEAK')
    monkeypatch.setattr('kev_laya.telemetry.atomic_json',fail)
    with pytest.warns(RuntimeWarning) as w:assert not writer.export()
    assert 'SECRET' not in str(w[0].message)
    assert writer.failures==1 and path.read_bytes()==good
    monkeypatch.undo()
    assert writer.export() and decode_snapshot(path.read_bytes())['monitoring_export_failures']==1

def test_local_registration_provider_cache_presentation(tmp_path,snapshot):
    path=tmp_path/'snapshot.json';writer=TelemetryWriter(path,snapshot);writer.update(metrics={'loss/ce':.7})
    registry=tmp_path/'registry.json';register(registry,path,'local')
    batch=collect_snapshots(registry)
    env=merge_snapshots(None,batch)
    view=presentation(env)
    assert len(view['runs'])==1 and view['runs'][0]['association']['status']=='local'
    assert view['runs'][0]['provider_status']['skypilot'] is None
    assert view['runs'][0]['git_url'].endswith('/commit/'+'a'*40)
    assert view['runs'][0]['resume']['capable'] is False
    assert not view['warnings']

def test_corrupt_source_keeps_last_valid_and_reports_error(tmp_path,snapshot):
    path=tmp_path/'snapshot.json';TelemetryWriter(path,snapshot).export()
    registry=tmp_path/'registry.json';register(registry,path,'local')
    env=merge_snapshots(None,collect_snapshots(registry))
    path.write_text('{corrupt')
    merged=merge_snapshots(env,collect_snapshots(registry))
    assert len(merged['records'])==1 and merged['warnings']

def test_disabled_optional_transports_not_imported(tmp_path,monkeypatch,snapshot):
    path=tmp_path/'snapshot.json';TelemetryWriter(path,snapshot).export()
    registry=tmp_path/'registry.json';register(registry,path,'local')
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name in {'wandb','boto3'}:raise AssertionError('disabled SDK imported')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert len(collect_snapshots(registry)['records'])==1

def test_wandb_enabled_structured_listing_and_dedup(tmp_path,snapshot):
    identity={'entity':'entity','project':'project','run_id':'wb-run'}
    snapshot['wandb']=identity
    path=tmp_path/'snapshot.json';TelemetryWriter(path,snapshot).export()
    written=json.loads(path.read_text())
    registry=tmp_path/'registry.json'
    atomic_json(registry,{'schema_version':1,'sources':[{'id':'file','transport':'local','path':str(path)},
        {'id':'wb','transport':'wandb','entity':'entity','project':'project','limit':5}]})
    class API:
        calls=0
        def fetch_snapshot_page(self, *, entity, project, first, cursor, timeout):
            self.calls+=1
            assert (entity,project,first,cursor)==('entity','project',5,None)
            payload=json.dumps({'data':{'project':{'runs':{'edges':[{'node':{
                'name':'wb-run','state':'running','summaryMetrics':json.dumps({'kev_laya/snapshot':written})}}],
                'pageInfo':{'hasNextPage':False,'endCursor':None}}}}}).encode()
            return {'Body':io.BytesIO(payload),'ContentLength':len(payload)}
        def run(self,*a):raise AssertionError('no per-run hydration')
    api=API();batch=collect_snapshots(registry,wandb_api=api);env=merge_snapshots(None,batch)
    assert api.calls==1 and len(batch['records'])==2 and len(env['records'])==1
    view=presentation(env)['runs'][0]
    assert set(view['transports'])=={'local','wandb'}
    assert view['provider_status']['wandb']=='running'

def test_s3_enabled_compact_snapshot(tmp_path,snapshot):
    registry=tmp_path/'registry.json';atomic_json(registry,{'schema_version':1,'sources':[{'id':'s3','transport':'s3','bucket':'bucket','key':'snapshot.json'}]})
    class S3:
        def get_object(self,**kwargs):return {'Body':io.BytesIO(json.dumps(snapshot).encode())}
    assert len(collect_snapshots(registry,s3_client=S3())['records'])==1

def test_sequence_regression_conflict_restart(snapshot):
    snapshot['sequence']=10
    env=merge_snapshots(None,collected(snapshot,ids=['file']))
    old=copy.deepcopy(snapshot);old['sequence']=9
    reg=merge_snapshots(env,collected(old,ids=['file']))
    assert reg['records'][0]['snapshot']['sequence']==10 and reg['warnings']
    conflict=copy.deepcopy(snapshot);conflict['metrics']={'loss/ce':1.}
    conflicted=merge_snapshots(env,collected(conflict,ids=['file']))
    assert conflicted['records'][0]['snapshot']['metrics']=={}
    restart=copy.deepcopy(snapshot);restart.update(attempt_id='attempt-1',parent_attempt_id='attempt-0',attempt_index=1,sequence=1)
    merged=merge_snapshots(env,collected(restart,ids=['file']))
    assert merged['records'][0]['snapshot']['attempt_index']==1
    rollback=merge_snapshots(merged,collected(snapshot,ids=['file']))
    assert rollback['records'][0]['snapshot']['attempt_index']==1

def test_ambiguous_attempt_identity_is_not_silently_chosen(snapshot):
    other=copy.deepcopy(snapshot);other['attempt_id']='conflicting-attempt'
    env=merge_snapshots(None,collected(snapshot,other))
    assert env['warnings']

def test_stale_heartbeat_and_explicit_namespace_join(snapshot):
    now=datetime.now(timezone.utc)
    then=(now-timedelta(minutes=10)).isoformat()
    snapshot.update(started_at=then,updated_at=then,heartbeat_at=then)
    snapshot['scheduler']={'provider':'skypilot','namespace':'controller-a','job_id':7}
    env=merge_snapshots(None,collected(snapshot),now=now)
    jobs=[{'job_id':7,'job_name':'different-name','status':'RUNNING','recovery_count':5,'scheduler_namespace':'controller-a'}]
    view=presentation(env,jobs,now=now)['runs'][0]
    assert view['freshness']=='stale' and view['association']['status']=='matched'
    assert view['recoveries']=={'total':5,'infrastructure':None,'application':None}
    assert view['provider_status']['skypilot']=='RUNNING'
    jobs[0]['scheduler_namespace']='controller-b'
    assert presentation(env,jobs,now=now)['runs'][0]['association']['status']=='unmatched'
    jobs[0]['scheduler_namespace']='controller-a'
    assert presentation(env,jobs+jobs,now=now)['runs'][0]['association']['status']=='ambiguous'

def test_source_revocation_removes_run(snapshot):
    env=merge_snapshots(None,collected(snapshot))
    result=merge_snapshots(env,{'records':[],'warnings':[],'configured_ids':[]})
    assert result['records']==[]

def test_reporting_functions_do_not_read_sources(monkeypatch,snapshot):
    env=merge_snapshots(None,collected(snapshot))
    def fail(*a,**k):raise AssertionError('report opened a source')
    monkeypatch.setattr(builtins,'open',fail)
    assert presentation(env)['runs'][0]['run_id']=='run'

def test_corrupt_cache_wrapper_does_not_crash_presentation(snapshot):
    env=merge_snapshots(None,collected(snapshot))
    env['records'].extend([{}, {'snapshot':{'wandb':{'entity':'missing'}}}])
    out=presentation(env)
    assert len(out['runs'])==1 and len(out['warnings'])==2

def test_retention_does_not_reingest_expired_registered_snapshot(snapshot):
    then=(datetime.now(timezone.utc)-timedelta(days=30)).isoformat()
    snapshot.update(started_at=then,updated_at=then,heartbeat_at=then)
    env=merge_snapshots(None,collected(snapshot),retention_days=14)
    assert env['records']==[] and env['warnings'][0]['code']=='expired_snapshot_ignored'

def test_counter_regression_rejected(snapshot):
    snapshot['progress']['optimizer_steps']=10
    env=merge_snapshots(None,collected(snapshot))
    other=copy.deepcopy(snapshot);other['sequence']+=1;other['progress']['optimizer_steps']=9
    out=merge_snapshots(env,collected(other))
    assert out['records'][0]['snapshot']['progress']['optimizer_steps']==10
    assert out['warnings'][-1]['code']=='progress_regression'

def test_explicit_optional_publish_contracts(tmp_path,snapshot):
    from kev_laya.publishing import publish_wandb,publish_s3
    identity={'entity':'e','project':'p','run_id':'r'};snapshot['wandb']=identity
    path=tmp_path/'s.json';TelemetryWriter(path,snapshot).export()
    calls=[]
    class Summary(dict):
        def update(self, *a, **kw):super().update(*a, **kw);calls.append('summary-update')
    class Run:
        summary=Summary()
        config={}
        def update(self):calls.append('run-update')
        def finish(self):raise AssertionError('snapshot publishing does not own run lifecycle')
    class API:
        def run(self, name):calls.append(name);return Run()
    class SDK:
        def Api(self, **kw):return API()
        def init(self, **kw):raise AssertionError('must not start/finish a live trainer run')
    publish_wandb(path,entity='e',project='p',sdk=SDK())
    assert calls[0]=='e/p/r' and calls[-1]=='summary-update'
    assert Run.summary['kev_laya/snapshot']['framework']=='kev_laya'
    with pytest.raises(ValueError):publish_wandb(path,entity='wrong',project='p',sdk=SDK())
    class S3:
        def put_object(self,**kw):calls.append(kw)
    publish_s3(path,bucket='example-bucket',key='telemetry/snapshot.json',client=S3())
    assert calls[-1]['ServerSideEncryption']=='AES256'

def test_collector_lock_exclusion_and_release(tmp_path):
    from overwatch.kev_laya_lock import collector_lock
    path=tmp_path/'collector.lock'
    with collector_lock(path):
        with pytest.raises(OSError):
            with collector_lock(path):pass
    with collector_lock(path):pass
