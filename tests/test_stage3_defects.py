"""Focused regressions for the independent-review defects. CPU/doubles are labeled."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from kev_laya.encoding import (ByteTokenizer, QwenTokenizer, LITERAL_ENCODING, LEGACY_ESCAPE,
    NATIVE_SERIALIZATION, Limits, encode_request, preprocessing_identity)
from kev_laya.schema import SystemOneRequest
from kev_laya.telemetry import new_snapshot, TelemetryWriter
from kev_laya.telemetry_contract import validate_snapshot, validate_attempt_records
from kev_laya.provenance import capture_source, extensions, ResourceSampler
from kev_laya.lineage import begin_attempt
from kev_laya.checkpoint import save_checkpoint, load_checkpoint
from kev_laya.exposure import verify_exposure
from kev_laya.training import train, TrainSettings
from kev_laya.objectives import ObjectiveConfig
from kev_laya.io import atomic_json, sha256_file
from overwatch.providers.kev_laya import collect_snapshots, CollectionLimits
from overwatch.kev_laya_adapter import merge_snapshots, presentation


def provenance():
    return {'model':'test','backbone':'fixture','backbone_revision':'fixture-v1','tokenizer':ByteTokenizer.identity,
       'config_sha256':'1'*64,'data_sha256':'2'*64,'split_hashes':{},'seed':42,'repository':None,
       'commit':None,'precision':'fp32','hardware':'cpu','context_limits':{'branch':512,'aggregate':8192},'evidence_class':'tiny-synthetic-fixture'}


def source_info():
    return {'source_tree_sha256':None,'archive_sha256':None,'git_dirty':None,'git_status':'unavailable','package_version':'test'}


def snapshot(state=None, *, v2=False):
    state = state or {'experiment_id':'e','run_id':'r','attempt_id':'a0','attempt_index':0,'parent_attempt_id':None}
    kw = {'schema_version':2, 'extensions':extensions(ByteTokenizer(),source_info(),lineage=state.get('attempt_lineage',[]))} if v2 else {}
    s = new_snapshot(state['experiment_id'],state['run_id'],state['attempt_id'],provenance(),
         attempt_index=state['attempt_index'],parent_attempt_id=state['parent_attempt_id'], **kw)
    s['phase'] = 'train'
    return s


def collection(*ss):
    return {'records':[{'snapshot':s,'source_id':f's{i}','transport':'local','provider_status':None} for i,s in enumerate(ss)],
            'warnings':[],'configured_ids':[f's{i}' for i in range(len(ss))]}


class LiteralTokenizerDouble:
    """Recording backend for semantic-contract checks, NOT native tokenizer evidence."""
    encode_special_tokens = False
    def no_padding(self):pass
    def no_truncation(self):pass
    @classmethod
    def from_file(cls,path):return cls()
    @classmethod
    def from_str(cls,text):
        obj=cls()
        obj.literal_backend=json.loads(text).get('added_tokens')==[]
        obj.encode_special_tokens=obj.literal_backend
        return obj
    def token_to_id(self,s):
        from kev_laya.encoding import SPECIAL
        return 256+SPECIAL.index(s)
    def get_added_tokens_decoder(self):
        return {} if getattr(self,'literal_backend',False) else {i:SimpleNamespace(special=True) for i in range(256,261)}
    def get_vocab_size(self):return 261
    def encode(self,text,add_special_tokens=False):
        from kev_laya.encoding import SPECIAL
        if not self.encode_special_tokens and text in SPECIAL:
            return SimpleNamespace(ids=[self.token_to_id(text)])
        return SimpleNamespace(ids=list(text.encode('utf-8')))
    def decode(self,ids,skip_special_tokens=False):return bytes(ids).decode('utf-8')


def qwen_double(tmp_path, monkeypatch, mode=LITERAL_ENCODING):
    monkeypatch.setitem(sys.modules,'tokenizers',SimpleNamespace(Tokenizer=LiteralTokenizerDouble))
    path=tmp_path/'tokenizer.json';path.write_text(json.dumps({'model':{'type':'BPE'}, 'decoder':{'type':'ByteLevel'}, 'added_tokens':[{'id':256}]}))
    return QwenTokenizer(path,literal_encoding=mode)


@pytest.mark.parametrize('text', ['<|fim_prefix|>','<¦fim_prefix¦>',r'\\<|fim_prefix|>\\',r'\u003c|fim_prefix|>',
    '<|fim_prefix|><|fim_prefix|>','é中文🙂\n\x00',r'\\\\','𝓽𝓮𝔁𝔱','e\u0301','<|im_start|>'])
def test_literal_contract_double_round_trip(tmp_path,monkeypatch,text):
    tok=qwen_double(tmp_path,monkeypatch)
    ids=tok.encode(text)
    assert tok.decode_literal(ids)==text
    assert not set(ids).intersection(tok.special)
    assert tok.metadata()['serialization']==NATIVE_SERIALIZATION


def test_literal_collision_fixed_and_legacy_not_silently_reinterpreted(tmp_path,monkeypatch):
    new=qwen_double(tmp_path,monkeypatch)
    old=qwen_double(tmp_path,monkeypatch,LEGACY_ESCAPE)
    assert new.encode('<|fim_prefix|>') != new.encode('<¦fim_prefix¦>')
    assert old.encode('<|fim_prefix|>') == old.encode('<¦fim_prefix¦>')
    assert preprocessing_identity(new) != preprocessing_identity(old)


def test_v2_nested_json_ids_accounting_and_overflow(tmp_path,monkeypatch):
    tok=qwen_double(tmp_path,monkeypatch)
    req=SystemOneRequest(state={'nested':['<|fim_prefix|>','<¦fim_prefix¦>',{'x':r'\\'}]},model='test',questions={
      'metadata-only-id':{'type':'choice','instructions':{'i':['é','<|fim_middle|>']},'criteria':{'<|box_start|>':['unicode','🙂'],'other':None}}})
    e=encode_request(req,tok,Limits(1024,2048))
    assert json.loads(tok.decode_literal(list(e.state[1:]))) == {'state':req.state}
    assert e.logical_tokens==len(e.state)+sum(len(b.ids) for b in e.branches)
    renamed=req.model_copy(update={'questions':{'renamed':next(iter(req.questions.values()))}})
    e2=encode_request(renamed,tok,Limits(1024,2048))
    assert e.state==e2.state and e.branches[0].ids==e2.branches[0].ids
    with pytest.raises(ValueError):encode_request(req,tok,Limits(e.logical_tokens-1,e.logical_tokens-1))
    text=req.model_copy(update={'state':'{}'}); obj=req.model_copy(update={'state':{}})
    assert encode_request(text,tok,Limits(1024,2048)).state != encode_request(obj,tok,Limits(1024,2048)).state


def test_lossy_backend_fails_instead_of_replacing(tmp_path,monkeypatch):
    tok=qwen_double(tmp_path,monkeypatch)
    tok._tokenizer.decode=lambda *a,**k:'WRONG'
    with pytest.raises(ValueError,match='reversible'):tok.encode('actual text')


def test_checkpoint_preprocessing_round_trip_and_mismatch(tmp_path,tiny,monkeypatch):
    tok=qwen_double(tmp_path,monkeypatch)
    p=tmp_path/'new.pt';save_checkpoint(p,tiny,tok)
    _,loaded,_=load_checkpoint(p)
    assert loaded.literal_encoding==LITERAL_ENCODING
    assert loaded.encode('<|fim_prefix|>')!=loaded.encode('<¦fim_prefix¦>')
    payload=torch.load(p,weights_only=True);payload['tokenizer']['serialization']='kev-laya-prefix-v1'
    bad=tmp_path/'bad.pt';torch.save(payload,bad)
    with pytest.raises(ValueError,match='serialization'):load_checkpoint(bad)
    old=tmp_path/'legacy.pt';save_checkpoint(old,tiny,qwen_double(tmp_path,monkeypatch,LEGACY_ESCAPE))
    _,t,point=load_checkpoint(old)
    assert t.literal_encoding==LEGACY_ESCAPE


def test_missed_attempts_durable_restart_and_cache_restart(tmp_path):
    st={'experiment_id':'e','run_id':'r','attempt_id':'a0','attempt_index':0,'parent_attempt_id':None}
    begin_attempt(tmp_path,st,resumed=False); first=copy.deepcopy(st)
    s0=snapshot(st,v2=True);env=merge_snapshots(None,collection(s0))
    begin_attempt(tmp_path,st,resumed=True)
    s1=snapshot(st,v2=True)
    # Restore an older optimizer checkpoint; receipt file still remembers attempt 1.
    st=copy.deepcopy(first);begin_attempt(tmp_path,st,resumed=True)
    assert st['attempt_index']==2
    s2=snapshot(st,v2=True)
    env=merge_snapshots(json.loads(json.dumps(env)),collection(s2))
    assert env['records'][0]['snapshot']['attempt_index']==2
    assert presentation(env)['runs'][0]['verification']=='verified'
    later=merge_snapshots(env,collection(s1))
    assert later['records'][0]['snapshot']['attempt_index']==2


def test_unproven_new_observation_is_visible_not_silently_current():
    s0=snapshot();env=merge_snapshots(None,collection(s0))
    new=copy.deepcopy(s0);new.update(attempt_id='a2',attempt_index=2,parent_attempt_id='a1')
    out=merge_snapshots(env,collection(new));view=presentation(out)['runs'][0]
    assert view['attempt_index']==0
    assert view['verification']=='last_verified_with_newer_uncertainty'
    assert view['unverified_observations'][0]['attempt_index']==2
    first=merge_snapshots(None,collection(new));view=presentation(first)['runs'][0]
    assert view['verification']=='unverified' and view['attempt_index']==2


def test_attempt_receipt_tampering_and_identity_rejected(tmp_path):
    st={'experiment_id':'e','run_id':'r','attempt_id':'a0','attempt_index':0,'parent_attempt_id':None}
    begin_attempt(tmp_path,st,resumed=False);begin_attempt(tmp_path,st,resumed=True)
    records=copy.deepcopy(st['attempt_lineage']);records[0]['attempt_id']='forged'
    with pytest.raises(ValueError):validate_attempt_records(records,'e','r')
    with pytest.raises(ValueError):validate_attempt_records(st['attempt_lineage'],'other','r')


def test_conflicting_transports_retained_as_uncertain():
    s=snapshot();other=copy.deepcopy(s);other['metrics']={'loss/ce':1.}
    env=merge_snapshots(None,collection(s,other))
    assert env['observations'] and presentation(env)['runs'][0]['verification']=='unverified'


def write_registry(path,sources):atomic_json(path,{'schema_version':1,'sources':sources})


class S3Double:
    def __init__(self,payload,*,known=True,on_read=None):
        self.payload,self.known,self.on_read=payload,known,on_read
        self.fetches=self.bytes=self.reads=0
    def get_object(self,**kw):
        self.fetches+=1;owner=self
        class Body(io.BytesIO):
            def read(self,n=-1):
                owner.reads+=1
                if owner.on_read:owner.on_read()
                raw=super().read(n);owner.bytes+=len(raw);return raw
        out={'Body':Body(self.payload)}
        if self.known:out['ContentLength']=len(self.payload)
        return out


def test_16mib_budget_stops_actual_reads_and_requests(tmp_path):
    p=tmp_path/'reg.json'
    sources=[{'id':f's{i}','transport':'s3','bucket':'b','key':str(i)} for i in range(70)]
    write_registry(p,sources)
    raw=json.dumps(snapshot()).encode().ljust(262144,b' ')
    sdk=S3Double(raw);out=collect_snapshots(p,s3_client=sdk,limits=CollectionLimits(seconds=60))
    assert sdk.bytes==16*1024*1024 and sdk.fetches==64
    assert out['accounting']['payload_bytes']==sdk.bytes
    assert len(out['configured_ids'])==70
    assert all(s['status']=='skipped_payload_budget' for s in out['sources'][64:])


def test_budget_overflow_probe_is_charged(tmp_path):
    p=tmp_path/'reg.json';write_registry(p,[{'id':'s','transport':'s3','bucket':'b','key':'x'}])
    sdk=S3Double(b'x'*300000,known=False)
    out=collect_snapshots(p,s3_client=sdk)
    assert sdk.bytes==262145 and out['accounting']['payload_bytes']==262145
    assert not out['records']


def test_final_partial_read_never_exceeds_remaining(tmp_path):
    p=tmp_path/'reg.json';write_registry(p,[{'id':f's{i}','transport':'s3','bucket':'b','key':str(i)} for i in range(3)])
    sdk=S3Double(b'x'*300,known=False)
    out=collect_snapshots(p,s3_client=sdk,limits=CollectionLimits(payload_bytes=101))
    assert sdk.bytes==101 and sdk.fetches==1
    assert len(out['configured_ids'])==3


def test_registry_late_invalid_entry_prevents_all_payload_io(tmp_path):
    p=tmp_path/'reg.json';write_registry(p,[{'id':'s','transport':'s3','bucket':'b','key':'x'}, {'id':'bad','transport':'unsupported'}])
    sdk=S3Double(b'{}')
    with pytest.raises(ValueError):collect_snapshots(p,s3_client=sdk)
    assert sdk.fetches==0


def test_cooperative_deadline_stops_further_io(tmp_path):
    p=tmp_path/'reg.json';write_registry(p,[{'id':f's{i}','transport':'s3','bucket':'b','key':str(i)} for i in range(5)])
    now=[0.];clock=lambda:now[0]
    sdk=S3Double(b'x'*100000,on_read=lambda:now.__setitem__(0,now[0]+1))
    out=collect_snapshots(p,s3_client=sdk,limits=CollectionLimits(seconds=2),clock=clock)
    assert sdk.fetches==1 and sdk.reads==2
    assert out['accounting']['stop_reason']=='deadline'


def test_request_budget_stops_before_next_get(tmp_path):
    p=tmp_path/'reg.json';write_registry(p,[{'id':f's{i}','transport':'s3','bucket':'b','key':str(i)} for i in range(5)])
    sdk=S3Double(json.dumps(snapshot()).encode())
    out=collect_snapshots(p,s3_client=sdk,limits=CollectionLimits(requests=2))
    assert sdk.fetches==2 and out['sources'][2]['status']=='skipped_request_budget'


def test_partial_and_missing_registry_preserve_cached_sources(tmp_path):
    s=snapshot();env=merge_snapshots(None,collection(s))
    p=tmp_path/'reg.json';write_registry(p,[{'id':'s0','transport':'s3','bucket':'b','key':'x'}])
    out=collect_snapshots(p,s3_client=S3Double(b'{}'),limits=CollectionLimits(seconds=0))
    merged=merge_snapshots(env,out)
    assert len(merged['records'])==1
    p.unlink();merged=merge_snapshots(env,collect_snapshots(p))
    assert len(merged['records'])==1
    write_registry(p,[]);assert merge_snapshots(env,collect_snapshots(p))['records']==[]


class WbPagesDouble:
    def __init__(self,snapshot,*,has_next=True):self.snapshot=snapshot;self.calls=0;self.has_next=has_next
    def fetch_snapshot_page(self,**kw):
        self.calls+=1
        node={'name':self.snapshot['wandb']['run_id'],'state':'running','summaryMetrics':json.dumps({'kev_laya/snapshot':self.snapshot})}
        raw=json.dumps({'data':{'project':{'runs':{'edges':[{'node':node}], 'pageInfo':{'hasNextPage':self.has_next,'endCursor':f'c{self.calls}'}}}}}).encode()
        return {'Body':io.BytesIO(raw),'ContentLength':len(raw)}


def test_wandb_pagination_and_payload_bounded(tmp_path):
    s=snapshot();s['wandb']={'entity':'e','project':'p','run_id':'r'}
    p=tmp_path/'reg.json';write_registry(p,[{'id':'w','transport':'wandb','entity':'e','project':'p','limit':10}])
    sdk=WbPagesDouble(s)
    out=collect_snapshots(p,wandb_api=sdk,limits=CollectionLimits(pages=2))
    assert sdk.calls==2 and out['accounting']['pages']==2
    assert out['accounting']['stop_reason']=='page_budget'
    assert out['accounting']['payload_bytes']>2*len(json.dumps(s).encode())


def test_calibration_preserves_verified_observed_exposure(tmp_path,tiny,suite,capsys):
    from kev_laya.cli import main
    data,manifest=suite
    run=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',TrainSettings(steps=2,save_every=2),ObjectiveConfig(),Limits(512,8192))
    parent=Path(run['checkpoint']);dest=parent.parent/'calibrated.pt'
    main(['calibrate','--checkpoint',str(parent),'--suite',str(tmp_path/'suite'),'--out',str(dest)])
    _,_,point=load_checkpoint(dest)
    assert point['training_state'] is None and point['optimizer'] is None
    report=verify_exposure(dest)
    assert report['status']=='verified' and report['observed']['examples']==4
    assert not report['native_32k_64k']
    assert report['observed']['maximum_branch_tokens']<512
    # Moving/losing the parent does not silently grant exposure.
    renamed=parent.with_name('moved.pt');parent.rename(renamed)
    assert verify_exposure(dest)['status']=='unknown'
    assert verify_exposure(dest,parents=[renamed])['status']=='verified'
    with renamed.open('ab') as f:f.write(b'tamper')
    assert verify_exposure(dest,parents=[renamed])['status']=='unknown'


def test_config_counters_and_legacy_artifacts_do_not_grant_exposure(tmp_path,tiny):
    p=tmp_path/'legacy.pt';tiny.training_steps=999
    save_checkpoint(p,tiny,ByteTokenizer(),training_state={'accumulation_position':0,'maximum_branch_tokens_seen':32768,'maximum_aggregate_tokens_seen':65536},
                    provenance={'context_limits':{'branch':32768,'aggregate':65536}})
    assert verify_exposure(p)['status']=='unknown'


def test_exposure_tamper_and_tokenizer_mismatch(tmp_path,tiny,suite):
    from kev_laya.evaluation import calibrate
    data,manifest=suite
    run=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',TrainSettings(steps=1),ObjectiveConfig(),Limits(512,8192))
    parent=Path(run['checkpoint']);dest=parent.parent/'cal.pt'
    save_checkpoint(dest,tiny,ByteTokenizer(),parent_checkpoint=parent,parent_sha256=sha256_file(parent),exposure_operation='calibration')
    point=torch.load(dest,weights_only=True)
    point['training_exposure']['observed']['maximum_branch_tokens']=32768
    tamper=parent.parent/'tamper.pt';torch.save(point,tamper)
    assert verify_exposure(tamper)['status']=='unknown'
    point['training_exposure']['identity']['serialization']='not-the-tokenizer'
    torch.save(point,tamper)
    assert 'mismatch' in verify_exposure(tamper)['reason']


def test_real_git_commit_dirty_and_archive_hash(tmp_path):
    code=tmp_path/'src/kev_laya';code.mkdir(parents=True);p=code/'a.py';p.write_text('x=1\n')
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    def git(*args):return subprocess.run(['git','-C',str(tmp_path),*args],capture_output=True,check=True,text=True).stdout.strip()
    # This fixture tests source capture, independently of a user's signing agent.
    git('add','.');git('-c','user.name=Test','-c','user.email=test@example.invalid',
                        '-c','commit.gpgsign=false','commit','-qm','fixture')
    git('remote','add','origin','https://github.com/example/source.git')
    source=capture_source(tmp_path)
    assert source['commit']==git('rev-parse','HEAD') and source['git_dirty'] is False
    assert source['repository']=='https://github.com/example/source'
    p.write_text('x=2\n');dirty=capture_source(tmp_path,archive=p)
    assert dirty['git_dirty'] is True and dirty['source_tree_sha256']!=source['source_tree_sha256']
    assert dirty['archive_sha256']==sha256_file(p)


def test_resource_sampling_is_measured_and_cadenced():
    now=[0.];sampler=ResourceSampler(interval=1,clock=lambda:now[0],device=torch.device('cpu'))
    a=sampler.sample();now[0]=.1;assert sampler.sample()==a
    now[0]=2;b=sampler.sample()
    assert b['units']=='bytes' and b['rss_bytes']>0
    assert b['gpu_allocated_bytes'] is None and 'cuda' in b['unavailable']


class WbSDKDouble:
    def __init__(self):
        self.snapshots=[];owner=self
        class Summary(dict):
            def update(self,*a,**kw):
                if a or kw:return super().update(*a,**kw)
                owner.snapshots.append(copy.deepcopy(self['kev_laya/snapshot']))
        self.run=SimpleNamespace(config={},summary=Summary(),update=lambda:None)
        self.calls=0
    def Api(self,**kw):
        owner=self
        class API:
            def run(self,path):owner.calls+=1;return owner.run
        return API()
    def init(self,**kw):raise AssertionError('non-owner must not initialize a W&B lifecycle')
    def finish(self):raise AssertionError('non-owner must not finish a W&B lifecycle')


def test_training_wandb_identity_resume_and_nonterminal_lifecycle(tmp_path,tiny,suite):
    data,manifest=suite;wb=WbSDKDouble();identity={'entity':'e','project':'p','run_id':'r'}
    settings=TrainSettings(steps=3,save_every=1)
    part=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',settings,ObjectiveConfig(),Limits(512,8192),
               stop_after=1,wandb_ref=identity,publish_wandb=True,wandb_sdk=wb)
    assert wb.snapshots[-1]['phase']=='train' and wb.calls==1
    model,tok,_=load_checkpoint(Path(part['checkpoint']))
    resumed=train(model,tok,data['train'],manifest,tmp_path/'run',settings,ObjectiveConfig(),Limits(512,8192),
                  resume=Path(part['checkpoint']),publish_wandb=True,wandb_sdk=wb)
    assert wb.snapshots[-1]['phase']=='completed'
    assert resumed['state']['wandb_identity']==identity
    assert wb.run.config=={'framework':'kev_laya','telemetry_schema_version':2}
    with pytest.raises(ValueError,match='W&B identity'):
        train(model,tok,data['train'],manifest,tmp_path/'run',settings,ObjectiveConfig(),Limits(512,8192),resume=Path(part['checkpoint']),wandb_ref=identity|{'run_id':'other'})


def test_publishing_disabled_makes_zero_remote_calls(tmp_path,tiny,suite):
    data,manifest=suite;wb=WbSDKDouble()
    train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',TrainSettings(steps=1),ObjectiveConfig(),Limits(512,8192),
          wandb_ref={'entity':'e','project':'p','run_id':'r'},publish_wandb=False,wandb_sdk=wb)
    assert wb.calls==0
