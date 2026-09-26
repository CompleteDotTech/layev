import copy
import json
from pathlib import Path
import pytest
import torch
from kev_laya.checkpoint import load_checkpoint, save_checkpoint
from kev_laya.data import load_suite, Sampler
from kev_laya.io import sha256_file, atomic_json
from kev_laya.encoding import ByteTokenizer, Limits
from kev_laya.model import DecisionEngine
from kev_laya.objectives import ObjectiveConfig
from kev_laya.training import train, TrainSettings
from kev_laya.evaluation import calibrate, evaluate, summarize

@pytest.mark.parametrize('reward', [0., .1])
def test_interrupted_equals_uninterrupted(tmp_path,tiny,suite,reward):
    data,manifest=suite
    initial=copy.deepcopy(tiny.state_dict())
    settings=TrainSettings(steps=6,accumulation=2,save_every=3,seed=19)
    cfg=ObjectiveConfig(reinforce=reward)
    limits=Limits(512,8192)
    full=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'full',settings,cfg,limits)
    split=DecisionEngine(tiny.cfg);split.load_state_dict(initial)
    part=train(split,ByteTokenizer(),data['train'],manifest,tmp_path/'split',settings,cfg,limits,stop_after=3)
    # Load and construct afresh, disturb the global RNG, then restore from the saved boundary.
    resumed,tokenizer,_=load_checkpoint(Path(part['checkpoint']))
    torch.randn(57)
    end=train(resumed,tokenizer,data['train'],manifest,tmp_path/'split',settings,cfg,limits,resume=Path(part['checkpoint']))
    for n,p in tiny.state_dict().items(): torch.testing.assert_close(p,resumed.state_dict()[n],atol=0,rtol=0)
    for k in ('step','examples','microbatches','forward_tokens'):
        assert full['state'][k]==end['state'][k]
    assert end['state']['run_id']==part['state']['run_id']
    assert end['state']['attempt_index']==1
    assert end['state']['parent_attempt_id']==part['state']['attempt_id']
    assert end['state']['attempt_id']!=part['state']['attempt_id']
    a=load_checkpoint(Path(full['checkpoint']))[2]; b=load_checkpoint(Path(end['checkpoint']))[2]
    assert a['sampler']==b['sampler']
    assert a['scheduler']==b['scheduler']
    for key, item in a['optimizer']['state'].items():
        for name,value in item.items():
            if isinstance(value,torch.Tensor): torch.testing.assert_close(value,b['optimizer']['state'][key][name],atol=0,rtol=0)
    assert a['training_steps']==b['training_steps']==6

def test_checkpoint_integrity_and_optimization_boundary(tmp_path,tiny):
    p=tmp_path/'checkpoint.pt'
    with pytest.raises(ValueError): save_checkpoint(p,tiny,ByteTokenizer(),training_state={'accumulation_position':1})
    manifest=save_checkpoint(p,tiny,ByteTokenizer())
    model,tok,payload=load_checkpoint(p,expected_sha256=manifest['sha256'])
    assert not manifest['resumable']
    assert payload['model_id'].startswith('kev-laya-0.1.0-')
    with pytest.raises(ValueError): load_checkpoint(p,expected_sha256='0'*64)
    with p.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError): load_checkpoint(p)

def test_refuse_resume_config_changes(tmp_path,tiny,suite):
    data,manifest=suite
    cfg=ObjectiveConfig();limits=Limits(512,8192)
    result=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',TrainSettings(steps=2),cfg,limits,stop_after=1)
    with pytest.raises(ValueError,match='configuration'):
        train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',TrainSettings(steps=3),cfg,limits,resume=Path(result['checkpoint']))

def test_sampler_continuation():
    s=Sampler(13,9)
    for _ in range(19): s.next()
    state=s.state_dict(); other=Sampler(13,42);other.load_state_dict(state)
    assert [s.next() for _ in range(90)]==[other.next() for _ in range(90)]

def test_dataset_checksum_and_group_leakage(tmp_path,suite):
    data,manifest=suite
    directory=tmp_path/'suite'
    path=directory/'test.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    rows[0]['meta']['group']=data['train'][0].meta['group']
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='checksum'):load_suite(directory)
    manifest['partitions']['test']['sha256']=sha256_file(path);atomic_json(directory/'manifest.json',manifest)
    with pytest.raises(ValueError,match='group leakage'):load_suite(directory)

@pytest.mark.parametrize('split',['train','development','test'])
def test_calibration_refuses_noncalibration_split(tiny,suite,split):
    data,manifest=suite
    with pytest.raises(ValueError):
        calibrate(tiny,data[split],ByteTokenizer(),Limits(512,8192),split=split,split_sha256=manifest['partitions'][split]['sha256'])

def test_temperature_fit_does_not_change_weights(tiny,suite):
    data,manifest=suite
    before=copy.deepcopy(tiny.state_dict())
    output=calibrate(tiny,data['calibration'][:4],ByteTokenizer(),Limits(512,8192),split='calibration',
                     split_sha256=manifest['partitions']['calibration']['sha256'])
    assert all(.2<=v<=5 for v in tiny.temperatures.values())
    for n,p in tiny.state_dict().items():torch.testing.assert_close(p,before[n],atol=0,rtol=0)
    assert tiny.calibration_provenance['status']=='fitted-held-out'

def test_evaluation_breakdowns_and_probability_metrics(tiny,suite):
    data,_=suite
    report=evaluate(tiny,data['test'][:4],ByteTokenizer(),Limits(512,8192),split='test')
    assert report['summary']['count']==12
    assert 0<=report['summary']['accuracy']<=1 and report['summary']['nll']>=0
    assert 0<=report['summary']['ece']<=1
    assert report['summary']['risk_coverage'][-1]['coverage']==1
    assert report['by']

def test_choice_permutation_preserves_targets_and_ordinal_order(suite):
    import random
    from kev_laya.data import permute_choices
    data,_=suite;item=data['train'][0]
    for seed in range(20):
        permuted=permute_choices(item,random.Random(seed))
        for (key,q),y,old_y in zip(permuted.request.questions.items(),permuted.targets,item.targets):
            old=item.request.questions[key]
            if q.type=='choice':
                assert dict(zip(q.criteria,y))==dict(zip(old.criteria,old_y))
            else:assert q==old and y==old_y

def test_permutation_augmented_resume_exact(tmp_path,tiny,suite):
    data,manifest=suite;initial=copy.deepcopy(tiny.state_dict())
    settings=TrainSettings(steps=4,accumulation=2,save_every=2,choice_permutation=True)
    cfg=ObjectiveConfig(reinforce=.1);limits=Limits(512,8192)
    train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'full-aug',settings,cfg,limits)
    other=DecisionEngine(tiny.cfg);other.load_state_dict(initial)
    part=train(other,ByteTokenizer(),data['train'],manifest,tmp_path/'split-aug',settings,cfg,limits,stop_after=2)
    train(other,ByteTokenizer(),data['train'],manifest,tmp_path/'split-aug',settings,cfg,limits,resume=Path(part['checkpoint']))
    for k,v in tiny.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],atol=0,rtol=0)
