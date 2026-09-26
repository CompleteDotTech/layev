"""Fresh final-source software exercise, NOT a general quality experiment.

Train -> interrupt -> resume twice, letting the collector miss attempt 1;
calibrate -> load -> verify exposure; ingest immutable v2 snapshots via real
local-file provider/adapter code in an isolated cache. Full Overwatch is a
separate native test, never claimed by this adapter-only exercise.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import json
from pathlib import Path
import sys
import time
import torch
from kev_laya.checkpoint import load_checkpoint
from kev_laya.cli import main as cli
from kev_laya.data import freeze_smoke, load_suite
from kev_laya.encoding import ByteTokenizer, Limits, encode_request
from kev_laya.evaluation import evaluate
from kev_laya.exposure import verify_exposure
from kev_laya.io import atomic_json, sha256_file
from kev_laya.model import DecisionEngine, BackboneConfig
from kev_laya.objectives import ObjectiveConfig
from kev_laya.registration import register
from kev_laya.telemetry import TelemetryWriter, artifact
from kev_laya.training import train, TrainSettings
from kev_laya.provenance import hardware_identity
from benchmark_parallel import make_request


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(47)
    freeze_smoke(out/'suite');suite,manifest=load_suite(out/'suite')
    model=DecisionEngine(BackboneConfig());initial=copy.deepcopy(model.state_dict());tok=ByteTokenizer();limits=Limits(512,8192)
    settings=TrainSettings(steps=8,save_every=2,accumulation=2,choice_permutation=True)
    objective=ObjectiveConfig(reinforce=.1)
    integration=Path(__file__).resolve().parents[1]/'integrations/overwatch/added/src'
    sys.path.insert(0,str(integration))
    from overwatch.providers.kev_laya import collect_snapshots
    from overwatch.kev_laya_adapter import merge_snapshots,presentation
    snapshots=[]
    point=None
    for step in (2,4,8):
        result=train(model,tok,suite['train'],manifest,out/'run',settings,objective,limits,
             stop_after=step,resume=point,experiment_id='stage3-software-proof',run_id='resume-proof')
        point=Path(result['checkpoint'])
        saved=out/f'attempt-{result["state"]["attempt_index"]}.json'
        saved.write_bytes(Path(result['snapshot']).read_bytes());snapshots.append(saved)
    # Same seed/config/data, uninterrupted control: exact CPU parameter continuation.
    control=DecisionEngine(BackboneConfig());control.load_state_dict(initial)
    train(control,tok,suite['train'],manifest,out/'control',settings,objective,limits,
          experiment_id='stage3-software-proof',run_id='uninterrupted-control')
    for name,tensor in model.state_dict().items():torch.testing.assert_close(tensor,control.state_dict()[name],atol=0,rtol=0)
    live=out/'collector-input.json';live.write_bytes(snapshots[0].read_bytes())
    register(out/'registry.json',live,'local-proof')
    cache=out/'isolated-cache.json';atomic_json(cache,merge_snapshots({},collect_snapshots(out/'registry.json')))
    # Polling missed the entirety of attempt 1. The persisted cache anchors attempt 0.
    live.write_bytes(snapshots[2].read_bytes())
    atomic_json(cache,merge_snapshots(json.loads(cache.read_text()),collect_snapshots(out/'registry.json')))
    live.unlink()
    view=presentation(json.loads(cache.read_text()))
    assert view['runs'][0]['attempt_index']==2 and view['runs'][0]['verification']=='verified'
    calibrated=out/'run'/'calibrated.pt'
    with (out/'calibration-cli.log').open('w') as log,contextlib.redirect_stdout(log):
        cli(['calibrate','--checkpoint',str(point),'--suite',str(out/'suite'),'--out',str(calibrated)])
    fitted,tokenizer,payload=load_checkpoint(calibrated)
    proof=verify_exposure(calibrated,expected_sha256=payload['checkpoint_sha256'])
    assert proof['status']=='verified' and proof['native_32k_64k'] is False
    assert payload['training_state'] is None
    observed=[]
    def capture(module,args,kwargs):
        ids=args[0];observed.append({'batch_size':int(ids.shape[0]) if torch.is_tensor(ids) else 1,
            'tokens_per_row':int(ids.shape[1]) if torch.is_tensor(ids) else len(ids)})
    hook=fitted.backbone.register_forward_pre_hook(capture,with_kwargs=True)
    encoding=encode_request(make_request(16,'uniform'),tokenizer,limits)
    with torch.inference_mode():logits,usage=fitted(encoding)
    hook.remove()
    assert len(observed)==2 and observed[0]['batch_size']==1 and observed[1]['batch_size']==16
    report=evaluate(fitted,suite['development'],tokenizer,limits,split='development')
    atomic_json(out/'software-evaluation.json',report)
    writer=TelemetryWriter(Path(result['snapshot']),json.loads(Path(result['snapshot']).read_text()))
    writer.update(artifacts=writer.snapshot['artifacts']+[artifact(out/'software-evaluation.json','evaluation')])
    final={'classification':'actual CPU miniature software exercise; adapter/provider only, not full Overwatch',
           'elapsed_seconds':time.perf_counter()-started,'hardware':hardware_identity(torch.device('cpu')),
           'torch':torch.__version__,'dtype':'fp32','checkpoint':str(calibrated),'checkpoint_sha256':payload['checkpoint_sha256'],
           'same_policy_resume_exact':True,'observed_backbone_calls':observed,'execution':usage,
           'missed_attempt_recovery':view,'exposure':proof,'inference_only_after_calibration':True,
           'source_provenance':payload['source_provenance'],'config':json.loads((out/'run'/'config.json').read_text()),
           'native_context':'unverified','full_overwatch':'unverified',
           'quality_note':'No threshold or predictive-quality claim is attached to this software exercise; the fixed counterfactual experiment is separate.'}
    atomic_json(out/'report.json',final)
    print(json.dumps({k:final[k] for k in ('elapsed_seconds','same_policy_resume_exact','observed_backbone_calls','inference_only_after_calibration')},indent=2))
if __name__=='__main__':main()
