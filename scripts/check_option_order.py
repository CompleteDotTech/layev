"""Explicit option-order diagnostic; separate from question isolation and quality claims."""
import argparse,copy,json
from pathlib import Path
import torch
from kev_laya.checkpoint import load_checkpoint
from kev_laya.cli import limits_for
from kev_laya.data import load_suite
from kev_laya.service import InferenceRuntime
from kev_laya.io import atomic_json
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--suite',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
a=p.parse_args();torch.set_num_threads(2)
model,tok,point=load_checkpoint(a.checkpoint);runtime=InferenceRuntime(model,tok,point['model_id'],limits_for(point))
suite,manifest=load_suite(a.suite);rows=[]
for datum in suite['test']:
    req=datum.request.model_copy(update={'model':'kev-laya-preview'})
    base,_=runtime.predict_encoded(runtime.encode(req))
    qs={key:q.model_copy(update={'criteria':dict(reversed(list(q.criteria.items())))}) if q.type=='choice' else q for key,q in req.questions.items()}
    alternate=req.model_copy(update={'questions':qs})
    permuted,_=runtime.predict_encoded(runtime.encode(alternate))
    for (key,q),target in zip(req.questions.items(),datum.targets):
        if q.type!='choice':continue
        one,two=base.answers[key],permuted.answers[key]
        labels=[k for k,_ in q.options()];gold=labels[max(range(len(target)),key=target.__getitem__)]
        rows.append({'record_id':datum.meta['id'],'question_id':key,'original':one.choice,'reversed':two.choice,
                     'flipped':one.choice!=two.choice,'gold':gold,'original_correct':one.choice==gold,'reversed_correct':two.choice==gold,
                     'total_variation':sum(abs(one.probabilities[k]-two.probabilities[k]) for k in labels)/2})
report={'checkpoint_sha256':point['checkpoint_sha256'],'evidence_class':'tiny-fixture engineering diagnostic',
        'count':len(rows),'flip_rate':sum(r['flipped'] for r in rows)/len(rows),
        'original_accuracy':sum(r['original_correct'] for r in rows)/len(rows),
        'reversed_accuracy':sum(r['reversed_correct'] for r in rows)/len(rows),
        'mean_total_variation':sum(r['total_variation'] for r in rows)/len(rows),'rows':rows}
atomic_json(a.out,report);print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
