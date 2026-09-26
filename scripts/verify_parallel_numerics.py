"""Record FP32 gradient/logit differences against independent full-row execution."""
import argparse
import json
from pathlib import Path
import torch
from kev_laya.encoding import ByteTokenizer, Limits, encode_request
from kev_laya.execution import BatchPolicy
from kev_laya.model import BackboneConfig, DecisionEngine
from kev_laya.objectives import objective, ObjectiveConfig
from benchmark_parallel import make_request


def measure(a,b,tolerance):
    delta=(a-b).abs()
    return {'max_absolute':delta.max().item(),
            'max_tolerance_ratio':(delta/(tolerance+tolerance*b.abs())).max().item()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2)
    encoding=encode_request(make_request(7,'ragged'),ByteTokenizer(),Limits(2048,65536))
    rows=[]
    for rank in (0,2):
        for checkpointing in (False,True):
            for kv_heads in (1,2,4):
                for cap in (2,16):
                    torch.manual_seed(913)
                    model=DecisionEngine(BackboneConfig(hidden_size=32,intermediate_size=64,pointer_dim=16,
                        lora_rank=rank,activation_checkpointing=checkpointing,num_key_value_heads=kv_heads,
                        max_position_embeddings=2048),BatchPolicy(max_branches=cap))
                    if rank:
                        with torch.no_grad():
                            for n,p in model.named_parameters():
                                if n.endswith('.b'):p.normal_(std=.02)
                    types=[b.question.type for b in encoding.branches]
                    targets=[]
                    for b in encoding.branches:
                        y=[1/len(b.option_ends)]*len(b.option_ends);y[0]+=.1;y[-1]-=.1;targets.append(y)
                    z,stats=model(encoding)
                    objective(z,targets,types,ObjectiveConfig(ordinal=.1))[0].backward()
                    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
                    model.zero_grad(set_to_none=True)
                    ref,_=model(encoding,reference=True)
                    objective(ref,targets,types,ObjectiveConfig(ordinal=.1))[0].backward()
                    errors={n:measure(grads[n],p.grad,2e-5) for n,p in model.named_parameters() if p.requires_grad}
                    for n,p in model.named_parameters():
                        if p.requires_grad:torch.testing.assert_close(grads[n],p.grad,atol=2e-5,rtol=2e-5)
                    for x,y in zip(z,ref):
                        torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
                        torch.testing.assert_close(x.softmax(-1),y.softmax(-1),atol=1e-5,rtol=1e-5)
                    rows.append({'lora_rank':rank,'activation_checkpointing':checkpointing,'kv_heads':kv_heads,
                                 'max_branches':cap,'execution':stats,
                                 'logits':measure(torch.cat(z),torch.cat(ref),1e-5),
                                 'probabilities':measure(torch.cat([x.softmax(-1) for x in z]),torch.cat([x.softmax(-1) for x in ref]),1e-5),
                                 'gradients':errors})
    out={'evidence_class':'CPU-FP32-fixture','configurations':rows,
         'max_logit_absolute_difference':max(r['logits']['max_absolute'] for r in rows),
         'max_probability_absolute_difference':max(r['probabilities']['max_absolute'] for r in rows),
         'max_gradient_absolute_difference':max(e['max_absolute'] for r in rows for e in r['gradients'].values()),
         'max_gradient_tolerance_ratio':max(e['max_tolerance_ratio'] for r in rows for e in r['gradients'].values()),
         'passed':True}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='configurations'},indent=2))
if __name__=='__main__':main()
