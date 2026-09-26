"""Native-only full-context acceptance harness. Never substitutes a fixture backbone.

Execution evidence and task quality are separate. This module intentionally fails
closed until an actual pinned-backbone checkpoint and sufficient training exposure
exist. Constructed marker cases are diagnostic, not an external quality benchmark.
"""
from __future__ import annotations
from dataclasses import asdict
import copy
from pathlib import Path
import time
import torch
from .checkpoint import load_checkpoint
from .encoding import ByteTokenizer, Limits, encode_request
from .io import atomic_json
from .schema import SystemOneRequest

PIN = '060db6499f32faf8b98477b0a26969ef7d8b9987'
PROTOCOL = {
    'version':'native-context-v2', 'required_branch_tokens':32768, 'required_aggregate_tokens':65536,
    'positions':['beginning','middle','end'], 'languages':['en','es'], 'choice_options':255,
    'score_levels':10, 'fp32_logit_atol':1e-5, 'fp32_logit_rtol':1e-5, 'probability_atol':1e-5, 'probability_rtol':1e-5,
    'gradient_atol':2e-5, 'gradient_rtol':2e-5, 'hf_hidden_atol':1e-4, 'hf_hidden_rtol':1e-4,
    'evidence_marker_accuracy_minimum':0.8,
    'training_exposure_required':True,
    'interpretation':'Marker diagnostics and source-kernel parity; not Jev-relative or general quality evidence'
}


def _fit_exact(make, measure, target, max_units=131072):
    """Size generated filler, never truncate submitted data; reject an unrepresentable boundary."""
    low, high = 0, max_units
    while low < high:
        mid = (low + high + 1) // 2
        if measure(make(mid)) <= target: low = mid
        else: high = mid - 1
    # BPE token counts can change at a join. Require actual count, not a guessed ratio.
    for n in range(max(0,low-16),min(max_units,low+16)+1):
        obj = make(n)
        if measure(obj) == target: return obj
    raise ValueError('could not construct exact serialized token boundary with the frozen filler')


def make_case(tokenizer, position: str, language: str) -> SystemOneRequest:
    if position not in PROTOCOL['positions'] or language not in PROTOCOL['languages']:
        raise ValueError('unknown diagnostic stratum')
    evidence = ('AUTHORITATIVE FACT: route=billing; urgency=7; escalate=true.' if language=='en'
                else 'HECHO AUTORIZADO: ruta=billing; urgencia=7; escalar=true.')
    definitions={
        'route': {'type':'choice','instructions':'Return the authoritative route, not any distractor.',
                  'criteria':{'billing':'Billing', **{f'option-{i:03d}':f'Unrelated department {i}' for i in range(254)}}},
        'urgency': {'type':'score','instructions':'Read the authoritative urgency level.', 'criteria':[str(i) for i in range(10)]},
        'escalate': {'type':'noul','instructions':'Read whether the authoritative fact says escalate=true.',
                     'criteria':{'false':'escalate=false','true':'escalate=true'}}}
    def make_state(n):
        # Construct a diagnostic, not a transformation of user input. Evidence stays at its declared position.
        filler = ' Ignore irrelevant data.' * 2000 + ' x' * n
        if position == 'beginning': return evidence + '\n' + filler
        if position == 'end': return filler + '\n' + evidence
        middle = len(filler) // 2
        return filler[:middle] + '\n' + evidence + '\n' + filler[middle:]
    relaxed=Limits(262144,524288,1024)
    def req(state,questions=definitions):
        return SystemOneRequest(state=state,questions=questions,model='kev-laya-preview')
    def longest(r):
        e=encode_request(r,tokenizer,relaxed)
        return len(e.state)+max(len(b.ids) for b in e.branches)
    request=_fit_exact(lambda n:req(make_state(n)),longest,32768)
    encoded=encode_request(request,tokenizer,Limits())
    questions=copy.deepcopy(definitions)
    # Replicate independent mixed questions, each with a complete shared prefix. IDs are metadata.
    counter=0
    while True:
        kind=('route','urgency','escalate')[counter%3]
        candidate=questions | {f'{kind}-{counter}':copy.deepcopy(definitions[kind])}
        new=req(request.state,candidate)
        count=encode_request(new,tokenizer,relaxed).logical_tokens
        if count>=64000:break
        questions=candidate;counter+=1
    current=req(request.state,questions)
    # Add independently scheduled branches until the aggregate can reach exactly 65,536.
    while True:
        key=f'budget-{counter}';counter+=1
        base_question={'type':'noul','instructions':'Only the authoritative fact determines this answer.','criteria':{'true':'escalate=true','false':'escalate=false'}}
        base_req=req(request.state,questions|{key:base_question})
        base_enc=encode_request(base_req,tokenizer,relaxed)
        if base_enc.logical_tokens>65536:
            raise ValueError('diagnostic base branch overshoots aggregate; protocol construction needs review')
        capacity=32768-len(base_enc.state)-len(base_enc.branches[-1].ids)
        remaining=65536-base_enc.logical_tokens
        desired=min(capacity,remaining)
        def make(n):
            q=base_question|{'instructions':base_question['instructions']+' x'*n}
            return req(request.state,questions|{key:q})
        expanded=_fit_exact(make,lambda r:encode_request(r,tokenizer,relaxed).logical_tokens,
                            base_enc.logical_tokens+desired)
        questions={k:v.model_dump() for k,v in expanded.questions.items()}
        if desired==remaining:
            check=encode_request(expanded,tokenizer,Limits())
            assert check.logical_tokens==65536
            assert max(len(check.state)+len(b.ids) for b in check.branches)==32768
            return expanded


def merged_backbone(model):
    """Map the independent native backbone to the pinned HF oracle, merging LoRA."""
    weights={}
    state=model.backbone.state_dict()
    for key,value in state.items():
        if key.endswith(('.a','.b')):continue
        if '.base.' in key:
            plain=key.replace('.base.','.')
            value=value.detach().clone()
            if key.endswith('.weight'):
                parent=key.rsplit('.base.',1)[0]
                value=value+(state[parent+'.b']@state[parent+'.a'])*(model.cfg.lora_alpha/model.cfg.lora_rank)
            weights[plain]=value
        else: weights[key]=value
    return weights


def hf_parity(model, ids):
    from transformers import Qwen2Config, Qwen2Model
    fields=asdict(model.cfg)
    keep={k:v for k,v in fields.items() if k in {
        'vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads',
        'num_key_value_heads','max_position_embeddings','rope_theta','rms_norm_eps'}}
    config=Qwen2Config(**keep,hidden_act='silu',use_sliding_window=False,attention_dropout=0.0)
    config._attn_implementation='eager'
    oracle=Qwen2Model(config).to(next(model.parameters()).device)
    oracle.load_state_dict(merged_backbone(model),strict=True)
    oracle.eval()
    with torch.no_grad():
        a,_=model.backbone(ids)
        b=oracle(input_ids=torch.tensor(ids,device=a.device)[None],use_cache=False).last_hidden_state[0]
    result={'maximum_hidden_absolute_error':float((a-b).abs().max()),'oracle':'transformers==4.57.1/Qwen2Model/eager'}
    result['passed']=torch.allclose(a,b,atol=1e-4,rtol=1e-4)
    del oracle
    return result


def run(checkpoint: Path, output: Path, device='cuda', precision='fp32') -> dict:
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('native report destination must be empty')
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'protocol.json',PROTOCOL)  # Freeze before examining outputs.
    if precision not in {'fp32', 'bf16'}:
        raise ValueError('supported measurement precisions are fp32 and bf16')
    model,tokenizer,point=load_checkpoint(checkpoint,device)
    if isinstance(tokenizer,ByteTokenizer) or not model.native_weights_loaded or model.cfg.revision!=PIN:
        raise ValueError('native validation requires actual pinned Qwen weights and tokenizer; no tiny fallback')
    from .encoding import NATIVE_SERIALIZATION
    if tokenizer.serialization != NATIVE_SERIALIZATION:
        raise ValueError('native v2 acceptance requires explicitly versioned lossless serialization, not legacy preprocessing')
    if model.training_steps<1:raise ValueError('a trained pointer head is required')
    if model.calibration_provenance.get('status') != 'fitted-held-out':
        raise ValueError('native acceptance requires the held-out calibrated artifact')
    if not torch.cuda.is_available() or device=='cpu':
        raise ValueError('this long-context memory profile requires CUDA; no claim is made for CPU feasibility')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('this device does not report BF16 support')
    model.eval();results=[]
    first=SystemOneRequest(state='An ordinary small reference state.',questions={'q':{'type':'noul','instructions':'Is this a short state?'}},model='kev-laya-preview')
    e=encode_request(first,tokenizer,Limits())
    parity=hf_parity(model,e.state+e.branches[0].ids)
    for language in PROTOCOL['languages']:
        for position in PROTOCOL['positions']:
            request=make_case(tokenizer,position,language)
            encoded=encode_request(request,tokenizer,Limits())
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
                logits,usage=model(encoded)
                torch.cuda.synchronize()
                optimized_seconds=time.perf_counter()-start
                optimized_peak=torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); ref_start=time.perf_counter()
                reference,_=model(encoded,reference=True)
                torch.cuda.synchronize()
                reference_seconds=time.perf_counter()-ref_start
                reference_peak=torch.cuda.max_memory_allocated()
                max_delta=max(float((a-b).abs().max()) for a,b in zip(logits,reference))
                numeric=all(torch.allclose(a,b,atol=1e-5,rtol=1e-5) for a,b in zip(logits,reference))
                probability_error=max(float(((a.double()/model.temperatures[branch.question.type]).softmax(-1)-
                                             (b.double()/model.temperatures[branch.question.type]).softmax(-1)).abs().max())
                                      for a,b,branch in zip(logits,reference,encoded.branches))
                numeric &= all(torch.allclose((a.double()/model.temperatures[branch.question.type]).softmax(-1),
                                               (b.double()/model.temperatures[branch.question.type]).softmax(-1),atol=1e-5,rtol=1e-5)
                               for a,b,branch in zip(logits,reference,encoded.branches))
            probe=[]
            for branch,z in zip(encoded.branches,logits):
                if branch.question_id not in {'route','urgency','escalate'}:continue
                labels=[k for k,_ in branch.question.options()]
                expected={'route':'billing','urgency':'7','escalate':'true'}[branch.question_id]
                predicted=labels[z.argmax().item()]
                probe.append({'question_id':branch.question_id,'correct':predicted==expected,'predicted':predicted,'expected':expected,
                              'probabilities':(z.double()/model.temperatures[branch.question.type]).softmax(-1).tolist()})
            overflow=False
            try: encode_request(request,tokenizer,Limits(32768,65535))
            except ValueError:overflow=True
            record={'language':language,'position':position,'state_tokens':len(encoded.state),'branch_lengths':[len(b.ids) for b in encoded.branches],
                    'logical_tokens':encoded.logical_tokens, 'forward_tokens':usage['forward_tokens'],
                    'latency_seconds':optimized_seconds,'cuda_peak_allocated_bytes':optimized_peak,
                    'reference_maximum_logit_error':max_delta, 'reference_maximum_probability_error':probability_error,
                    'numeric_tolerance_passed':bool(numeric),'reference_latency_seconds':reference_seconds,
                    'reference_cuda_peak_allocated_bytes':reference_peak, 'execution':usage,
                    'precision':precision, 'parameter_dtype':str(next(model.parameters()).dtype),
                    'overflow_rejected':overflow,'decisions':probe}
            atomic_json(output/f'{language}-{position}.request.json',request.model_dump())
            atomic_json(output/f'{language}-{position}.result.json',record);results.append(record)
    from .exposure import verify_exposure
    exposure_evidence=verify_exposure(checkpoint,expected_sha256=point['checkpoint_sha256'])
    training_exposure=exposure_evidence.get('native_32k_64k',False)
    accuracy=sum(p['correct'] for r in results for p in r['decisions'])/sum(len(r['decisions']) for r in results)
    report={'protocol':PROTOCOL,'precision':precision, 'batch_policy':model.batch_policy.to_dict(),
            'bf16_note':'Measured against unchanged strict FP32 tolerances; no automatic tolerance relaxation','checkpoint_sha256':point['checkpoint_sha256'],'hf_parity':parity,'results':results,
            'training_exposure_verified':training_exposure,'training_exposure_evidence':exposure_evidence,'marker_accuracy':accuracy,
            'passed':bool(parity['passed'] and training_exposure and accuracy>=.8 and all(r['numeric_tolerance_passed'] and r['overflow_rejected'] for r in results)),
            'cost_usd':None,'cost_status':'requires deployment billing evidence','general_quality':'unverified','jev_relative_quality':'unverified'}
    atomic_json(output/'report.json',report)
    return report
