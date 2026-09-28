"""Native-only full-context acceptance harness. Never substitutes a fixture backbone.

Execution evidence and task quality are separate. This module intentionally fails
closed until an actual pinned-backbone checkpoint and sufficient training exposure
exist. Constructed marker cases are diagnostic, not an external quality benchmark.
"""
from __future__ import annotations
from dataclasses import asdict
from importlib import metadata
import math
import copy
from pathlib import Path
import time
import torch
from .checkpoint import load_checkpoint
from .encoding import ByteTokenizer, ContextOverflow, Limits, encode_request
from .io import atomic_json
from .schema import SystemOneRequest

PIN = '060db6499f32faf8b98477b0a26969ef7d8b9987'
REFERENCE_VERSION = '4.57.1'
PROTOCOL = {
    'version':'native-context-v3', 'required_branch_tokens':32768, 'required_aggregate_tokens':65536,
    'positions':['beginning','middle','end'], 'languages':['en','es'], 'choice_options':255,
    'score_levels':10, 'fp32_logit_atol':1e-5, 'fp32_logit_rtol':1e-5, 'probability_atol':1e-5, 'probability_rtol':1e-5,
    'gradient_atol':2e-5, 'gradient_rtol':2e-5, 'hf_hidden_atol':1e-4, 'hf_hidden_rtol':1e-4,
    'evidence_marker_accuracy_minimum':0.8,
    'training_exposure_required':True,
    'interpretation':'Marker diagnostics and source-kernel parity; not Jev-relative or general quality evidence'
}


def require_reference_version() -> str:
    """Check the installed distribution; a hard-coded receipt is not version proof."""
    try:
        version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        raise ValueError("native reference requires transformers==" + REFERENCE_VERSION) from None
    if version != REFERENCE_VERSION:
        raise ValueError("native reference requires transformers==" + REFERENCE_VERSION)
    return version


def measurement_device(device, precision: str) -> torch.device:
    """Resolve one CUDA device before model loading; never fall back to CPU."""
    if precision not in {"fp32", "bf16"}:
        raise ValueError("supported measurement precisions are fp32 and bf16")
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("native validation requires CUDA and actual pinned weights; no tiny fallback")
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device is unavailable")
    selected = torch.device("cuda", index)
    # is_bf16_supported examines the current device, unlike APIs with a device arg.
    if precision == "bf16":
        with torch.cuda.device(selected):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("the selected CUDA device does not report BF16 support")
    return selected


def verify_context_boundaries(request, tokenizer, encoded) -> dict:
    """Prove exact logical lengths and BOTH limit rejections, not any ValueError.

    This tests a fixed complete request against ceilings one token below its
    measured size. It never truncates or mutates caller content. Unrelated errors
    propagate and cannot become a passing native acceptance receipt.
    """
    branch_limit = PROTOCOL["required_branch_tokens"]
    aggregate_limit = PROTOCOL["required_aggregate_tokens"]
    lengths = [len(encoded.state) + len(b.ids) for b in encoded.branches]
    if (not lengths or max(lengths) != branch_limit
            or encoded.logical_tokens != aggregate_limit
            or len(encoded.state) + sum(len(b.ids) for b in encoded.branches) != aggregate_limit):
        raise ValueError("native context requires exact branch and aggregate accounting")
    if tuple(b.question_id for b in encoded.branches) != tuple(request.questions):
        raise ValueError("native context encoding must cover every requested question in order")
    if len(encoded.state) >= branch_limit - 1:
        raise ValueError("native context diagnostic has no valid branch headroom")
    failing_branch = next(b for b, n in zip(encoded.branches, lengths, strict=True)
                          if n > branch_limit - 1)
    cases = (
        ("aggregate_input", Limits(branch_limit, aggregate_limit - 1),
         aggregate_limit, aggregate_limit - 1, None),
        ("state_plus_branch", Limits(branch_limit - 1, aggregate_limit),
         branch_limit, branch_limit - 1, failing_branch.question_id),
    )
    evidence = {}
    for name, limits, actual, maximum, question_id in cases:
        expected = {"code": "context_overflow", "limit": name, "actual": actual,
                    "maximum": maximum, "question_id": question_id}
        try:
            encode_request(request, tokenizer, limits)
        except ContextOverflow as exc:
            if exc.detail != expected:
                raise ValueError("unexpected native context overflow detail") from exc
            evidence[name] = dict(exc.detail)
        else:
            raise ValueError("native context overflow was not rejected: " + name)
    return evidence


def compare_branch_outputs(encoded, logits, reference, temperatures) -> dict:
    """Compare every branch and option without zip truncation or broadcasting."""
    count = len(encoded.branches)
    if (not count or not isinstance(logits, (list, tuple))
            or not isinstance(reference, (list, tuple))
            or len(logits) != count or len(reference) != count):
        raise ValueError("native comparison requires one output per branch on both paths")
    maximum_logit_error = maximum_probability_error = 0.0
    passed = True
    for branch, a, b in zip(encoded.branches, logits, reference, strict=True):
        shape = (len(branch.question.options()),)
        if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                or a.shape != shape or b.shape != shape
                or a.dtype != b.dtype or a.device != b.device
                or not a.is_floating_point() or not b.is_floating_point()):
            raise ValueError("native comparison requires matching complete option vectors")
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("native comparison requires finite outputs")
        temperature = temperatures.get(branch.question.type)
        if (type(temperature) not in (int, float) or not math.isfinite(temperature)
                or temperature <= 0):
            raise ValueError("native comparison requires finite positive calibration temperatures")
        pa = (a.double() / temperature).softmax(-1)
        pb = (b.double() / temperature).softmax(-1)
        if not bool(torch.isfinite(pa).all() and torch.isfinite(pb).all()):
            raise ValueError("native comparison requires finite probabilities")
        maximum_logit_error = max(maximum_logit_error, float((a - b).abs().max()))
        maximum_probability_error = max(maximum_probability_error, float((pa - pb).abs().max()))
        passed &= bool(torch.allclose(a, b, atol=PROTOCOL["fp32_logit_atol"],
                                      rtol=PROTOCOL["fp32_logit_rtol"]))
        passed &= bool(torch.allclose(pa, pb, atol=PROTOCOL["probability_atol"],
                                      rtol=PROTOCOL["probability_rtol"]))
    return {"maximum_logit_error": maximum_logit_error,
            "maximum_probability_error": maximum_probability_error, "passed": passed}


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
        # Split complete filler units. Splitting the raw string midpoint can
        # bisect a BPE unit and skip an otherwise reachable exact boundary.
        prefix = ' Ignore irrelevant data.' * 1000 + ' x' * (n // 2)
        suffix = ' Ignore irrelevant data.' * 1000 + ' x' * (n - n // 2)
        return prefix + '\n' + evidence + '\n' + suffix
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


def merged_backbone(model, *, merge_lora=True):
    """Map native backbone weights to pinned HF names; optionally merge LoRA."""
    weights={}
    state=model.backbone.state_dict()
    for key,value in state.items():
        if key.endswith(('.a','.b')):continue
        if '.base.' in key:
            plain=key.replace('.base.','.')
            value=value.detach().clone()
            if merge_lora and key.endswith('.weight'):
                parent=key.rsplit('.base.',1)[0]
                value=value+(state[parent+'.b']@state[parent+'.a'])*(model.cfg.lora_alpha/model.cfg.lora_rank)
            weights[plain]=value
        else: weights[key]=value
    return weights


class ReferenceLoRA(torch.nn.Module):
    """Independent Qwen oracle adapter using the deployed two-linear arithmetic."""

    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base = base
        self.register_buffer('a', a.detach().clone())
        self.register_buffer('b', b.detach().clone())
        self.scale = scale

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.a), self.b) * self.scale


def hf_parity(model, ids):
    actual_version = require_reference_version()
    import transformers
    if transformers.__version__ != actual_version:
        raise ValueError("imported Transformers disagrees with the pinned distribution")
    from transformers import Qwen2Config, Qwen2Model
    fields=asdict(model.cfg)
    keep={k:v for k,v in fields.items() if k in {
        'vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads',
        'num_key_value_heads','max_position_embeddings','rope_theta','rms_norm_eps'}}
    config=Qwen2Config(**keep,hidden_act='silu',use_sliding_window=False,attention_dropout=0.0)
    config._attn_implementation='eager'
    oracle=Qwen2Model(config).to(next(model.parameters()).device)
    # A merged matrix changes FP32 reduction order after LoRA is trained.
    # Keep HF's independent Qwen backbone and apply the same adapter operation
    # order as deployed inference, without changing training or model weights.
    oracle.load_state_dict(merged_backbone(model, merge_lora=False),strict=True)
    if model.cfg.lora_rank:
        for native_layer, reference_layer in zip(model.backbone.layers, oracle.layers, strict=True):
            for name in ('q_proj', 'v_proj'):
                native = getattr(native_layer.self_attn, name)
                reference = getattr(reference_layer.self_attn, name)
                setattr(reference_layer.self_attn, name, ReferenceLoRA(
                    reference, native.a, native.b, native.scale))
    oracle.eval()
    with torch.no_grad():
        a,_=model.backbone(ids)
        b=oracle(input_ids=torch.tensor(ids,device=a.device)[None],use_cache=False).last_hidden_state[0]
    expected_shape = (len(ids), model.cfg.hidden_size)
    if a.shape != expected_shape or b.shape != expected_shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("native oracle requires matching complete hidden-state tensors")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("native oracle requires finite hidden states")
    result={'maximum_hidden_absolute_error':float((a-b).abs().max()),
            'oracle':'transformers==' + actual_version + '/Qwen2Model/eager' +
                     ('+two-linear-LoRA' if model.cfg.lora_rank else ''),
            'transformers_version': actual_version}
    result['passed']=torch.allclose(a,b,atol=PROTOCOL['hf_hidden_atol'],rtol=PROTOCOL['hf_hidden_rtol'])
    del oracle
    return result


def run(checkpoint: Path, output: Path, device='cuda', precision='fp32') -> dict:
    device = measurement_device(device, precision)
    require_reference_version()
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('native report destination must be empty')
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'protocol.json',PROTOCOL)  # Freeze before examining outputs.
    model,tokenizer,point=load_checkpoint(checkpoint,device)
    if isinstance(tokenizer,ByteTokenizer) or not model.native_weights_loaded or model.cfg.revision!=PIN:
        raise ValueError('native validation requires actual pinned Qwen weights and tokenizer; no tiny fallback')
    from .encoding import NATIVE_SERIALIZATION
    if tokenizer.serialization != NATIVE_SERIALIZATION:
        raise ValueError('native v2 acceptance requires explicitly versioned lossless serialization, not legacy preprocessing')
    if model.training_steps<1:raise ValueError('a trained pointer head is required')
    if model.calibration_provenance.get('status') != 'fitted-held-out':
        raise ValueError('native acceptance requires the held-out calibrated artifact')
    if next(model.parameters()).device != device:
        raise ValueError('native model was not loaded on the selected CUDA measurement device')
    model.eval();results=[]
    first=SystemOneRequest(state='An ordinary small reference state.',questions={'q':{'type':'noul','instructions':'Is this a short state?'}},model='kev-laya-preview')
    e=encode_request(first,tokenizer,Limits())
    parity=hf_parity(model,e.state+e.branches[0].ids)
    for language in PROTOCOL['languages']:
        for position in PROTOCOL['positions']:
            request=make_case(tokenizer,position,language)
            encoded=encode_request(request,tokenizer,Limits())
            boundary_evidence = verify_context_boundaries(request, tokenizer, encoded)
            torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);start=time.perf_counter()
            # Autocast's constructor checks BF16 support on the current CUDA
            # device. Enter the selected device before constructing it and
            # restore the caller's current device even if construction fails.
            with torch.cuda.device(device), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
                logits,usage=model(encoded)
                torch.cuda.synchronize(device)
                optimized_seconds=time.perf_counter()-start
                optimized_peak=torch.cuda.max_memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); ref_start=time.perf_counter()
                reference,_=model(encoded,reference=True)
                torch.cuda.synchronize(device)
                reference_seconds=time.perf_counter()-ref_start
                reference_peak=torch.cuda.max_memory_allocated(device)
                comparison = compare_branch_outputs(encoded, logits, reference, model.temperatures)
                max_delta = comparison['maximum_logit_error']
                probability_error = comparison['maximum_probability_error']
                numeric = comparison['passed']
            probe=[]
            for branch,z in zip(encoded.branches,logits,strict=True):
                if branch.question_id not in {'route','urgency','escalate'}:continue
                labels=[k for k,_ in branch.question.options()]
                expected={'route':'billing','urgency':'7','escalate':'true'}[branch.question_id]
                predicted=labels[z.argmax().item()]
                probe.append({'question_id':branch.question_id,'correct':predicted==expected,'predicted':predicted,'expected':expected,
                              'probabilities':(z.double()/model.temperatures[branch.question.type]).softmax(-1).tolist()})
            record={'language':language,'position':position,'state_tokens':len(encoded.state),'branch_lengths':[len(b.ids) for b in encoded.branches],
                    'logical_tokens':encoded.logical_tokens, 'forward_tokens':usage['forward_tokens'],
                    'latency_seconds':optimized_seconds,'cuda_peak_allocated_bytes':optimized_peak,
                    'reference_maximum_logit_error':max_delta, 'reference_maximum_probability_error':probability_error,
                    'numeric_tolerance_passed':bool(numeric),'reference_latency_seconds':reference_seconds,
                    'reference_cuda_peak_allocated_bytes':reference_peak, 'execution':usage,
                    'precision':precision, 'parameter_dtype':str(next(model.parameters()).dtype),
                    'overflow_rejected':True,'overflow_evidence':boundary_evidence,'measurement_device':str(device),'decisions':probe}
            atomic_json(output/f'{language}-{position}.request.json',request.model_dump())
            atomic_json(output/f'{language}-{position}.result.json',record);results.append(record)
    from .exposure import verify_exposure
    exposure_evidence=verify_exposure(checkpoint,expected_sha256=point['checkpoint_sha256'])
    training_exposure=exposure_evidence.get('native_32k_64k',False)
    accuracy=sum(p['correct'] for r in results for p in r['decisions'])/sum(len(r['decisions']) for r in results)
    report={'protocol':PROTOCOL,'precision':precision, 'batch_policy':model.batch_policy.to_dict(),
            'bf16_note':'Measured against unchanged strict FP32 tolerances; no automatic tolerance relaxation','checkpoint_sha256':point['checkpoint_sha256'],'hf_parity':parity,'results':results,
            'training_exposure_verified':training_exposure,'training_exposure_evidence':exposure_evidence,'marker_accuracy':accuracy,
            'passed':bool(parity['passed'] and training_exposure and accuracy>=PROTOCOL['evidence_marker_accuracy_minimum'] and all(r['numeric_tolerance_passed'] and r['overflow_rejected'] for r in results)),
            'cost_usd':None,'cost_status':'requires deployment billing evidence','general_quality':'unverified','jev_relative_quality':'unverified'}
    atomic_json(output/'report.json',report)
    return report
