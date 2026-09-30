"""Real CPU tensor-batch tests, not tokenizer/HTTP concurrency substitutes.

Logits/probabilities: atol=rtol=1e-5. Gradients: atol=rtol=2e-5.
These are the task's frozen tolerances, not fitted to observed deltas.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import weakref
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from kev_laya.checkpoint import load_checkpoint, save_checkpoint
from kev_laya.encoding import ByteTokenizer, Limits, encode_request
from kev_laya.execution import BatchBudgetExceeded, BatchPolicy, plan_batches
from kev_laya.model import BackboneConfig, DecisionEngine, PointerHead, RMSNorm
from kev_laya.native_gradients import CanonicalMediumAttention
from kev_laya.objectives import ObjectiveConfig, objective
from kev_laya.schema import SystemOneRequest
from kev_laya.service import InferenceRuntime, ServeSettings, create_app
from kev_laya.telemetry_contract import validate_snapshot
from kev_laya.training import TrainSettings, train


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for BF16 branch graph lifetime')
def test_bf16_branch_activations_released_before_optimizer(tmp_path, suite, monkeypatch):
    if not torch.cuda.is_bf16_supported():
        pytest.skip('GPU does not support BF16')
    saved = []

    class Branch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, count):
            activation = value.new_ones(4096)
            saved.append(weakref.ref(activation))
            ctx.save_for_backward(activation)
            return value.expand(count).clone()

        @staticmethod
        def backward(ctx, upstream):
            activation, = ctx.saved_tensors
            return upstream.sum() * activation[0], None

    model = make_model(lora=2, policy=BatchPolicy(max_branches=2)).cuda()
    original_forward = model.forward

    def forward(encoded):
        with torch.no_grad():
            _, usage = original_forward(encoded)
        shared = sum(parameter.sum() for parameter in model.parameters() if parameter.requires_grad)
        logits = [Branch.apply(shared, len(branch.question.options())) for branch in encoded.branches]
        return logits, usage

    monkeypatch.setattr(model, 'forward', forward)
    original_step = torch.optim.AdamW.step
    checked = []

    def step(optimizer, *args, **kwargs):
        assert len(saved) >= 2
        assert all(reference() is None for reference in saved)
        checked.append(True)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    data, manifest = suite
    result = train(model, ByteTokenizer(), data['train'], manifest, tmp_path / 'graph-release',
                   TrainSettings(steps=1, precision='bf16'), ObjectiveConfig(), Limits(512, 8192))
    assert checked == [True]
    assert result['state']['step'] == 1


def request(count=7):
    questions = {}
    for i in range(count):
        kind = ('choice', 'score', 'noul')[i % 3]
        q = {'type': kind, 'instructions': 'read field ' + str(i) + ' x' * (i % 4) * 7}
        if kind == 'choice':
            q['criteria'] = {f'option{j}': None for j in range(2 + i % 4)}
        elif kind == 'score':
            q['criteria'] = ['low', {'meaning': 'medium'}, 'high']
        else:
            q['criteria'] = {'false': 'not present', 'true': 'present'}
        questions[f'q{i}'] = q
    return {'state': {'text': 'preserve the entire state', 'data': [4, 8]},
            'model': 'kev-laya-preview', 'questions': questions}


def encode(data, branch=2048):
    return encode_request(SystemOneRequest(**data), ByteTokenizer(), Limits(branch, 65536))


def make_model(lora=0, checkpointing=False, kv_heads=2, policy=None):
    torch.manual_seed(913)
    model = DecisionEngine(BackboneConfig(hidden_size=32, intermediate_size=64, pointer_dim=16,
                           max_position_embeddings=2048, lora_rank=lora,
                           activation_checkpointing=checkpointing, num_key_value_heads=kv_heads), policy)
    if lora:
        # Exercise both LoRA matrices; a zero-initialized B hides errors in A's gradient.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith('.b'):
                    param.normal_(std=.02)
    return model


def spy_backbone(model):
    calls = []
    def spy(_module, args, kwargs):
        ids = args[0]
        parent = args[1] if len(args) > 1 else kwargs.get('past')
        calls.append({'batch': ids.shape[0] if isinstance(ids, torch.Tensor) else 1,
                      'length': ids.shape[1] if isinstance(ids, torch.Tensor) else len(ids),
                      'has_prefix': parent is not None, 'parent': parent})
    return calls, model.backbone.register_forward_pre_hook(spy, with_kwargs=True)


def test_actual_multi_question_backbone_pass_and_single_prefix():
    model = make_model()
    encoding = encode(request())
    calls, handle = spy_backbone(model)
    logits, stats = model(encoding)
    handle.remove()
    assert len(logits) == 7
    assert [c['batch'] for c in calls] == [1, 7]
    assert [c['has_prefix'] for c in calls] == [False, True]
    assert stats['prefix_passes'] == stats['branch_passes'] == 1
    assert stats['effective_batch_sizes'] == [7]
    assert stats['padding_tokens'] > 0
    assert stats['compute_tokens'] == stats['forward_tokens'] + stats['padding_tokens']
    assert stats['forward_tokens'] == stats['logical_input_tokens'] == encoding.logical_tokens


@pytest.mark.parametrize('lora', [0, 2])
@pytest.mark.parametrize('checkpointing', [False, True])
@pytest.mark.parametrize('kv_heads', [1, 2, 4])
@pytest.mark.parametrize('max_branches', [2, 16])
def test_parallel_reference_logits_probabilities_and_all_gradients(lora, checkpointing, kv_heads, max_branches):
    model = make_model(lora, checkpointing, kv_heads, BatchPolicy(max_branches=max_branches))
    encoding = encode(request(5))
    targets = [[1 / len(b.option_ends)] * len(b.option_ends) for b in encoding.branches]
    # Nonuniform soft targets exercise the exact same shared objective.
    for y in targets:
        y[0] += .1
        y[-1] -= .1
    config = ObjectiveConfig(ordinal=.1)
    types = [b.question.type for b in encoding.branches]
    zs, _ = model(encoding)
    objective(zs, targets, types, config)[0].backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    ref, _ = model(encoding, reference=True)
    objective(ref, targets, types, config)[0].backward()
    for z, r in zip(zs, ref, strict=True):
        torch.testing.assert_close(z, r, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(z.softmax(-1), r.softmax(-1), atol=1e-5, rtol=1e-5)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            torch.testing.assert_close(grads[name], p.grad, atol=2e-5, rtol=2e-5)
        else:
            assert p.grad is None
    assert any(g.abs().max() > 0 for n, g in grads.items() if n.startswith('backbone.'))


@pytest.mark.parametrize('sort', [False, True])
@pytest.mark.parametrize('cap', [1, 2, 3, 16])
def test_composition_order_ids_and_microbatch_boundaries(sort, cap):
    model = make_model(policy=BatchPolicy(max_branches=cap, sort_by_length=sort)).eval()
    data = request()
    with torch.inference_mode():
        first, stats = model(encode(data))
        reordered = copy.deepcopy(data)
        reordered['questions'] = dict(reversed(list(data['questions'].items())))
        other, _ = model(encode(reordered))
        for a, b in zip(first, reversed(other)):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        extended = copy.deepcopy(data)
        extended['questions']['long new question'] = {'type': 'choice', 'instructions': 'long ' * 170,
                                                      'criteria': {'yes': None, 'no': None}}
        added, _ = model(encode(extended))
        for a, b in zip(first, added):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        renamed = copy.deepcopy(data)
        renamed['questions'] = {f'untrusted-ID-{i}': q for i, q in enumerate(data['questions'].values())}
        changed, _ = model(encode(renamed))
        for a, b in zip(first, changed):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        serial, serial_stats = model(encode(data), serial_reference=True)
        for a, b in zip(first, serial):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        assert stats['forward_tokens'] == serial_stats['forward_tokens']
        assert stats['branch_passes'] == (7 + cap - 1) // cap


def test_padding_values_cannot_affect_real_hidden_or_gradients():
    model = make_model()
    encoding = encode(request(2))
    _, prefix = model.backbone(encoding.state)
    clones = [(k.clone(), v.clone()) for k, v in prefix]
    first, second = encoding.branches
    length = max(len(first.ids), len(second.ids)) + 11
    def rows(pad):
        return torch.tensor([list(b.ids) + [pad] * (length - len(b.ids)) for b in encoding.branches])
    a, _ = model.backbone(rows(250), prefix)
    b, _ = model.backbone(rows(251), prefix)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    for i, branch in enumerate(encoding.branches):
        torch.testing.assert_close(a[i, :len(branch.ids)], b[i, :len(branch.ids)], atol=0, rtol=0)
    sum(a[i, len(br.ids)-1].square().sum() for i, br in enumerate(encoding.branches)).backward()
    assert model.backbone.embed_tokens.weight.grad[250].count_nonzero() == 0
    for kv, old in zip(prefix, clones):
        for x, y in zip(kv, old):
            torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_parent_cache_is_the_same_immutable_object_across_batches():
    model = make_model(policy=BatchPolicy(max_branches=2))
    clones = []
    def capture(_module, args, result):
        if not isinstance(args[0], torch.Tensor):
            clones.extend((k.clone(), v.clone()) for k, v in result[1])
    calls, handle = spy_backbone(model)
    h = model.backbone.register_forward_hook(capture)
    zs, _ = model(encode(request(7)))
    sum(z.square().mean() for z in zs).backward()
    handle.remove(); h.remove()
    parents = [c['parent'] for c in calls if c['has_prefix']]
    assert len(parents) == 4 and all(p is parents[0] for p in parents)
    for kv, old in zip(parents[0], clones):
        for x, y in zip(kv, old):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
            assert x.grad_fn is not None  # not detached from the shared state computation


def test_concurrent_requests_are_isolated():
    model = make_model(policy=BatchPolicy(max_branches=3)).eval()
    def predict(i):
        data = request(3 + i % 4)
        data['state'] = ('unique state ' + str(i)) * (i + 1)
        with torch.inference_mode():
            zs, stats = model(encode(data))
            return [z.clone() for z in zs], stats
    expected = [predict(i) for i in range(8)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(predict, range(8)))
    for (zs, stats), (ref, ref_stats) in zip(actual, expected):
        assert stats == ref_stats
        for a, b in zip(zs, ref):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not hasattr(model, 'last_diagnostics') and not hasattr(model, 'prefix_cache')


@pytest.mark.parametrize('field,value', [('max_branches',0),('max_padded_tokens',0),('max_cache_bytes',0),
                                       ('max_branches',True),('max_padded_tokens',3.5),('sort_by_length',1)])
def test_policy_validates_types_and_positive_bounds(field, value):
    with pytest.raises(ValueError):
        BatchPolicy(**{field: value})


def test_policy_version_and_unknown_fields_rejected():
    with pytest.raises(ValueError): BatchPolicy.from_dict({'version':'not-supported'})
    with pytest.raises(TypeError): BatchPolicy.from_dict({'surprise': 1})


@pytest.mark.parametrize('field,value,limit', [('max_padded_tokens',10,'padded_tokens'),
                                             ('max_cache_bytes',32,'cache_bytes')])
def test_impossible_budget_fails_before_any_backbone_work(field, value, limit):
    model = make_model(policy=BatchPolicy(**{field:value}))
    calls, hook = spy_backbone(model)
    with pytest.raises(BatchBudgetExceeded) as exc:
        model(encode(request()))
    hook.remove()
    assert exc.value.detail['limit'] == limit
    assert not calls


def test_padded_token_and_cache_budgets_include_parent_and_gqa():
    encoding = encode(request())
    # A larger next row increases padding for EVERY row, not just the next one.
    plan = plan_batches(encoding, BatchPolicy(max_padded_tokens=900, max_cache_bytes=25000),
                        kv_bytes_per_token=8, gqa_bytes_per_token=16)
    assert len(plan) > 1 and any(len(b.indices)>1 for b in plan)
    seen = []
    for b in plan:
        assert b.padded_tokens_with_prefix == len(b.indices)*(len(encoding.state)+b.padded_length)
        assert b.estimated_cache_bytes == len(encoding.state)*8 + b.padded_tokens_with_prefix*24
        assert b.padded_tokens_with_prefix <= 900 and b.estimated_cache_bytes <= 25000
        seen += list(b.indices)
    assert sorted(seen) == list(range(7))


def test_255_choice_and_10_level_actual_model_execution():
    data = request(1)
    data['questions'] = {
        'maximum choice': {'type':'choice','instructions':'select', 'criteria':{str(i):None for i in range(255)}},
        'maximum score': {'type':'score','instructions':'rate', 'criteria':[str(i) for i in range(10)]},
        'binary': {'type':'noul','instructions':'true?', 'criteria':{'false':'no', 'true':'yes'}}}
    encoding = encode(data, 16384)
    torch.manual_seed(19)
    model = DecisionEngine(BackboneConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, pointer_dim=4, max_position_embeddings=16384),
                         BatchPolicy(max_padded_tokens=11000))
    with torch.inference_mode():
        zs, stats = model(encoding)
        ref, _ = model(encoding, reference=True)
    assert sorted(stats['effective_batch_sizes']) == [1, 2]
    assert [z.numel() for z in zs] == [255, 10, 2]
    for a,b in zip(zs,ref):
        assert torch.isfinite(a).all()
        assert abs(float(a.softmax(-1).sum())-1)<1e-6
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize("hidden_dtype", [torch.bfloat16, torch.float32])
def test_bf16_pointer_scores_are_batch_shape_invariant_and_differentiable(hidden_dtype):
    torch.manual_seed(67)
    head = PointerHead(32, 16)
    decide = torch.randn(2, 32, dtype=hidden_dtype, requires_grad=True)
    options = torch.randn(6, 32, dtype=hidden_dtype, requires_grad=True)
    owner = torch.tensor([0, 0, 0, 1, 1, 1])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        batched = head.logits(decide, options, owner)
        separate = torch.cat((head(decide[0], options[:3]),
                              head(decide[1], options[3:])))
    with torch.autocast("cpu", enabled=False):
        query = torch.nn.functional.linear(decide.double(), head.q.weight.double(), head.q.bias.double())
        keys = torch.nn.functional.linear(options.double(), head.k.weight.double(), head.k.bias.double())
        expected = (keys * query[owner]).sum(-1).float() * head.scale
    assert batched.dtype == torch.float32
    torch.testing.assert_close(batched, separate, atol=0, rtol=0)
    torch.testing.assert_close(batched, expected, atol=0, rtol=0)
    batched.sum().backward()
    assert all(gradient is not None and torch.isfinite(gradient).all()
               for gradient in (decide.grad, options.grad, head.q.weight.grad, head.k.weight.grad))


@pytest.mark.parametrize('lora,checkpointing', [(0,False),(0,True),(2,False),(2,True)])
def test_parallel_resume_exact_across_accumulation_and_reward(tmp_path,suite,lora,checkpointing):
    data, manifest = suite
    model = make_model(lora, checkpointing, policy=BatchPolicy(max_branches=2))
    initial = copy.deepcopy(model.state_dict())
    settings = TrainSettings(steps=4, accumulation=3, save_every=2, choice_permutation=True, seed=52)
    cfg, limits = ObjectiveConfig(reinforce=.1), Limits(512,8192)
    full = train(model,ByteTokenizer(),data['train'],manifest,tmp_path/'full',settings,cfg,limits)
    other = make_model(lora,checkpointing,policy=model.batch_policy)
    other.load_state_dict(initial)
    part = train(other,ByteTokenizer(),data['train'],manifest,tmp_path/'part',settings,cfg,limits,stop_after=2)
    resumed, tok, point = load_checkpoint(Path(part['checkpoint']))
    assert point['execution'] == model.batch_policy.to_dict()
    torch.randn(50)
    end = train(resumed,tok,data['train'],manifest,tmp_path/'part',settings,cfg,limits,resume=Path(part['checkpoint']))
    for name, p in model.state_dict().items():
        torch.testing.assert_close(p,resumed.state_dict()[name],atol=0,rtol=0)
    assert full['state']['execution_counters'] == end['state']['execution_counters']
    assert end['state']['execution_counters']['max_batch_size'] == 2
    assert end['state']['execution_counters']['prefix_passes'] == 12
    validate_snapshot(json.loads(Path(end['snapshot']).read_text()))


@pytest.mark.parametrize('change', [{'max_branches':1},{'max_padded_tokens':30000},
                                    {'max_cache_bytes':300000000},{'sort_by_length':False}])
def test_resume_rejects_any_batch_policy_change(tmp_path,tiny,suite,change):
    data,manifest=suite
    settings=TrainSettings(steps=2)
    part=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',settings,ObjectiveConfig(),Limits(512,8192),stop_after=1)
    tiny.batch_policy=replace(tiny.batch_policy,**change)
    with pytest.raises(ValueError,match='configuration'):
        train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'run',settings,ObjectiveConfig(),Limits(512,8192),resume=Path(part['checkpoint']))


def test_legacy_checkpoint_weights_and_identity_preserved(tmp_path,tiny,request_data):
    path=tmp_path/'legacy.pt'
    tiny.training_steps=5
    save_checkpoint(path,tiny,ByteTokenizer())
    payload=torch.load(path,weights_only=True)
    payload.pop('execution')
    torch.save(payload,path)
    path.with_suffix('.manifest.json').unlink()
    model,tok,loaded=load_checkpoint(path)
    assert model.batch_policy == BatchPolicy()
    assert model.training_steps==5
    assert loaded['model_id'].startswith('kev-laya-0.1.0-')
    for n,p in tiny.state_dict().items(): torch.testing.assert_close(p,model.state_dict()[n],atol=0,rtol=0)
    zs,stats=model(encode(request_data))
    ref,_=model(encode(request_data),serial_reference=True)
    assert stats['effective_batch_sizes']==[3]
    for a,b in zip(zs,ref):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)


def test_service_diagnostics_preserve_body_and_v1_telemetry(tiny,request_data):
    tiny.training_steps=1
    runtime=InferenceRuntime(tiny,ByteTokenizer(),'kev-laya-tested',Limits(512,8192))
    app=create_app(runtime,ServeSettings(allow_unauthenticated=True))
    with TestClient(app) as client:
        result=client.post('/v1/systemone',json=request_data)
        assert result.status_code==200
        assert result.headers['x-kev-laya-batch-sizes']=='3'
        assert result.headers['x-kev-laya-prefix-passes']=='1'
        assert result.headers['x-kev-laya-branch-passes']=='1'
        body=result.json()
        assert list(body['answers'])==list(request_data['questions'])
        assert body['model']=='kev-laya-tested'
        assert body['usage']['input_tokens']==body['usage']['forward_tokens']
        assert set(app.state.hook.values)=={'requests','errors','input_tokens','forward_tokens','output_tokens','questions','prefix_reuses','latency_ms'}
        assert client.get('/v1/models').json()['question_batching']==BatchPolicy().to_dict()


def test_service_budget_overflow_is_explicit_422(tiny,request_data):
    tiny.training_steps=1
    tiny.batch_policy=BatchPolicy(max_padded_tokens=20)
    app=create_app(InferenceRuntime(tiny,ByteTokenizer(),'kev-laya-tested',Limits(512,8192)),ServeSettings(allow_unauthenticated=True))
    with TestClient(app) as client:
        result=client.post('/v1/systemone',json=request_data)
        assert result.status_code==422
        assert result.json()['detail']['code']=='question_batch_budget_exceeded'
        assert app.state.hook.values['errors']==1


def test_diagnostic_export_failure_does_not_corrupt_checkpoint(tmp_path,tiny,suite,monkeypatch):
    import kev_laya.training as module
    original=module.atomic_json
    def write(path,value):
        if path.name.startswith('execution-'): raise OSError('fixture disk failure')
        return original(path,value)
    monkeypatch.setattr(module,'atomic_json',write)
    data,manifest=suite
    result=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'train',TrainSettings(steps=1),ObjectiveConfig(),Limits(512,8192))
    assert result['monitoring_export_failures']==1
    _,_,payload=load_checkpoint(Path(result['checkpoint']))
    assert payload['training_state']['step']==1


def test_legacy_serial_resume_is_explicitly_rejected(tmp_path,tiny,suite):
    data,manifest=suite
    settings=TrainSettings(steps=2)
    part=train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'old',settings,ObjectiveConfig(),Limits(512,8192),stop_after=1)
    path=Path(part['checkpoint']);payload=torch.load(path,weights_only=True);payload.pop('execution')
    torch.save(payload,path);path.with_suffix('.manifest.json').unlink()
    with pytest.raises(ValueError,match='legacy serial checkpoint'):
        train(tiny,ByteTokenizer(),data['train'],manifest,tmp_path/'old',settings,ObjectiveConfig(),Limits(512,8192),resume=path)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; no native GPU evidence')
def test_cuda_fixture_fp32_parallel_and_gradients():
    old=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        model=make_model(checkpointing=True,policy=BatchPolicy(max_branches=3)).cuda()
        encoding=encode(request(7))
        a,_=model(encoding);sum(z.square().mean() for z in a).backward()
        gradients={n:p.grad.clone() for n,p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        b,_=model(encoding,reference=True);sum(z.square().mean() for z in b).backward()
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
        for n,p in model.named_parameters():torch.testing.assert_close(gradients[n],p.grad,atol=2e-5,rtol=2e-5)
        torch.cuda.synchronize()
    finally:
        torch.backends.cuda.matmul.allow_tf32=old



@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable; native cache unverified')
@pytest.mark.parametrize('precision', ['fp32', 'bf16', 'native_bf16', 'native_fp16'])
def test_cuda_training_cache_plan_matches_double_prefix_and_rejects_before_forward(precision):
    if precision in ('bf16', 'native_bf16') and not torch.cuda.is_bf16_supported():
        pytest.skip('GPU does not support BF16')
    encoded = encode(request(2))
    model = make_model(lora=2, checkpointing=True,
                       policy=BatchPolicy(max_branches=2)).cuda().train()
    if precision == 'native_bf16':
        model = model.to(torch.bfloat16)
    elif precision == 'native_fp16':
        model = model.to(torch.float16)
    enabled = precision == 'bf16'
    cache_element_size = 8 if precision != 'native_fp16' else 2
    cfg = model.cfg
    kv_rate = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (
        cfg.hidden_size // cfg.num_attention_heads) * cache_element_size
    gqa_rate = 2 * cfg.num_hidden_layers * cfg.hidden_size * cache_element_size
    expected = plan_batches(encoded, model.batch_policy,
                            kv_bytes_per_token=kv_rate, gqa_bytes_per_token=gqa_rate)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=enabled):
        logits, usage = model(encoded)
    assert len(logits) == len(encoded.branches)
    assert usage['prefix_cache_bytes'] == len(encoded.state) * kv_rate
    assert usage['estimated_peak_cache_bytes'] == max(batch.estimated_cache_bytes for batch in expected)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=enabled):
        _, prepass_usage = model(encoded, _training_cache_plan=True)
    assert prepass_usage['estimated_peak_cache_bytes'] == usage['estimated_peak_cache_bytes']
    singletons = plan_batches(encoded, BatchPolicy(max_branches=1),
                              kv_bytes_per_token=kv_rate, gqa_bytes_per_token=gqa_rate)
    tight_limit = min(batch.estimated_cache_bytes for batch in singletons) - 1
    restricted = BatchPolicy(max_branches=2, max_cache_bytes=tight_limit)
    calls, handle = spy_backbone(model)
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=enabled):
            with pytest.raises(BatchBudgetExceeded) as failure:
                model(encoded, policy=restricted)
        assert failure.value.detail['limit'] == 'cache_bytes'
        assert calls == []
    finally:
        handle.remove()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_fp32_medium_cache_budget_accounts_observed_double_gqa(monkeypatch):
    from kev_laya.native_gradients import CanonicalLongSDPA
    payload = request(2)
    payload['state'] = {'text': 'x' * 300}
    encoded = encode(payload)
    assert 256 < len(encoded.state) < 8192
    model = make_model(lora=2, checkpointing=False,
                       policy=BatchPolicy(max_branches=2)).cuda().train()
    original = CanonicalLongSDPA.forward
    observations = []

    def observe(ctx, query, key, value, *args):
        observations.append((key.dtype, value.dtype, key.shape[1],
                             key.element_size(), value.element_size()))
        return original(ctx, query, key, value, *args)

    monkeypatch.setattr(CanonicalLongSDPA, 'forward', staticmethod(observe))
    _, usage = model(encoded)
    assert observations
    assert all(kdtype == vdtype == torch.float64 and heads == model.cfg.num_attention_heads
               and kbytes == vbytes == 8
               for kdtype, vdtype, heads, kbytes, vbytes in observations)
    cfg = model.cfg
    kv_rate = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (
        cfg.hidden_size // cfg.num_attention_heads) * observations[0][3]
    gqa_rate = 2 * cfg.num_hidden_layers * cfg.hidden_size * observations[0][3]
    expected = plan_batches(encoded, model.batch_policy,
                            kv_bytes_per_token=kv_rate, gqa_bytes_per_token=gqa_rate)
    assert usage['prefix_cache_bytes'] == len(encoded.state) * kv_rate
    assert usage['estimated_peak_cache_bytes'] == max(batch.estimated_cache_bytes for batch in expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; BF16 training unverified')
def test_cuda_fixture_bf16_training(tmp_path,suite):
    if not torch.cuda.is_bf16_supported():pytest.skip('GPU does not support BF16')
    model=make_model(lora=2,checkpointing=True,policy=BatchPolicy(max_branches=2)).cuda()
    data,manifest=suite
    result=train(model,ByteTokenizer(),data['train'],manifest,tmp_path/'bf16',TrainSettings(steps=2,precision='bf16'),
                 ObjectiveConfig(reinforce=.1),Limits(512,8192))
    assert result['state']['execution_counters']['max_batch_size']==2
    from kev_laya.native_gradients import DERIVATIVE_VERSION
    assert result['state']['derivative_version']==DERIVATIVE_VERSION


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; BF16 parity unverified')
def test_cuda_bf16_short_branch_vjp_parity():
    if not torch.cuda.is_bf16_supported():pytest.skip('GPU does not support BF16')
    model=make_model(lora=2,checkpointing=True,policy=BatchPolicy(max_branches=2)).cuda().train()
    encoded=encode(request(3))
    assert max(len(encoded.state)+len(branch.ids) for branch in encoded.branches)<=256
    targets=[[1/len(branch.option_ends)]*len(branch.option_ends) for branch in encoded.branches]
    kinds=[branch.question.type for branch in encoded.branches]
    outputs={}
    for mode,kwargs in (('batch',{}),('serial',{'serial_reference':True}),
                        ('full',{'reference':True})):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,_=model(encoded,**kwargs)
            loss,_=objective(logits,targets,kinds,ObjectiveConfig(ordinal=.1))
        logit_gradients=torch.autograd.grad(loss,logits)
        for i,(logit,gradient) in enumerate(zip(logits,logit_gradients,strict=True)):
            torch.autograd.backward(logit,gradient,retain_graph=i+1<len(logits))
        outputs[mode]=([value.detach().float().cpu() for value in logits],
                       {name:parameter.grad.detach().float().cpu()
                        for name,parameter in model.named_parameters() if parameter.requires_grad})
    for left,right in (('batch','serial'),('serial','full'),('batch','full')):
        for actual,expected in zip(outputs[left][0],outputs[right][0],strict=True):
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        assert outputs[left][1].keys()==outputs[right][1].keys()
        for name in outputs[left][1]:
            torch.testing.assert_close(outputs[left][1][name],outputs[right][1][name],
                                       atol=2e-5,rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; medium BF16 parity unverified')
def test_cuda_bf16_mixed_short_medium_branch_vjp_parity():
    if not torch.cuda.is_bf16_supported():pytest.skip('GPU does not support BF16')
    model=make_model(lora=2,checkpointing=True,policy=BatchPolicy(max_branches=2)).cuda().train()
    encoded=encode(request(4))
    totals=[len(encoded.state)+len(branch.ids) for branch in encoded.branches]
    assert max(totals)>256 and min(totals)<=256
    targets=[[1/len(branch.option_ends)]*len(branch.option_ends) for branch in encoded.branches]
    kinds=[branch.question.type for branch in encoded.branches]
    outputs={}
    for mode,kwargs in (('batch',{}),('serial',{'serial_reference':True}),
                        ('full',{'reference':True})):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,_=model(encoded,**kwargs)
            loss,_=objective(logits,targets,kinds,ObjectiveConfig(ordinal=.1))
        gradients=torch.autograd.grad(loss,logits)
        for index,(logit,gradient) in enumerate(zip(logits,gradients,strict=True)):
            torch.autograd.backward(logit,gradient,retain_graph=index+1<len(logits))
        outputs[mode]=([value.detach().float().cpu() for value in logits],
                       {name:parameter.grad.detach().float().cpu()
                        for name,parameter in model.named_parameters() if parameter.requires_grad})
    for left,right in (('batch','serial'),('serial','full'),('batch','full')):
        for actual,expected in zip(outputs[left][0],outputs[right][0],strict=True):
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        for name in outputs[left][1]:
            torch.testing.assert_close(outputs[left][1][name],outputs[right][1][name],
                                       atol=2e-5,rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; BF16 RMSNorm unverified')
def test_cuda_bf16_prefix_norm_independent_of_longer_row():
    if not torch.cuda.is_bf16_supported():pytest.skip('GPU does not support BF16')
    torch.manual_seed(20260929)
    full=torch.randn(1,583,896,device='cuda')*8
    prefix=full[:,:13].clone()
    norm=RMSNorm(896,1e-6).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        whole=norm(full)
        separate=norm(prefix)
    torch.testing.assert_close(whole[:,:13],separate,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; medium BF16 VJP unverified')
def test_cuda_bf16_medium_vjp_preserves_double_cache_contributions():
    if not torch.cuda.is_bf16_supported():pytest.skip('GPU does not support BF16')
    torch.manual_seed(91)
    query=torch.randn(1,1,257,8,device='cuda',dtype=torch.bfloat16)
    key=torch.randn(1,1,257,8,device='cuda',dtype=torch.float64)
    value=torch.randn(1,1,257,8,device='cuda',dtype=torch.float64,requires_grad=True)
    output=CanonicalMediumAttention.apply(query,key,value,0)
    upstream=torch.zeros_like(output)
    upstream[...,-1,0]=0.1171875
    value_gradient=torch.autograd.grad(output,value,upstream)[0]
    scores=(query[...,-1:,:].double() @ key.to(torch.bfloat16).double().transpose(-2,-1))
    probabilities=torch.softmax((scores.to(torch.bfloat16)*(8**-0.5)),
                                dim=-1,dtype=torch.float32).to(torch.bfloat16)
    expected=probabilities.double()*0.1171875
    assert value_gradient.dtype==torch.float64
    assert float((expected-expected.to(torch.bfloat16)).abs().max())>1e-6
    torch.testing.assert_close(value_gradient[...,0],expected.squeeze(-2),atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; streamed BF16 unverified')
def test_cuda_bf16_streamed_branch_gradients_and_resume_identity(tmp_path,suite):
    if not torch.cuda.is_bf16_supported():pytest.skip('GPU does not support BF16')
    from kev_laya.training import _streamed_bf16_backward
    ordinary=make_model(lora=2,checkpointing=True,policy=BatchPolicy(max_branches=2)).cuda().train()
    streamed=copy.deepcopy(ordinary)
    encoded=encode(request(3))
    targets=[[1/len(branch.option_ends)]*len(branch.option_ends) for branch in encoded.branches]
    kinds=[branch.question.type for branch in encoded.branches]
    config=ObjectiveConfig(ordinal=.1,reinforce=.1)
    torch.manual_seed(551)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits,_=ordinary(encoded)
        loss,_=objective(logits,targets,kinds,config)
    gradients=torch.autograd.grad(loss,logits)
    for index,(logit,gradient) in enumerate(zip(logits,gradients,strict=True)):
        torch.autograd.backward(logit,gradient,retain_graph=index+1<len(logits))
    torch.manual_seed(551)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        parts,usage=_streamed_bf16_backward(streamed,encoded,targets,kinds,config,1)
    assert parts['numerics/recomputed_logits_max_abs']<=1e-5
    assert usage['prefix_passes']==1
    assert max(usage['effective_batch_sizes'])==2
    for (name,left),(other,right) in zip(ordinary.named_parameters(),streamed.named_parameters(),strict=True):
        assert name==other
        if left.requires_grad:
            torch.testing.assert_close(left.grad,right.grad,atol=2e-5,rtol=2e-5)

    data,manifest=suite
    settings=TrainSettings(steps=1,precision='bf16',streamed_branch_vjp=True)
    result=train(make_model(lora=2,checkpointing=True,policy=BatchPolicy(max_branches=2)).cuda(),
                 ByteTokenizer(),data['train'],manifest,tmp_path/'streamed',settings,
                 ObjectiveConfig(reinforce=.1),Limits(512,8192))
    assert result['state']['step']==1
    assert result['metrics'][0]['numerics/recomputed_logits_max_abs']<=1e-5
    restored,tokenizer,_=load_checkpoint(Path(result['checkpoint']),'cuda:0')
    with pytest.raises(ValueError,match='resume configuration or frozen data differs'):
        train(restored,tokenizer,data['train'],manifest,tmp_path/'streamed',
              TrainSettings(steps=1,precision='bf16'),ObjectiveConfig(reinforce=.1),
              Limits(512,8192),resume=Path(result['checkpoint']))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; streamed resume unverified')
def test_cuda_bf16_streamed_resume_matches_uninterrupted_with_accumulation(tmp_path,suite):
    if not torch.cuda.is_bf16_supported():pytest.skip('GPU does not support BF16')
    data,manifest=suite
    initial=make_model(lora=2,checkpointing=True,policy=BatchPolicy(max_branches=2)).cuda()
    settings=TrainSettings(steps=2,accumulation=2,save_every=1,precision='bf16',
                           streamed_branch_vjp=True)
    config=ObjectiveConfig(reinforce=.1)
    limits=Limits(512,8192)
    full_model=copy.deepcopy(initial)
    full=train(full_model,ByteTokenizer(),data['train'],manifest,tmp_path/'full-streamed',
               settings,config,limits)
    part=train(copy.deepcopy(initial),ByteTokenizer(),data['train'],manifest,
               tmp_path/'split-streamed',settings,config,limits,stop_after=1)
    resumed,tokenizer,_=load_checkpoint(Path(part['checkpoint']),'cuda:0')
    end=train(resumed,tokenizer,data['train'],manifest,tmp_path/'split-streamed',
              settings,config,limits,resume=Path(part['checkpoint']))
    assert full['state']['step']==end['state']['step']==2
    assert full['state']['forward_tokens']==end['state']['forward_tokens']
    for name,value in full_model.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; precision resume unverified')
@pytest.mark.parametrize('precision,version_field',[
    ('fp32','FP32_DERIVATIVE_VERSION'),
    ('bf16','DERIVATIVE_VERSION'),
    ('fp32','CUDA_FP32_ATTENTION_OPERATOR_VERSION'),
    ('fp32','CUDA_TRAINING_CACHE_PLAN_VERSION'),
    ('bf16','CUDA_TRAINING_CACHE_PLAN_VERSION'),
])
def test_cuda_resume_rejects_derivative_version_change(tmp_path,suite,monkeypatch,precision,version_field):
    if precision=='bf16' and not torch.cuda.is_bf16_supported():pytest.skip('GPU does not support BF16')
    model=make_model(lora=2,checkpointing=True,policy=BatchPolicy(max_branches=2)).cuda()
    data,manifest=suite
    settings=TrainSettings(steps=2,precision=precision)
    root=tmp_path/'bf16-version'
    first=train(model,ByteTokenizer(),data['train'],manifest,root,settings,
                ObjectiveConfig(reinforce=.1),Limits(512,8192),stop_after=1)
    restored,tokenizer,_=load_checkpoint(Path(first['checkpoint']),'cuda:0')
    import kev_laya.training as training_module
    if version_field in ('DERIVATIVE_VERSION','FP32_DERIVATIVE_VERSION'):
        assert first['state']['derivative_version']==getattr(training_module,version_field)
    if version_field=='CUDA_TRAINING_CACHE_PLAN_VERSION':
        config=json.loads((root/'config.json').read_text(encoding='utf-8'))
        assert config['cache_plan_version']==getattr(training_module,version_field)
    monkeypatch.setattr(training_module,version_field,'unreviewed-derivative')
    with pytest.raises(ValueError,match='resume configuration or frozen data differs'):
        train(restored,tokenizer,data['train'],manifest,root,settings,
              ObjectiveConfig(reinforce=.1),Limits(512,8192),
              resume=Path(first['checkpoint']))
