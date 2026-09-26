import copy
import pytest
import torch
from kev_laya.model import DecisionEngine, BackboneConfig
from kev_laya.schema import SystemOneRequest
from kev_laya.encoding import encode_request, ByteTokenizer, Limits
from kev_laya.objectives import objective, ObjectiveConfig, padded

def encoded(obj): return encode_request(SystemOneRequest(**obj), ByteTokenizer(), Limits(512, 8192))

@pytest.mark.parametrize('lora', [0, 2])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_reference_shared_logits_and_all_gradients(request_data, lora, checkpointing):
    torch.manual_seed(73)
    m = DecisionEngine(BackboneConfig(hidden_size=32, intermediate_size=64, pointer_dim=16,
                                      lora_rank=lora, activation_checkpointing=checkpointing))
    e = encoded(request_data)
    zs, usage = m(e)
    loss = sum(z.square().mean() for z in zs)
    loss.backward()
    gradients = {n:p.grad.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
    m.zero_grad(set_to_none=True)
    ref, compute = m(e, reference=True)
    sum(z.square().mean() for z in ref).backward()
    for a,b in zip(zs,ref): torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    for n,p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            torch.testing.assert_close(gradients[n],p.grad,atol=2e-5,rtol=2e-5)
        else: assert p.grad is None
    assert usage['forward_tokens'] == e.logical_tokens
    assert compute['forward_tokens'] == e.logical_tokens + 2 * len(e.state)
    assert usage['prefix_reuses'] == 2 and usage['prefix_cache_bytes'] > 0

def test_question_order_addition_and_ids_isolation(request_data, tiny):
    tiny.eval()
    # Batch shape may change GEMM reduction rounding, within the frozen 1e-5 contract.
    initial, _ = tiny(encoded(request_data))
    changed = copy.deepcopy(request_data)
    changed['questions'] = dict(reversed(list(changed['questions'].items())))
    flipped, _ = tiny(encoded(changed))
    for x,y in zip(initial,reversed(flipped)): torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    changed['questions']['additional'] = {'type':'choice','instructions':'unrelated instructions', 'criteria':{'u':None,'v':None}}
    added, _ = tiny(encoded(changed))
    for x,y in zip(flipped,added): torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    changed['questions'] = {'x'+str(i):v for i,v in enumerate(request_data['questions'].values())}
    renamed, _ = tiny(encoded(changed))
    for x,y in zip(initial,renamed): torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)

def test_prefix_not_mutated_and_cross_request_isolated(request_data,tiny):
    e = encoded(request_data)
    _, prefix = tiny.backbone(e.state)
    copied = [[x.clone() for x in kv] for kv in prefix]
    tiny.backbone(e.branches[0].ids, prefix)
    for kv,old in zip(prefix,copied):
        for x,y in zip(kv,old): torch.testing.assert_close(x,y,atol=0,rtol=0)
    initial,_ = tiny(e)
    other = copy.deepcopy(request_data); other['state'] = 'a completely different state'
    tiny(encoded(other))
    repeated,_ = tiny(e)
    for x,y in zip(initial,repeated): torch.testing.assert_close(x,y,atol=0,rtol=0)

def test_option_order_is_measured_not_question_isolation(request_data,tiny):
    a,_=tiny(encoded(request_data))
    request_data['questions']['color']['criteria']={'blue':None,'red':None}
    b,_=tiny(encoded(request_data))
    # Do not assert permutation invariance: the architecture intentionally does not enforce it.
    sensitivity=(a[0].softmax(-1)-b[0].flip(0).softmax(-1)).abs().max().item()
    assert 0 <= sensitivity <= 1
    for x,y in zip(a[1:],b[1:]): torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)

@pytest.mark.parametrize('reinforce', [0, .1, 1.0])
def test_objective_finite_backbone_head_gradients(request_data,tiny,reinforce):
    z,_=tiny(encoded(request_data))
    loss,metrics=objective(z, [[1,0],[0,1,0],[0,1]], ['choice','score','noul'],
                           ObjectiveConfig(ce=0 if reinforce==1 else 1,reinforce=reinforce,ordinal=.1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in tiny.parameters())
    assert any(p.grad.abs().max()>0 for p in tiny.head.parameters())
    assert any(p.grad.abs().max()>0 for p in tiny.backbone.parameters())
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())

def test_padded_options_have_zero_gradient():
    z1=torch.tensor([.2,.7],requires_grad=True); z2=torch.tensor([.3],requires_grad=True)
    loss,_=objective([z1,z2],[[.3,.7],[1]],['choice','choice'],ObjectiveConfig(reinforce=.1))
    loss.backward()
    assert z2.grad.item() == 0.0
    assert torch.isfinite(z1.grad).all()

@pytest.mark.parametrize('target', [[.1], [.2,.2], [float('nan'),0], [-1,2]])
def test_invalid_targets(target):
    with pytest.raises(ValueError): padded([torch.tensor([0.,1.])],[target])

@pytest.mark.parametrize('cfg', [{'ce':0,'ordinal':0,'reinforce':0}, {'sigma':0}, {'samples':1}, {'reinforce':-1}])
def test_bad_objective_config(cfg):
    with pytest.raises(ValueError): ObjectiveConfig(**cfg)

def test_refuses_hybrid_and_sliding():
    with pytest.raises(ValueError): BackboneConfig.from_qwen({'model_type':'qwen3_5'})
    with pytest.raises(ValueError): BackboneConfig.from_qwen({'model_type':'qwen2','use_sliding_window':True})

def test_merged_lora_weights_equal_native_projection():
    from kev_laya.native_validation import merged_backbone
    m=DecisionEngine(BackboneConfig(hidden_size=32,intermediate_size=64,pointer_dim=16,lora_rank=2))
    layer=m.backbone.layers[0].self_attn.q_proj
    with torch.no_grad():layer.b.normal_()
    weights=merged_backbone(m); x=torch.randn(1,3,32)
    actual=layer(x)
    expected=torch.nn.functional.linear(x,weights['layers.0.self_attn.q_proj.weight'],weights['layers.0.self_attn.q_proj.bias'])
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)

def test_native_harness_refuses_fixture(tmp_path,tiny):
    from kev_laya.checkpoint import save_checkpoint
    from kev_laya.native_validation import run
    p=tmp_path/'tiny.pt';save_checkpoint(p,tiny,ByteTokenizer())
    with pytest.raises(ValueError,match='native validation requires'):
        run(p,tmp_path/'native-report','cpu')

def test_pgps_monte_carlo_gradient_matches_pathwise_reference():
    from kev_laya.objectives import proper_reward
    mu=torch.tensor([.3,-.2,.1],requires_grad=True)
    cfg=ObjectiveConfig(ce=0,reinforce=1,sigma=.35,samples=20000)
    torch.manual_seed(103)
    loss,_=objective([mu],[[.2,.7,.1]],['score'],cfg);loss.backward()
    score_gradient=mu.grad.clone()
    z=mu.detach().clone().requires_grad_()
    torch.manual_seed(103)
    eps=torch.randn((20000,1,3))*.35;eps=eps-eps.mean(-1,keepdim=True)
    samples=z[None,None]-z.mean()+eps
    mask=torch.ones(1,3,dtype=torch.bool)
    reward=proper_reward(samples.log_softmax(-1),torch.tensor([[.2,.7,.1]]),mask,torch.ones(1),.75,1.)
    (-reward.mean()).backward()
    torch.testing.assert_close(score_gradient,z.grad,atol=.02,rtol=.02)
