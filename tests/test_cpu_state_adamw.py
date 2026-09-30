"""Candidate qualification tests; copy under tests/ only after frozen run ends."""
import copy
import pytest
import torch

from kev_laya.cpu_state_adamw import CPUStateStreamingAdamW, TORCH_VERSION


def _optimizer(p, budget=1_000_000):
    return CPUStateStreamingAdamW([("weight", p)], lr=3e-4, weight_decay=0.01,
                                  max_state_bytes=budget)


def test_cpu_or_insufficient_budget_fails_closed():
    if torch.__version__ != TORCH_VERSION:
        with pytest.raises(RuntimeError, match="requires torch"):
            _optimizer(torch.nn.Parameter(torch.ones(4)))
        return
    with pytest.raises(ValueError, match="CUDA FP32"):
        _optimizer(torch.nn.Parameter(torch.ones(4)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streaming_matches_single_tensor_cuda_adamw_and_resumes():
    a = torch.nn.Parameter(torch.linspace(-1, 1, 513, device="cuda"))
    b = torch.nn.Parameter(a.detach().clone())
    streamed = _optimizer(a)
    native = torch.optim.AdamW([b], lr=3e-4, weight_decay=0.01,
                               foreach=False, fused=False, capturable=False)
    saved = None
    for index in range(3):
        grad = torch.cos(torch.arange(a.numel(), device="cuda", dtype=torch.float32) + index)
        a.grad = grad.clone(); b.grad = grad.clone()
        streamed.step(); native.step()
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert streamed.state[a][key].device.type == "cpu"
            torch.testing.assert_close(streamed.state[a][key], native.state[b][key].cpu(), atol=0, rtol=0)
        if index == 0:
            saved = copy.deepcopy(streamed.state_dict())
            c = torch.nn.Parameter(a.detach().clone())
            resumed = _optimizer(c)
            resumed.load_state_dict(saved)
            assert all(v.device.type == "cpu" for v in resumed.state[c].values())
        elif index > 0:
            c.grad = grad.clone(); resumed.step()
            torch.testing.assert_close(c, a, atol=0, rtol=0)
    assert saved is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_corruption_and_identity_rejected_without_cuda_migration():
    p = torch.nn.Parameter(torch.ones(13, device="cuda"))
    opt = _optimizer(p)
    with pytest.raises(MemoryError):
        _optimizer(p, budget=1)
    p.grad = torch.ones_like(p); opt.step()
    clean = opt.state_dict()
    for mutate in (
        lambda x: x["param_groups"][0].update(algorithm_version="wrong"),
        lambda x: x["param_groups"][0].update(param_specs=[("other", (13,), "torch.float32")]),
        lambda x: x["state"][0].update(exp_avg=torch.ones(13, device="cuda")),
        lambda x: x["state"][0].update(exp_avg=torch.ones(12)),
        lambda x: x["state"][0]["exp_avg"].fill_(float("nan")),
        lambda x: x["state"][0]["exp_avg_sq"].fill_(float("inf")),
        lambda x: x["state"][0]["exp_avg_sq"].fill_(-1),
        lambda x: x["state"][0].update(step=torch.tensor(-1.0)),
        lambda x: x["state"].update({0: None}),
        lambda x: x["state"].update({0: []}),
        lambda x: x["state"].update({0: 0}),
    ):
        broken = copy.deepcopy(clean); mutate(broken)
        new = _optimizer(torch.nn.Parameter(torch.ones(13, device="cuda")))
        with pytest.raises(ValueError):
            new.load_state_dict(broken)
        assert not new.state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rejected_load_preserves_populated_state_and_learning_rate():
    p = torch.nn.Parameter(torch.ones(13, device="cuda"))
    opt = _optimizer(p)
    p.grad = torch.full_like(p, 0.25); opt.step()
    baseline = copy.deepcopy(opt.state_dict())
    rejected = copy.deepcopy(baseline)
    rejected["param_groups"][0]["lr"] = 0.123
    rejected["state"][0]["exp_avg_sq"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        opt.load_state_dict(rejected)
    after = opt.state_dict()
    assert after["param_groups"][0]["lr"] == baseline["param_groups"][0]["lr"]
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(after["state"][0][key], baseline["state"][0][key], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_alternating_none_gradients_match_stock_and_resume():
    a = torch.nn.Parameter(torch.linspace(-1, 1, 17, device="cuda"))
    b = torch.nn.Parameter(torch.linspace(1, -1, 17, device="cuda"))
    x = torch.nn.Parameter(a.detach().clone())
    y = torch.nn.Parameter(b.detach().clone())
    streamed = CPUStateStreamingAdamW([("a", a), ("b", b)], lr=3e-4,
                                       weight_decay=0.01, max_state_bytes=1_000_000)
    stock = torch.optim.AdamW([x, y], lr=3e-4, weight_decay=0.01,
                              foreach=False, fused=False, capturable=False)
    resumed = None
    for index, active in enumerate(((True, True), (True, False), (False, True), (True, True))):
        for parameter, reference, enabled, shift in ((a, x, active[0], 0), (b, y, active[1], 1)):
            gradient = torch.cos(torch.arange(17, device="cuda", dtype=torch.float32) + index + shift)
            parameter.grad = gradient.clone() if enabled else None
            reference.grad = gradient.clone() if enabled else None
        streamed.step(); stock.step()
        for parameter, reference in ((a, x), (b, y)):
            torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
            for field in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(streamed.state[parameter][field],
                                           stock.state[reference][field].cpu(), atol=0, rtol=0)
        if index == 0:
            c = torch.nn.Parameter(a.detach().clone())
            d = torch.nn.Parameter(b.detach().clone())
            resumed = CPUStateStreamingAdamW([("a", c), ("b", d)], lr=3e-4,
                                              weight_decay=0.01, max_state_bytes=1_000_000)
            resumed.load_state_dict(copy.deepcopy(streamed.state_dict()))
        else:
            c.grad = a.grad.clone() if a.grad is not None else None
            d.grad = b.grad.clone() if b.grad is not None else None
            resumed.step()
            torch.testing.assert_close(c, a, atol=0, rtol=0)
            torch.testing.assert_close(d, b, atol=0, rtol=0)
    assert resumed is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_equal_shape_parameter_id_swap_and_mutable_options_fail_closed():
    a = torch.nn.Parameter(torch.ones(8, device="cuda"))
    b = torch.nn.Parameter(torch.zeros(8, device="cuda"))
    opt = CPUStateStreamingAdamW([("a", a), ("b", b)], lr=3e-4,
                                 weight_decay=0.01, max_state_bytes=1_000_000)
    a.grad = torch.ones_like(a); b.grad = torch.full_like(b, 2)
    opt.step()
    clean = opt.state_dict()
    swapped = copy.deepcopy(clean)
    swapped["param_groups"][0]["params"] = [1, 0]
    target = CPUStateStreamingAdamW(
        [("a", torch.nn.Parameter(torch.ones(8, device="cuda"))),
         ("b", torch.nn.Parameter(torch.zeros(8, device="cuda")))],
        lr=3e-4, weight_decay=0.01, max_state_bytes=1_000_000)
    with pytest.raises(ValueError, match="count/order"):
        target.load_state_dict(swapped)
    assert not target.state
    for key, changed in (("foreach", True), ("fused", True), ("amsgrad", True),
                         ("capturable", True), ("weight_decay", 0.5),
                         ("algorithm_version", "wrong")):
        mutation = copy.deepcopy(clean)
        mutation["param_groups"][0][key] = changed
        with pytest.raises(ValueError, match="options"):
            target.load_state_dict(mutation)
        assert not target.state
        opt.param_groups[0][key] = changed
        with pytest.raises(ValueError, match="options"):
            opt.step()
        opt.param_groups[0][key] = clean["param_groups"][0][key]
    alias = copy.deepcopy(clean)
    alias["state"][1]["exp_avg"] = alias["state"][0]["exp_avg"]
    with pytest.raises(ValueError, match="aliased"):
        target.load_state_dict(alias)
    assert not target.state
    step_alias = copy.deepcopy(clean)
    step_alias["state"][0]["exp_avg"][0] = 1.0
    step_alias["state"][0]["step"] = step_alias["state"][0]["exp_avg"][0]
    with pytest.raises(ValueError, match="aliased"):
        target.load_state_dict(step_alias)
    assert not target.state
    cross_step_alias = copy.deepcopy(clean)
    cross_step_alias["state"][1]["step"] = cross_step_alias["state"][0]["step"]
    with pytest.raises(ValueError, match="aliased"):
        target.load_state_dict(cross_step_alias)
    assert not target.state
    with pytest.raises(ValueError, match="duplicate"):
        CPUStateStreamingAdamW([("same", a), ("same", b)], lr=3e-4,
                               weight_decay=0.01, max_state_bytes=1_000_000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_live_equal_shape_parameter_swap_rejected_before_step_or_serialization():
    a = torch.nn.Parameter(torch.ones(8, device="cuda"))
    b = torch.nn.Parameter(torch.zeros(8, device="cuda"))
    opt = CPUStateStreamingAdamW([("a", a), ("b", b)], lr=3e-4,
                                 weight_decay=0.01, max_state_bytes=1_000_000)
    a.grad = torch.ones_like(a); b.grad = torch.ones_like(b)
    initial_a, initial_b = a.detach().clone(), b.detach().clone()
    opt.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError, match="identity/order/layout"):
        opt.step()
    with pytest.raises(ValueError, match="identity/order/layout"):
        opt.state_dict()
    with pytest.raises(ValueError, match="identity/order/layout"):
        opt.load_state_dict({"state": {}, "param_groups": []})
    torch.testing.assert_close(a, initial_a, atol=0, rtol=0)
    torch.testing.assert_close(b, initial_b, atol=0, rtol=0)
    opt.param_groups[0]["params"].reverse()
    opt.param_groups[0]["params"][0] = torch.nn.Parameter(a.detach().clone())
    with pytest.raises(ValueError, match="identity/order/layout"):
        opt.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_none_gradient_does_not_advance_state_or_decay_parameter():
    p = torch.nn.Parameter(torch.ones(8, device="cuda"))
    opt = _optimizer(p)
    before = p.detach().clone()
    opt.step()
    assert not opt.state[p]
    torch.testing.assert_close(p, before, atol=0, rtol=0)
