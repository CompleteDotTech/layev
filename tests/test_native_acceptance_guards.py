"""Acceptance-harness software tests, NOT pretrained/CUDA/context evidence.

Byte tokenization and CPU tensors exercise contracts. Device, oracle, checkpoint
and exposure doubles below test orchestration only; they cannot supply a native
acceptance receipt. No remote calls, weights or actual CUDA work are used.
"""
from contextlib import contextmanager
from dataclasses import replace
from importlib import metadata
from types import SimpleNamespace
import copy
import json
import sys

import pytest
import torch

from kev_laya import native_validation as native
from kev_laya.encoding import ByteTokenizer, ContextOverflow, Limits, encode_request
from kev_laya.schema import SystemOneRequest


@pytest.fixture
def small_encoding():
    request = SystemOneRequest(state="contract fixture", model="kev-laya-preview", questions={
        "route": {"type": "choice", "instructions": "route", "criteria": {"billing": None, "other": None}},
        "urgency": {"type": "score", "instructions": "level", "criteria": ["low", "medium", "high"]},
        "escalate": {"type": "noul", "instructions": "escalate?"},
    })
    return encode_request(request, ByteTokenizer(), Limits())


def vectors(encoded):
    return [torch.arange(len(b.question.options()), dtype=torch.float32) for b in encoded.branches]


@pytest.fixture
def exact_request():
    # Exact SERIALIZED BYTE lengths only. These are not native-tokenizer lengths.
    request = SystemOneRequest(state="boundary software fixture", model="kev-laya-preview", questions={
        "route": {"type": "choice", "instructions": "route", "criteria": {"billing": "billing", **{
            f"option-{i}": str(i) for i in range(254)}}},
        "urgency": {"type": "score", "instructions": "level", "criteria": [str(i) for i in range(10)]},
        "escalate": {"type": "noul", "instructions": "escalate?"},
    })
    tokenizer = ByteTokenizer()
    obj = request.model_dump()
    encoded = encode_request(request, tokenizer, Limits())
    obj["questions"]["route"]["instructions"] += "x" * (32768 - len(encoded.state) - len(encoded.branches[0].ids))
    for index, name in ((1, "urgency"), (2, "escalate")):
        request = SystemOneRequest(**obj)
        encoded = encode_request(request, tokenizer, Limits())
        extra = min(65536 - encoded.logical_tokens, 32768 - len(encoded.state) - len(encoded.branches[index].ids))
        obj["questions"][name]["instructions"] += "x" * extra
    request = SystemOneRequest(**obj)
    encoded = encode_request(request, tokenizer, Limits())
    assert encoded.logical_tokens == 65536
    assert max(len(encoded.state) + len(b.ids) for b in encoded.branches) == 32768
    return request, tokenizer, encoded


def test_exact_both_boundaries_using_actual_byte_encoder(exact_request):
    request, tokenizer, encoded = exact_request
    before = copy.deepcopy(request.model_dump())
    evidence = native.verify_context_boundaries(request, tokenizer, encoded)
    assert evidence == {
        "aggregate_input": {"code": "context_overflow", "limit": "aggregate_input", "actual": 65536,
                            "maximum": 65535, "question_id": None},
        "state_plus_branch": {"code": "context_overflow", "limit": "state_plus_branch", "actual": 32768,
                              "maximum": 32767, "question_id": "route"},
    }
    assert request.model_dump() == before


@pytest.mark.parametrize("failure", [ValueError("tokenizer failure"), TypeError("unexpected type"),
                                     RuntimeError("runtime failure")])
def test_unrelated_errors_never_count_as_overflow(exact_request, monkeypatch, failure):
    def broken(*args):
        raise failure
    monkeypatch.setattr(native, "encode_request", broken)
    with pytest.raises(type(failure), match=str(failure)):
        native.verify_context_boundaries(*exact_request)


@pytest.mark.parametrize("field,value", [("limit", "questions"), ("actual", 65537), ("maximum", 1),
                                         ("question_id", "wrong"), ("code", "other")])
def test_wrong_overflow_detail_is_rejected(exact_request, monkeypatch, field, value):
    def broken(*args):
        error = ContextOverflow("aggregate_input", 65536, 65535)
        error.detail[field] = value
        raise error
    monkeypatch.setattr(native, "encode_request", broken)
    with pytest.raises(ValueError, match="unexpected native context overflow detail"):
        native.verify_context_boundaries(*exact_request)


def test_silent_boundary_acceptance_is_rejected(exact_request, monkeypatch):
    monkeypatch.setattr(native, "encode_request", lambda *args: exact_request[2])
    with pytest.raises(ValueError, match="overflow was not rejected"):
        native.verify_context_boundaries(*exact_request)


@pytest.mark.parametrize("mutation", ["logical_count", "branch_length", "missing_branch", "order"])
def test_incomplete_or_inexact_encoding_is_rejected(exact_request, mutation):
    request, tokenizer, encoded = exact_request
    if mutation == "logical_count":
        encoded = replace(encoded, logical_tokens=65535)
    elif mutation == "branch_length":
        branches = list(encoded.branches)
        branches[0] = replace(branches[0], ids=branches[0].ids[:-1])
        encoded = replace(encoded, branches=tuple(branches))
    elif mutation == "missing_branch":
        encoded = replace(encoded, branches=encoded.branches[:-1])
    else:
        encoded = replace(encoded, branches=tuple(reversed(encoded.branches)))
    with pytest.raises(ValueError, match="exact branch|every requested question"):
        native.verify_context_boundaries(request, tokenizer, encoded)


def test_all_branch_vectors_match_without_mutation(small_encoding):
    left = vectors(small_encoding)
    right = [x.clone() for x in left]
    before = [x.clone() for x in left]
    result = native.compare_branch_outputs(small_encoding, left, right, {"choice": 1., "score": 2., "noul": .5})
    assert result == {"passed": True, "maximum_logit_error": 0., "maximum_probability_error": 0.}
    assert all(torch.equal(a, b) for a, b in zip(left, before, strict=True))


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "iterator", "none"])
def test_every_output_path_has_exact_branch_cardinality(small_encoding, side, mutation):
    outputs = {"left": vectors(small_encoding), "right": vectors(small_encoding)}
    values = outputs[side]
    outputs[side] = {"missing": values[:-1], "extra": values + [values[0]],
                     "iterator": iter(values), "none": None}[mutation]
    with pytest.raises(ValueError, match="one output per branch"):
        native.compare_branch_outputs(small_encoding, outputs["left"], outputs["right"], {})


@pytest.mark.parametrize("mutation", ["broadcast", "missing_option", "extra_option", "scalar", "integer", "dtype", "not_tensor"])
def test_invalid_option_vectors_cannot_broadcast_into_pass(small_encoding, mutation):
    left, right = vectors(small_encoding), vectors(small_encoding)
    left[0] = {"broadcast": left[0][None], "missing_option": left[0][:1],
               "extra_option": torch.zeros(3), "scalar": torch.tensor(1.),
               "integer": left[0].long(), "dtype": left[0].double(), "not_tensor": [0., 1.]}[mutation]
    with pytest.raises(ValueError, match="complete option vectors"):
        native.compare_branch_outputs(small_encoding, left, right, {})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("side", [0, 1])
def test_nonfinite_vectors_are_rejected(small_encoding, value, side):
    paths = [vectors(small_encoding), vectors(small_encoding)]
    paths[side][0][0] = value
    with pytest.raises(ValueError, match="finite outputs"):
        native.compare_branch_outputs(small_encoding, *paths, {})


@pytest.mark.parametrize("temperature", [None, 0., -1., True, "1", float("nan"), float("inf")])
def test_invalid_calibration_cannot_pass(small_encoding, temperature):
    with pytest.raises(ValueError, match="positive calibration"):
        native.compare_branch_outputs(small_encoding, vectors(small_encoding), vectors(small_encoding),
                                      {"choice": temperature, "score": 1., "noul": 1.})


def test_logit_tolerance_is_not_relaxed(small_encoding):
    a, b = vectors(small_encoding), vectors(small_encoding)
    b[-1][-1] += 1.e-3
    result = native.compare_branch_outputs(small_encoding, a, b, {"choice": 1., "score": 1., "noul": 1.})
    assert result["passed"] is False
    assert result["maximum_logit_error"] > 1.e-4


def test_probability_tolerance_is_independent(small_encoding):
    a = [torch.zeros_like(t) for t in vectors(small_encoding)]
    b = [t.clone() for t in a]
    b[0][0] = 9.e-6
    b[0][1] = -9.e-6  # both pass logits; probabilities exceed atol PLUS rtol
    assert torch.allclose(a[0], b[0], atol=1.e-5, rtol=1.e-5)
    result = native.compare_branch_outputs(small_encoding, a, b, {"choice": .2, "score": 1., "noul": 1.})
    assert result["passed"] is False
    assert result["maximum_probability_error"] > 1.e-5


def test_protocol_thresholds_are_unchanged():
    assert native.PROTOCOL["version"] == "native-context-v3"
    assert native.PROTOCOL["required_branch_tokens"] == 32768
    assert native.PROTOCOL["required_aggregate_tokens"] == 65536
    for name in ("fp32_logit_atol", "fp32_logit_rtol", "probability_atol", "probability_rtol"):
        assert native.PROTOCOL[name] == 1.e-5
    for name in ("gradient_atol", "gradient_rtol"):
        assert native.PROTOCOL[name] == 2.e-5
    for name in ("hf_hidden_atol", "hf_hidden_rtol"):
        assert native.PROTOCOL[name] == 1.e-4
    assert native.PROTOCOL["evidence_marker_accuracy_minimum"] == .8


@pytest.mark.parametrize("version", ["4.57.0", "4.57.1+unreviewed", "5.0.0", ""])
def test_unpinned_reference_is_refused(monkeypatch, version):
    monkeypatch.setattr(native.metadata, "version", lambda name: version)
    with pytest.raises(ValueError, match="transformers==4.57.1"):
        native.require_reference_version()


def test_reference_distribution_absent(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(native.metadata, "version", missing)
    with pytest.raises(ValueError, match="transformers==4.57.1"):
        native.require_reference_version()


def test_reference_import_and_distribution_must_agree(monkeypatch):
    monkeypatch.setattr(native.metadata, "version", lambda name: "4.57.1")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="0.0.fake"))
    with pytest.raises(ValueError, match="disagrees"):
        native.hf_parity(None, [1, 2])


def test_correct_reference_distribution_version_is_recorded(monkeypatch):
    monkeypatch.setattr(native.metadata, "version", lambda name: "4.57.1")
    assert native.require_reference_version() == "4.57.1"


@pytest.fixture
def cuda_spy(monkeypatch):
    # Simulated topology only. Never claims that this runtime has CUDA devices.
    state = {"current": 0, "entered": []}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    @contextmanager
    def device(selected):
        previous = state["current"]
        state["current"] = torch.device(selected).index
        state["entered"].append(state["current"])
        try:
            yield
        finally:
            state["current"] = previous
    monkeypatch.setattr(torch.cuda, "device", device)
    return state


def test_default_cuda_resolves_current_device(cuda_spy):
    cuda_spy["current"] = 1
    assert native.measurement_device("cuda", "fp32") == torch.device("cuda:1")


def test_explicit_cuda_preserves_current_device(cuda_spy):
    assert native.measurement_device("cuda:1", "fp32") == torch.device("cuda:1")
    assert cuda_spy["current"] == 0


@pytest.mark.parametrize("supported", [False, True])
def test_bf16_capability_uses_selected_device_and_restores(cuda_spy, monkeypatch, supported):
    def capability():
        assert cuda_spy["current"] == 1
        return supported
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", capability)
    if supported:
        assert native.measurement_device("cuda:1", "bf16") == torch.device("cuda:1")
    else:
        with pytest.raises(ValueError, match="does not report BF16"):
            native.measurement_device("cuda:1", "bf16")
    assert cuda_spy["entered"] == [1] and cuda_spy["current"] == 0


@pytest.mark.parametrize("device,precision", [("cpu", "fp32"), ("cuda:2", "fp32"),
                                             ("cuda", "fp16"), ("meta", "bf16")])
def test_invalid_device_or_precision_cannot_fall_back(cuda_spy, device, precision):
    with pytest.raises(ValueError):
        native.measurement_device(device, precision)


def test_unavailable_cuda_precedes_load_and_output_writes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(native, "load_checkpoint", lambda *args: calls.append("load"))
    target = tmp_path / "untouched-output"
    with pytest.raises(ValueError, match="native validation requires CUDA"):
        native.run(tmp_path / "absent.pt", target, "cuda")
    assert not target.exists() and calls == []


@pytest.fixture
def orchestration(monkeypatch, exact_request):
    # All model/native/hardware prerequisites are EXPLICIT DOUBLES. This tests
    # that the real harness rejects bad evidence; no native model was executed.
    from kev_laya import exposure
    request, byte_tokenizer, encoded = exact_request
    class TokenizerDouble:
        serialization = "kev-laya-prefix-v2"
        special = byte_tokenizer.special
        def encode(self, text):
            return byte_tokenizer.encode(text)
    class ModelDouble:
        native_weights_loaded = True
        training_steps = 1
        calibration_provenance = {"status": "fitted-held-out", "fits": {
            kind: {"status": "fitted-on-calibration", "count": 1}
            for kind in ("choice", "score", "noul")}}
        temperatures = {"choice": 1., "score": 1., "noul": 1.}
        cfg = SimpleNamespace(revision=native.PIN)
        batch_policy = SimpleNamespace(to_dict=lambda: {"fixture_only": True})
        behavior = "valid"
        def parameters(self):
            yield torch.zeros(1)  # Actual CPU, never represented as native hardware
        def eval(self):
            return self
        def __call__(self, encoding, reference=False):
            results = []
            for branch in encoding.branches:
                z = torch.zeros(len(branch.question.options()))
                expected = {"route": "billing", "urgency": "7", "escalate": "true"}[branch.question_id]
                index = [k for k, v in branch.question.options()].index(expected)
                z[index] = 1.
                if self.behavior == "broadcast" and not reference:
                    z = z[None]
                results.append(z)
            if self.behavior == "missing":
                results = results[:1]
            return results, {"forward_tokens": encoding.logical_tokens}
    model = ModelDouble()
    device = torch.device("cpu")  # deliberate isolated CPU orchestration test
    @contextmanager
    def selected_device(selected):
        # The test uses CPU tensors explicitly; no real CUDA context is entered.
        assert selected == device
        yield
    monkeypatch.setattr(torch.cuda, "device", selected_device)
    monkeypatch.setattr(native, "measurement_device", lambda *args: device, raising=False)
    monkeypatch.setattr(native, "require_reference_version", lambda: "4.57.1", raising=False)
    monkeypatch.setattr(native, "load_checkpoint", lambda *args: (model, TokenizerDouble(), {"checkpoint_sha256": "0" * 64}))
    monkeypatch.setattr(native, "make_case", lambda *args: request)
    monkeypatch.setattr(native, "hf_parity", lambda *args: {"passed": True, "scope": "injected-not-native"})
    monkeypatch.setattr(exposure, "verify_exposure", lambda *args, **kwargs: {"native_32k_64k": True, "scope": "injected"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    for name in ("reset_peak_memory_stats", "synchronize", "max_memory_allocated"):
        def record(selected=None, _name=name):
            calls.append((_name, selected))
            return 1 if _name == "max_memory_allocated" else None
        monkeypatch.setattr(torch.cuda, name, record)
    # Use a preconstructed REAL byte encoding to avoid relabeling the double's
    # serialization as native tokenization. Only overflow exceptions are encoded.
    original_encoder = native.encode_request
    def encoding(r, tokenizer, limits):
        if r is request:
            if limits.aggregate < 65536 or limits.branch < 32768:
                return original_encoder(r, byte_tokenizer, limits)
            return encoded
        return original_encoder(r, byte_tokenizer, limits)
    monkeypatch.setattr(native, "encode_request", encoding)
    # Keep any simulated passing native-report dict strictly inside this test.
    writes = []
    monkeypatch.setattr(native, "atomic_json", lambda path, value: writes.append((path.name, copy.deepcopy(value))))
    return SimpleNamespace(model=model, calls=calls, writes=writes, encoder=encoding, device=device)


@pytest.mark.parametrize("behavior", ["missing", "broadcast"])
def test_real_run_rejects_incomplete_or_broadcast_outputs(orchestration, tmp_path, behavior):
    orchestration.model.behavior = behavior
    with pytest.raises(ValueError, match="one output per branch|complete option vectors"):
        native.run(tmp_path / "injected.pt", tmp_path / "out", torch.device("cpu"))
    assert not any(name == "report.json" for name, _ in orchestration.writes)


def test_real_run_rejects_partial_calibration(orchestration, tmp_path):
    orchestration.model.calibration_provenance['fits']['choice'] = {
        'status': 'unfitted-no-samples', 'count': 0}
    with pytest.raises(ValueError, match='held-out calibrated artifact'):
        native.run(tmp_path / 'injected.pt', tmp_path / 'out', torch.device('cpu'))


def test_real_run_propagates_nonoverflow_error(orchestration, monkeypatch, tmp_path):
    def broken(request, tokenizer, limits):
        if limits.aggregate == 65535:
            raise ValueError("injected tokenizer failure, not a token overflow")
        return orchestration.encoder(request, tokenizer, limits)
    monkeypatch.setattr(native, "encode_request", broken)
    with pytest.raises(ValueError, match="injected tokenizer failure"):
        native.run(tmp_path / "injected.pt", tmp_path / "out", torch.device("cpu"))
    assert not any(name == "report.json" for name, _ in orchestration.writes)


def test_real_run_measures_selected_device_and_records_both_boundaries(orchestration, tmp_path):
    report = native.run(tmp_path / "injected.pt", tmp_path / "out", torch.device("cpu"))
    assert report["passed"]  # SIMULATED only; not written or exported as native proof
    assert len(report["results"]) == 6
    assert all(selected == orchestration.device for name, selected in orchestration.calls)
    assert {name for name, _ in orchestration.calls} == {"reset_peak_memory_stats", "synchronize", "max_memory_allocated"}
    for record in report["results"]:
        assert record["measurement_device"] == "cpu"  # explicit injected topology
        assert record["overflow_rejected"] is True
        assert set(record["overflow_evidence"]) == {"aggregate_input", "state_plus_branch"}
        assert len(record["decisions"]) == 3
    assert report["general_quality"] == report["jev_relative_quality"] == "unverified"
    # This validates JSON compatibility, not provenance authenticity.
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", [None, "autocast_constructor", "model"])
def test_bf16_autocast_uses_selected_device_and_restores_on_exit(
        orchestration, cuda_spy, monkeypatch, tmp_path, failure):
    # Real PyTorch autocast construction with an explicitly injected CUDA
    # topology; tensors, model, oracle and exposure remain the CPU doubles.
    # Device 0 cannot run BF16, while the requested device 1 can.
    selected = torch.device("cuda:1")
    monkeypatch.setattr(native, "measurement_device", lambda *args: selected)
    monkeypatch.setattr(orchestration.model, "parameters", lambda: iter([
        SimpleNamespace(device=selected, dtype=torch.float32)]))
    checked_devices = []
    def capability():
        checked_devices.append(cuda_spy["current"])
        assert cuda_spy["current"] == 1
        if failure == "autocast_constructor":
            raise RuntimeError("injected capability failure during autocast construction")
        return True
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", capability)
    if failure == "model":
        def broken(*args, **kwargs):
            assert cuda_spy["current"] == 1
            raise RuntimeError("injected model failure inside autocast")
        monkeypatch.setattr(type(orchestration.model), "__call__", broken)
    if failure is None:
        report = native.run(tmp_path / "injected.pt", tmp_path / "out", selected, "bf16")
        assert report["passed"]  # in-memory orchestration result, never native proof
        assert checked_devices == [1] * 6
        assert cuda_spy["entered"] == [1] * 6
    else:
        with pytest.raises(RuntimeError, match="injected capability|injected model"):
            native.run(tmp_path / "injected.pt", tmp_path / "out", selected, "bf16")
        assert checked_devices == [1]
        assert cuda_spy["entered"] == [1]
        assert not any(name == "report.json" for name, _ in orchestration.writes)
    assert cuda_spy["current"] == 0
    assert all(device == selected for _, device in orchestration.calls)


@pytest.mark.parametrize("mutation", ["valid", "row_count", "hidden_width", "dtype", "nan", "tolerance"])
def test_oracle_output_contract_with_explicit_reference_double(monkeypatch, tiny, mutation):
    # Real tiny CPU backbone + synthetic oracle output, not pinned Transformers.
    tiny.eval()
    ids = [1, 2, 3]
    with torch.no_grad():
        hidden, _ = tiny.backbone(ids)
    reference = hidden.clone()
    if mutation == "row_count":
        reference = reference[:-1]
    elif mutation == "hidden_width":
        reference = reference[:, :-1]
    elif mutation == "dtype":
        reference = reference.double()
    elif mutation == "nan":
        reference[0, 0] = float("nan")
    elif mutation == "tolerance":
        reference[0, 0] += .1
    class ConfigDouble:
        def __init__(self, **kwargs):
            self.fields = kwargs
    class OracleDouble:
        def __init__(self, config):
            assert config._attn_implementation == "eager"
            assert config.fields["hidden_size"] == tiny.cfg.hidden_size
        def to(self, device):
            assert device == torch.device("cpu")
            return self
        def load_state_dict(self, weights, strict):
            assert weights and strict is True
        def eval(self):
            return self
        def __call__(self, input_ids, use_cache):
            assert input_ids.tolist() == [ids] and use_cache is False
            return SimpleNamespace(last_hidden_state=reference[None])
    monkeypatch.setattr(native.metadata, "version", lambda name: "4.57.1")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        __version__="4.57.1", Qwen2Config=ConfigDouble, Qwen2Model=OracleDouble))
    if mutation in {"row_count", "hidden_width", "dtype", "nan"}:
        with pytest.raises(ValueError, match="matching complete hidden|finite hidden"):
            native.hf_parity(tiny, ids)
    else:
        result = native.hf_parity(tiny, ids)
        assert result["passed"] is (mutation == "valid")
        assert result["transformers_version"] == "4.57.1"


def test_reference_prerequisite_precedes_weights_and_output(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(native, "measurement_device", lambda *args: torch.device("cuda:0"))
    monkeypatch.setattr(native.metadata, "version", lambda name: "unreviewed")
    monkeypatch.setattr(native, "load_checkpoint", lambda *args: calls.append("load"))
    target = tmp_path / "untouched"
    with pytest.raises(ValueError, match="transformers==4.57.1"):
        native.run(tmp_path / "no-checkpoint.pt", target)
    assert calls == [] and not target.exists()
