"""CPU-only trainer integration contracts; run after candidate source integration."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import kev_laya.training as training
from kev_laya.encoding import ByteTokenizer, Limits
from kev_laya.objectives import ObjectiveConfig
from kev_laya.schema import canonical


LEGACY_HASHES = {
    "default": "4c67e545c11fb19196a5298b56b144f2e334059b2735b6573b814521dde08100",
    "fused": "1d88349ed3bf464bb39a83d257f1f2d58a8db54a8ba24fecc047c92794ee46af",
}


class BeforeOptimizer(Exception):
    pass


@pytest.mark.parametrize("backend,budget", [
    ("cpu_state_streaming_v1", None),
    ("cpu_state_streaming_v1", 0),
    ("cpu_state_streaming_v1", -1),
    ("cpu_state_streaming_v1", True),
    ("default", 1),
    ("fused", 1),
])
def test_backend_budget_is_explicit_and_fail_closed(backend, budget):
    with pytest.raises(ValueError):
        training.TrainSettings(optimizer_backend=backend,
                               optimizer_state_max_bytes=budget)


def _capture_config_hash(monkeypatch, tmp_path: Path, tiny, backend: str,
                         *, pretend_cuda: bool = False) -> str:
    seen = []
    original_canonical = training.canonical

    def capture(value):
        if isinstance(value, dict) and set(value) >= {"settings", "objective", "limits"}:
            seen.append(original_canonical(value))
        return original_canonical(value)

    def stop(*args, **kwargs):
        raise BeforeOptimizer

    with monkeypatch.context() as patch:
        patch.setattr(training, "canonical", capture)
        patch.setattr(training.torch.optim, "AdamW", stop)
        patch.setattr(training, "CPUStateStreamingAdamW", stop)
        if pretend_cuda:
            original_next = next
            patch.setattr(training, "next", lambda iterator: SimpleNamespace(
                device=torch.device("cuda")) if iterator is not None else original_next(iterator),
                          raising=False)
        settings = training.TrainSettings(steps=1, optimizer_backend=backend,
            **({"optimizer_state_max_bytes": 1_000_000} if backend == "cpu_state_streaming_v1" else {}))
        try:
            training.train(tiny, ByteTokenizer(), [], {"train": "frozen"},
                           tmp_path / (backend + "-capture"), settings,
                           ObjectiveConfig(), Limits(512, 8192))
        except BeforeOptimizer:
            pass
        except ValueError as exc:
            if backend != "fused" or pretend_cuda or "requires CUDA" not in str(exc):
                raise
        else:
            raise AssertionError("config capture ran past optimizer construction")
    assert len(seen) == 1
    return hashlib.sha256(seen[0].encode()).hexdigest()


@pytest.mark.parametrize("backend", ["default", "fused"])
def test_legacy_config_hash_continuity(monkeypatch, tmp_path, tiny, backend):
    assert _capture_config_hash(monkeypatch, tmp_path, tiny, backend) == LEGACY_HASHES[backend]


@pytest.mark.parametrize("source_backend,target_backend", [
    ("default", "cpu_state_streaming_v1"),
    ("fused", "cpu_state_streaming_v1"),
    ("cpu_state_streaming_v1", "default"),
    ("cpu_state_streaming_v1", "fused"),
])
def test_cross_backend_resume_refused_before_update(monkeypatch, tmp_path, tiny,
                                                     source_backend, target_backend):
    source_hash = _capture_config_hash(monkeypatch, tmp_path, tiny,
                                       source_backend, pretend_cuda=True)
    target_hash = _capture_config_hash(monkeypatch, tmp_path, tiny,
                                       target_backend, pretend_cuda=True)
    assert source_hash != target_hash
    manifest = {"train": "frozen"}
    data_hash = hashlib.sha256(canonical(manifest).encode()).hexdigest()

    def fake_load(*args, **kwargs):
        return tiny, ByteTokenizer(), {"execution": {}, "optimizer": {}, "sampler": {},
            "training_state": {"data_sha256": data_hash, "config_sha256": source_hash}}

    def cpu_optimizer(named_parameters, *, lr, weight_decay, max_state_bytes):
        return torch.optim.AdamW((p for _, p in named_parameters), lr=lr,
                                 weight_decay=weight_decay)

    with monkeypatch.context() as patch:
        patch.setattr(training, "next", lambda iterator: SimpleNamespace(
            device=torch.device("cuda")), raising=False)
        patch.setattr(training, "load_checkpoint", fake_load)
        patch.setattr(training, "CPUStateStreamingAdamW", cpu_optimizer)
        target = training.TrainSettings(steps=1, optimizer_backend=target_backend,
            **({"optimizer_state_max_bytes": 1_000_000}
               if target_backend == "cpu_state_streaming_v1" else {}))
        with pytest.raises(ValueError, match="resume configuration or frozen data differs"):
            training.train(tiny, ByteTokenizer(), [object()], manifest,
                           tmp_path / (source_backend + "-to-" + target_backend),
                           target, ObjectiveConfig(), Limits(512, 8192),
                           resume=tmp_path / "synthetic-prior.pt")
