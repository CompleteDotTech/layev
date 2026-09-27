"""Real Safetensors/CPU weight I/O; synthetic tiny weights are NOT native Qwen.

CLI ordering tests explicitly replace the unavailable Qwen tokenizer and writer.
No test downloads weights, touches a workstation, or establishes native quality.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors import SafetensorError
from safetensors import torch as safe

from kev_laya import cli
from kev_laya.model import BackboneConfig, DecisionEngine


@pytest.fixture(autouse=True)
def preserve_rng():
    old = torch.get_rng_state()
    threads = torch.get_num_threads()
    yield
    torch.set_rng_state(old)
    torch.set_num_threads(threads)


def config(rank=0):
    return BackboneConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1,
                          pointer_dim=4, lora_rank=rank)


def source_tensors(value=0.125, dtype=torch.float32):
    return {"model." + k: torch.full_like(v, value, dtype=dtype)
            for k, v in DecisionEngine(config()).backbone.state_dict().items()}


def write_source(path, value=0.125, dtype=torch.float32):
    values = source_tensors(value, dtype)
    raw = safe.save(values)
    path.write_bytes(raw)
    return values, raw, hashlib.sha256(raw).hexdigest()


def assert_weights(model, values):
    for key, value in model.backbone.state_dict().items():
        if not key.endswith((".a", ".b")):
            assert torch.equal(value, values["model." + key.replace(".base.", ".")].to(value.dtype))


@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_exact_real_safetensors_load_and_digest(tmp_path, rank, dtype):
    path = tmp_path / "weights.safetensors"
    values, _, digest = write_source(path, dtype=dtype)
    model = DecisionEngine(config(rank))
    nonbackbone = {k: v.clone() for k, v in model.state_dict().items()
                   if k.startswith("head.") or k.endswith((".a", ".b"))}
    model.load_qwen_weights(path, expected_sha256=digest)
    assert_weights(model, values)
    assert model.native_weights_loaded is True
    assert model.loaded_backbone_sha256 == digest
    for k, value in nonbackbone.items():
        assert torch.equal(model.state_dict()[k], value)


@pytest.mark.parametrize("rank", [0, 2])
def test_path_replacement_does_not_misattribute_loaded_weights(tmp_path, monkeypatch, rank):
    path = tmp_path / "weights.safetensors"
    values, _, digest = write_source(path)
    replacement = tmp_path / "replacement.safetensors"
    write_source(replacement, value=0.25)
    load_bytes, load_file = safe.load, safe.load_file
    calls = []

    def replace_after_parse(loader, *args, **kwargs):
        tensors = loader(*args, **kwargs)
        replacement.replace(path)
        calls.append("replaced")
        return tensors

    monkeypatch.setattr(safe, "load", lambda *a, **kw: replace_after_parse(load_bytes, *a, **kw))
    monkeypatch.setattr(safe, "load_file", lambda *a, **kw: replace_after_parse(load_file, *a, **kw))
    model = DecisionEngine(config(rank))
    model.load_qwen_weights(path)
    assert calls == ["replaced"]
    assert_weights(model, values)
    assert model.loaded_backbone_sha256 == digest
    assert hashlib.sha256(path.read_bytes()).hexdigest() != digest


@pytest.mark.parametrize("bad", ["", "0" * 63, "G" * 64, "A" * 64, 123, [], True])
def test_bad_expected_digest_rejected_before_source_read(tmp_path, bad):
    model = DecisionEngine(config())
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        model.load_qwen_weights(tmp_path / "absent", expected_sha256=bad)
    assert not model.native_weights_loaded


def test_expected_digest_checked_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "weights.safetensors"
    write_source(path)
    monkeypatch.setattr(safe, "load", lambda *_: pytest.fail("mismatch must precede parser"))
    model = DecisionEngine(config())
    with pytest.raises(ValueError, match="checksum mismatch"):
        model.load_qwen_weights(path, expected_sha256="0" * 64)
    assert not model.native_weights_loaded


@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("defect", ["shape", "missing", "extra", "nan", "infinity", "integer", "malformed", "overflow"])
def test_bad_reload_preserves_every_previous_parameter(tmp_path, rank, defect):
    path = tmp_path / "weights.safetensors"
    _, _, digest = write_source(path)
    model = DecisionEngine(config(rank))
    model.load_qwen_weights(path)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    values = source_tensors(0.25)
    key = "model.norm.weight"
    if defect == "shape":
        values[key] = torch.zeros(3)
    elif defect == "missing":
        values.pop(key)
    elif defect == "extra":
        values["model.unexpected"] = torch.ones(1)
    elif defect == "nan":
        values[key][0] = float("nan")
    elif defect == "infinity":
        values[key][0] = float("inf")
    elif defect == "integer":
        values[key] = torch.ones(8, dtype=torch.int64)
    elif defect == "malformed":
        # Invalid serialized bytes must be rejected without changing the model.
        path.write_bytes(b"not a valid tensor container")
    else:
        values[key] = torch.full((8,), 1e300, dtype=torch.float64)
    if defect != "malformed":
        path.write_bytes(safe.save(values))
    expected_error = SafetensorError if defect == "malformed" else (ValueError, RuntimeError)
    with pytest.raises(expected_error):
        model.load_qwen_weights(path)
    assert model.native_weights_loaded is True
    assert model.loaded_backbone_sha256 == digest
    for k, value in before.items():
        assert torch.equal(model.state_dict()[k], value)


@pytest.mark.parametrize("rank", [0, 2])
def test_unexpected_copy_failure_clears_success_receipt(tmp_path, monkeypatch, rank):
    path = tmp_path / "weights.safetensors"
    write_source(path)
    model = DecisionEngine(config(rank))
    model.load_qwen_weights(path)
    write_source(path, value=0.25)

    def broken(*args, **kwargs):
        with torch.no_grad():
            model.backbone.norm.weight.add_(1)
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(model.backbone, "load_state_dict", broken)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        model.load_qwen_weights(path)
    assert model.native_weights_loaded is False
    assert model.loaded_backbone_sha256 is None


def test_non_backbone_head_keys_are_not_imported(tmp_path):
    path = tmp_path / "weights.safetensors"
    values = source_tensors()
    values["lm_head.weight"] = torch.zeros(3, 7)
    path.write_bytes(safe.save(values))
    model = DecisionEngine(config())
    model.load_qwen_weights(path)
    assert_weights(model, values)


def test_empty_backbone_is_refused(tmp_path):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(safe.save({"lm_head.weight": torch.ones(1)}))
    with pytest.raises(ValueError, match="no model backbone"):
        DecisionEngine(config()).load_qwen_weights(path)


def make_snapshot(tmp_path):
    """Artificial declaration, NOT a pinned upstream artifact or native receipt."""
    root = tmp_path / "source"
    root.mkdir()
    cfg = asdict(config()) | {"model_type": "qwen2", "hidden_act": "silu"}
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "tokenizer.json").write_text('{"explicit_test_double":true}')
    (root / "tokenizer_config.json").write_text('{}')
    (root / "LICENSE").write_text('Artificial test input, not an upstream license.')
    write_source(root / "model.safetensors")
    manifest = {"repository": "Qwen/Qwen2.5-0.5B", "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
                "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}}
    (root / "source.json").write_text(json.dumps(manifest))
    return root, manifest


class TokenizerDouble:
    special = (256, 257, 258, 259, 260)

    def __init__(self, path):
        self.path = path
        self.identity = "qwen-tokenizer-sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    def metadata(self):
        return {"vocab_size": 261}


def invoke(root, out):
    return cli.main(["native-init", "--backbone-dir", str(root), "--out", str(out), "--lora-rank", "0"])


def test_cli_tokenizer_failure_precedes_model_allocation(tmp_path, monkeypatch):
    root, _ = make_snapshot(tmp_path)
    events = []

    def tokenizer(path):
        events.append("tokenizer")
        raise ValueError("intentional tokenizer failure")

    def model(cfg):
        events.append("model")
        return DecisionEngine(cfg)

    monkeypatch.setattr(cli, "QwenTokenizer", tokenizer)
    monkeypatch.setattr(cli, "DecisionEngine", model)
    with pytest.raises(ValueError, match="intentional tokenizer failure"):
        invoke(root, tmp_path / "new" / "init.pt")
    assert events == ["tokenizer"]
    assert not (tmp_path / "new").exists()


def test_cli_existing_output_fails_before_loading_any_input(tmp_path):
    out = tmp_path / "existing.pt"
    out.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        invoke(tmp_path / "absent", out)
    assert out.read_bytes() == b"preserve"


@pytest.mark.parametrize("bad", [None, [], "bad", {"config.json": "bad"}])
def test_cli_malformed_manifest_is_explicit(tmp_path, monkeypatch, bad):
    root, manifest = make_snapshot(tmp_path)
    manifest["sha256"] = bad
    (root / "source.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cli, "DecisionEngine", lambda *_: pytest.fail("no model allocation"))
    with pytest.raises(ValueError, match="manifest"):
        invoke(root, tmp_path / "out.pt")


@pytest.mark.parametrize("mode", ["config_changed", "tokenizer_changed", "vocab_too_large"])
def test_cli_source_identity_checked_before_model(tmp_path, monkeypatch, mode):
    root, manifest = make_snapshot(tmp_path)
    if mode == "config_changed":
        (root / "config.json").write_text('{}')
    elif mode == "tokenizer_changed":
        (root / "tokenizer.json").write_text('{"changed":true}')
    class TooLarge(TokenizerDouble):
        def metadata(self):
            return {"vocab_size": 262}
    monkeypatch.setattr(cli, "QwenTokenizer", TooLarge if mode == "vocab_too_large" else TokenizerDouble)
    monkeypatch.setattr(cli, "DecisionEngine", lambda *_: pytest.fail("no model allocation"))
    with pytest.raises(ValueError):
        invoke(root, tmp_path / "out.pt")


def test_cli_passes_manifest_digest_and_single_tokenizer_to_writer(tmp_path, monkeypatch):
    root, manifest = make_snapshot(tmp_path)
    events = []
    tokenizers = []
    class Tokenizer(TokenizerDouble):
        def __init__(self, path):
            super().__init__(path)
            tokenizers.append(self)
            events.append("tokenizer")
    class Model(DecisionEngine):
        def __init__(self, cfg):
            events.append("model")
            super().__init__(cfg)
        def load_qwen_weights(self, path, *, expected_sha256=None):
            assert expected_sha256 == manifest["sha256"]["model.safetensors"]
            events.append("weights")
            return super().load_qwen_weights(path, expected_sha256=expected_sha256)
    def save(path, model, tokenizer, **kwargs):
        assert tokenizer is tokenizers[0]
        assert model.loaded_backbone_sha256 == manifest["sha256"]["model.safetensors"]
        events.append("save")
        return {"scope": "test-writer-double"}
    monkeypatch.setattr(cli, "QwenTokenizer", Tokenizer)
    monkeypatch.setattr(cli, "DecisionEngine", Model)
    monkeypatch.setattr(cli, "save_checkpoint", save)
    assert invoke(root, tmp_path / "out.pt") == 0
    assert events == ["tokenizer", "model", "weights", "save"]
    assert len(tokenizers) == 1


def test_cli_changed_weights_between_preflight_and_load_are_rejected(tmp_path, monkeypatch):
    root, _ = make_snapshot(tmp_path)
    monkeypatch.setattr(cli, "QwenTokenizer", TokenizerDouble)
    def model(cfg):
        result = DecisionEngine(cfg)
        write_source(root / "model.safetensors", value=0.25)
        return result
    monkeypatch.setattr(cli, "DecisionEngine", model)
    monkeypatch.setattr(cli, "save_checkpoint", lambda *_a, **_kw: pytest.fail("must not export mismatched weights"))
    with pytest.raises(ValueError, match="native weight checksum mismatch"):
        invoke(root, tmp_path / "new" / "out.pt")
    assert not (tmp_path / "new").exists()


def test_cli_configuration_uses_exact_verified_bytes(tmp_path, monkeypatch):
    root, _ = make_snapshot(tmp_path)
    class Tokenizer(TokenizerDouble):
        def __init__(self, path):
            super().__init__(path)
            (root / "config.json").write_text('{}')
    monkeypatch.setattr(cli, "QwenTokenizer", Tokenizer)
    seen = []
    def save(path, model, tokenizer, **kw):
        seen.append(model.cfg.hidden_size)
        return {"scope": "test-double"}
    monkeypatch.setattr(cli, "save_checkpoint", save)
    assert invoke(root, tmp_path / "out.pt") == 0
    assert seen == [8]


def test_cli_existing_manifest_is_preserved_without_reading_sources(tmp_path):
    out = tmp_path / "checkpoint.pt"
    manifest = out.with_suffix('.manifest.json')
    manifest.write_bytes(b"preserve existing receipt")
    with pytest.raises(FileExistsError):
        invoke(tmp_path / "absent", out)
    assert manifest.read_bytes() == b"preserve existing receipt"
    assert not out.exists()


def test_cli_output_appearing_during_load_is_preserved(tmp_path, monkeypatch):
    root, _ = make_snapshot(tmp_path)
    out = tmp_path / "out.pt"
    monkeypatch.setattr(cli, "QwenTokenizer", TokenizerDouble)
    class Model(DecisionEngine):
        def load_qwen_weights(self, path, **kw):
            super().load_qwen_weights(path, **kw)
            out.write_bytes(b"concurrent result")
    monkeypatch.setattr(cli, "DecisionEngine", Model)
    monkeypatch.setattr(cli, "save_checkpoint", lambda *_a, **_kw: pytest.fail("must not replace result"))
    with pytest.raises(FileExistsError):
        invoke(root, out)
    assert out.read_bytes() == b"concurrent result"


@pytest.mark.parametrize("rank", [0, 2])
def test_failed_reload_retains_finite_matching_forward_outputs(tmp_path, rank):
    path = tmp_path / 'weights.safetensors'
    write_source(path)
    model = DecisionEngine(config(rank)).eval()
    with torch.inference_mode():
        before, _ = model.backbone((1, 2, 3))
    bad = source_tensors(0.25)
    bad['model.norm.weight'] = torch.ones(9)
    path.write_bytes(safe.save(bad))
    with pytest.raises((ValueError, RuntimeError)):
        model.load_qwen_weights(path)
    with torch.inference_mode():
        after, _ = model.backbone((1, 2, 3))
    assert torch.isfinite(after).all()
    assert torch.equal(before, after)
