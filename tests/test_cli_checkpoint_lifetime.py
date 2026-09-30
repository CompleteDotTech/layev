"""CPU-only regression for CLI checkpoint payload ownership."""
import gc
import json
import weakref

import pytest
import torch

from kev_laya import cli


@pytest.mark.parametrize("command", ["train", "reward-train"])
@pytest.mark.parametrize("initial_arg", ["--init", "--resume"])
def test_training_cli_releases_checkpoint_payload(tmp_path, monkeypatch, command, initial_arg):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"training": {"steps": 2}, "objective": {"reinforce": 1.0}}))
    observed = {}
    digest = "a" * 64

    def load_checkpoint(path, device):
        tensor = torch.empty(1, device="cpu")
        observed["payload_tensor"] = weakref.ref(tensor)
        return object(), object(), {"checkpoint_sha256": digest, "model": {"weight": tensor}}

    def train(model, tokenizer, data, manifest, output, settings, objective, limits, **kwargs):
        assert settings.steps == 2
        gc.collect()
        assert observed["payload_tensor"]() is None, "CLI retained checkpoint tensor during train"
        assert kwargs["parent_sha256"] == (digest if initial_arg == "--init" else None)
        assert kwargs["resume"] == (tmp_path / "checkpoint.pt" if initial_arg == "--resume" else None)
        return {"state": {"step": 1}}

    monkeypatch.setattr(cli, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(cli, "load_suite", lambda path: ({"train": []}, {}))
    monkeypatch.setattr(cli, "train", train)
    assert cli.main([command, "--suite", str(tmp_path), "--out", str(tmp_path / "run"),
                     "--config", str(config), initial_arg, str(tmp_path / "checkpoint.pt")]) == 0
