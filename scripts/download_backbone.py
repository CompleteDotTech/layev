"""Explicit free public-weight acquisition. No API evaluation or cloud job launch."""
import argparse
from pathlib import Path
import urllib.request
from kev_laya.io import sha256_file, atomic_json

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
if a.out.exists() and any(a.out.iterdir()):
    raise FileExistsError("choose an empty backbone directory")
a.out.mkdir(parents=True, exist_ok=True)
repo, revision = "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"
files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "LICENSE"]
checksums = {}
for name in files:
    destination = a.out / name
    temporary = destination.with_suffix(destination.suffix + ".partial")
    urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", temporary)
    temporary.replace(destination)
    checksums[name] = sha256_file(destination)
atomic_json(a.out / "source.json", {"repository": repo, "revision": revision, "license": "Apache-2.0", "sha256": checksums})
print(a.out / "source.json")
