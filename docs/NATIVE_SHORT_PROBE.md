# Short pinned-weight CUDA parity probe

This probe supports Layev issues #2 and #3. It verifies a pinned Qwen weight
snapshot, reloads a local checkpoint, checks its tokenizer and preprocessing
identity, compares the backbone hidden states to pinned Transformers 4.57.1,
and runs one short native forward. It is deliberately separate from the full
trained/context validator; a passing report does not prove training, resume,
serving, long-context exposure, calibration, quality or Jev comparison.

Use a separate CUDA environment with Python 3.13.12, PyTorch 2.10.0+cu128,
Transformers 4.57.1 and tokenizers 0.22.1. Install the current Layev source
in that environment and acquire the pinned Qwen snapshot with
`scripts/download_backbone.py` into a fresh private directory outside Git.
Run `scripts/verify_backbone_source.py` on the actual bytes first. Then use:

```sh
python scripts/verify_short_native.py --source <verified-snapshot> \
  --checkpoint <initialized-or-trained-checkpoint> \
  --out <new-private-report.json> --device cuda:0
```

The runner refuses to overwrite a report. It exits nonzero if pinned source,
checkpoint identity or the fixed hidden-state comparison fails. Its report
records exact model and runner source hashes, weight/checkpoint/tokenizer hashes,
device/runtime, token count, timing, peak allocated GPU memory and explicit
untested dimensions. The independent reference uses the repository's unchanged
`1e-4` absolute/relative hidden-state comparison; the maximum absolute
difference alone can exceed `1e-4` while the elementwise combined tolerance
still passes.

The [original failed probe](evidence/native/short-cuda-before-20260927.json)
used actual pinned Qwen weights and a 62-token request on an RTX 3060. CUDA
SDPA yielded a maximum hidden-state difference of `0.001323699951171875`
and failed the unchanged gate. Its receipt SHA-256 is
`1c11bee2e9f26079708f09622ca9632eebe659823890016459a2df8af4291413`.
The source diagnosis found that short CUDA SDPA
reductions differ from the pinned Transformers eager attention arithmetic.
The targeted fix uses explicit eager attention only for CUDA requests with at
most 256 total key/value tokens. Longer CUDA sequences retain the existing
fused-attention requirement and refusal behavior.

The [fresh corrected probe](evidence/native/short-cuda-after-20260927.json)
passed on the same initialized checkpoint and source weights. It recorded 62
tokens, maximum hidden-state difference `0.0002231597900390625`, native
forward `0.09524280000186991` seconds and peak allocated GPU memory
`4058755584` bytes. The checkpoint has **zero training steps** and unfitted
calibration. These are local measurements for that device/input, not a
performance or native-training guarantee. The raw failure and corrected
reports are preserved, with no weights, private data or machine paths. The
corrected receipt SHA-256 is
`c96199a9178111fddc02e888e378be0dee72ee0e6d7c89421289b7042c375f19`.
