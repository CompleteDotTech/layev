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
oracle source hash,
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

## Two-step trained checkpoint and independent oracle

The [portable training receipt](evidence/native/two-step-train-resume-serve-20260927.md)
records a two-step FP32 LoRA/activation-checkpointed CUDA run from the pinned
initialization, an interruption after step 1, same-policy resume to step 2,
checkpoint hashes, preprocessing identities, measured resource samples and a
loopback trained service response. Its 64-row suite is generated smoke data,
with no representative-quality or 32k/64k exposure claim. The step-2 checkpoint
SHA-256 is `b6e0db32a65bbf240a5cdec8c938040054a45284e5d4cf35cd310160262b6efb`.

The [first trained oracle attempt](evidence/native/trained-oracle-before-20260927.json)
failed the unchanged hidden-state gate with maximum absolute difference
`0.000293731689453125`. Its receipt SHA-256 is
`42b6e911b7d8b0af96d24da69dd316362ae131a5e2b095442444a53daa70ff40`.
The old oracle had multiplied trained LoRA matrices into each base matrix;
that changes FP32 reduction order relative to deployed two-linear LoRA.
The corrected independent Transformers backbone keeps the original base weights
and attaches inference-only two-linear LoRA adapters using the saved matrices.
It does not change model or optimizer execution. The [final initialized](evidence/native/short-cuda-final-init-20260927.json)
and [trained](evidence/native/short-cuda-final-trained-20260927.json) reports
both pass the unchanged combined tolerance. Their receipt SHA-256 values are
`7dbba09ca49259708cbff0003781347289330495ed2d2a9825720bd7374696be`
and `b6f14e189fc213dfc83eabdadeb357a5ede35b5fd48c00a6dcf362cd7574eb7e`.
The trained report has two steps, 62 inference tokens and maximum absolute
hidden-state difference `0.00011444091796875`. A separate GPU regression test
uses the actual trained checkpoint when `KEV_LAYA_TRAINED_NATIVE_CHECKPOINT`
is set. Full-weight and BF16 modes, broader gradient parity and long-context
training remain open.
