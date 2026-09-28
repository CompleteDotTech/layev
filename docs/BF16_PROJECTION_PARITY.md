# CUDA BF16 backbone projection parity

The native Qwen path previously produced different BF16 projection values for
identical suffix tokens when the same request was evaluated as cached parallel
branches and as complete independent rows. The first observed difference was
layer-0 V; after changing that projection only, differences appeared in MLP
down and later Q projections. The frozen 1e-5 logit and probability tolerance
failed on the real 65,536-token request.

`backbone_project` gives every CUDA BF16-autocast backbone projection a
1,024-row matrix shape, padding and trimming the last tile. It applies to the
LoRA base and low-rank terms as well as ordinary Linear layers. CPU, CUDA FP32,
and other autocast modes retain the original calls. The pointer head's separate
precision rule remains in place. Fused long-context attention is unchanged;
there is no serial or dense-attention fallback.

## Measured local evidence, 2026-09-28

The source-free fixed-row intervention ran on the pinned calibrated actual-weight
checkpoint SHA-256 `119a7cd06ea93a3e6c264316f12941557f46fead14098324d43c0f5122b5f239`
and the original 65,536-logical-token, 19-question request SHA-256
`1f0922b37a8233a3922a8f6fe74d5cf60e27b7667604c040af24e7576570c6ee`.
It passed the existing parallel/full-row comparison with zero maximum logit and
probability error. Its private receipt SHA-256 is
`26d67660923595c0d8a3ca04178aedc3f748c7f3d20c4349a97c42ec14703f28`.

The edited source, `model.py` SHA-256
`7ac3a0e7844fa1202bb0527a34517c1cd35424b5aa2a18fdc987d6aa68be0f05`,
passed the same real-weight BF16 request with zero error. Batched execution
took 16.60 seconds and 3,471,737,344 peak allocated CUDA bytes; the independent
full-row reference took 94.98 seconds and 3,511,134,720 peak allocated bytes.
The private source receipt SHA-256 is
`959ce48ba5fbe3024ebfe68f767257484bbcd77b5bb4e9b0df63c1e3b5df0490`.
The 70% PyTorch CUDA allocator cap was active on the local RTX 3060. These are
single-run timings, not a latency distribution or total process memory.

The CUDA BF16 projection test checks exact values and finite, matching input
gradients for ordinary and LoRA projections at different row counts. CPU source
tests and the targeted CUDA parallel/attention tests also pass. On this exact
`model.py` source hash, a fresh full-weight BF16
fused-AdamW run completed one step, stopped, then resumed the same run identity
for step two. The final resumable checkpoint SHA-256 is
`c452955a8e2c7afe65ce8da234850834a4574ae0210596f270b89bd8567716cd`;
its parent step-one SHA-256 is
`69bf1b8bc921a33cda0966e57d1e03358dab967c7c4111c851f7b0420b34f9ce`.
The final loss and gradient norm were finite, and the peak PyTorch CUDA
allocation was 8,488,682,496 bytes. Telemetry SHA-256 is
`e4c76fa18a000b21ed86c90c5a0c87f8061842ca20cecfdb6fdcc337eab9d3f4`.
This used two short examples and 288 useful forward tokens.
An independent checkpoint reload verified the SHA-256 and resumable optimizer,
scheduler, sampler and training state, then returned finite short BF16 Choice,
Noul and Score logits. Its private receipt SHA-256 is
`92aa525a0a3a55c1e9b8d9027cb41912f6c366bfca782379ebe72bff55864d99`.

A subsequent complete native BF16 validator on merged main
`39758e61241923800b34b4d54398b38d787042db` passed numerical parity and
structured overflow checks in all six English/Spanish beginning/middle/end
65,536-token cases with zero logit/probability error. The independent short
Transformers oracle and the checkpoint's recorded 32k/64k training exposure
chain also verified. **The overall validator failed:** diagnostic marker
accuracy was 0/18, below its unchanged 80% requirement. Private full report
SHA-256 `0cba4a5181ecd55176a73fe32166952a7411913b12adbfecee4fb8a6c756df`.
Trained 32k/64k backward resource evidence and representative predictive
quality remain separate gates. The historical observed `case=99999` quality
regression is still 1/6 on the pinned calibrated checkpoint.
