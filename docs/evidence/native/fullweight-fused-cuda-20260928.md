# Bounded full-weight CUDA training diagnostic

This is partial issue #3 evidence from the actual pinned `Qwen/Qwen2.5-0.5B`
weights (SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`)
on an RTX 3060, PyTorch 2.10.0+cu128. The training suite is generated smoke
data, not representative quality data. The source tree contained the changes
in this PR when the diagnostic ran; the post-merge replay must verify the
merged source identity.

The default AdamW backend exhausted a 70% per-process CUDA memory cap at its
first optimizer step. A prior fused optimizer attempt exposed a second issue:
resume constructed a duplicate GPU model. After staging that checkpoint model
on CPU and removing a redundant full-gradient finite mask, a fresh run using
`training.optimizer_backend: "fused"` completed a full-weight FP32 optimizer
step and resumed the same run for step 2 under the same 70% cap. This backend
is opt-in, requires CUDA, and is bound into the resume configuration hash;
omitted/default settings retain their previous config hashes.

The private config SHA-256 is
`f52eb9f1bdf1a1552b5edec727f21b975b83eb9b3a064aa7d848f413d2d0146c`.
The step-1 and step-2 checkpoint SHA-256 values are respectively
`152f33cc08cd042ba536ef74404e531effdac77965535f2ce8839615861e5f25`
and `9d895ae3451f8924d73237539c5ecf3c4cbe26ba3203220fb5fddcb7df52e114`.
Both attempts have the same run ID and config hash
`2c57561e419cd74c079a0db6efff7ec2d80a68b38feb6b28f9dde5a92a357587`;
the second records parent attempt `31506507ea7d4fe3aa271f956c8ab872`.
The final telemetry SHA-256 is
`e18bd31d3f54d647f9d531236ff6a3e535e10cefc366f63b99cdfddc0bbba5ba`.
It records two optimizer steps, two examples, 288 forward tokens, maximum
branch/aggregate lengths 69/144, peak GPU allocation 8,485,471,232 bytes,
and no monitoring export failures.

This diagnostic establishes bounded short-sequence full-weight train/resume
mechanics. Strict FP32/BF16 reference numerics, long-context training,
representative quality, and remote publication remain separate acceptance
criteria. The earlier failed attempts and their receipts are preserved.
