# CUDA FP32 backbone projection parity

The pinned real-weight 65,536-token request previously failed the frozen
parallel versus independent full-row FP32 comparison. CUDA GEMM output changed
with the number of projection rows. Applying 1,024-row tiles fixed that long
comparison in a source-free diagnostic, but broke the independent short
Transformers oracle: maximum hidden error rose to `0.00299072265625`.

The short oracle sweep found that 64-row tiles preserved its pinned 1e-4 hidden
tolerance, while 128, 256, 512 and 1,024 rows failed. The production backbone
now uses 64-row tiles for CUDA FP32, including short requests, so a question's
projection arithmetic does not depend on the length of another question in the
same request. CUDA BF16 retains its separate 1,024-row rule. CPU arithmetic,
attention kernels, model state keys and public request/response shapes remain
unchanged. Padding is discarded after each projection tile; the same rule
applies to LoRA base and low-rank terms.

## Measured local evidence, 2026-09-28

The pinned trained/calibrated checkpoint SHA-256 was
`119a7cd06ea93a3e6c264316f12941557f46fead14098324d43c0f5122b5f239`.
The original 65,536-logical-token/19-question request SHA-256 was
`1f0922b37a8233a3922a8f6fe74d5cf60e27b7667604c040af24e7576570c6ee`.
On edited `model.py` SHA-256
`fb9e21d067f199a20d06b1a0a0a6e2222b1a577c0f6858e0c33b04942d39e024`,
the FP32 parallel/full-row comparison **passed** its unchanged absolute and
relative 1e-5 tolerances: maximum logit error `1.1444091796875e-05`, maximum
probability error `1.6229137018985984e-07`. The private receipt SHA-256 is
`d3660adf0b0c96bc0550e79d1b4383cbcda1458f6511c1eda650816c66812f2f`.
The synchronized batched pass took 87.52 seconds; the independent reference
took 525.22 seconds. These are single-run timings and show a real reference
cost, not a production latency distribution. A 70% PyTorch CUDA allocator cap
was active on the local RTX 3060.

The source also passed the independent pinned Transformers short oracle at 62
tokens, maximum hidden absolute error `0.000152587890625`; private receipt
SHA-256 `29737485bad8760a23133ae9f63e86bd0867dd50aa1841ce40cc3692abbbfe6b`.
CUDA tests check exact value and input-gradient equality across row counts for
ordinary and LoRA projections. The eager attention and right-aligned mask test
continues to assert exact equality using the same projection arithmetic.
The unchanged BF16 path was rerun on the same 65,536-token/19-question request
after the FP32 source edit and passed with zero logit/probability error; private
receipt SHA-256 `b073266bd0efaa9f3f5c7e9df3f50c1e9c8e805586aed60761ff98a5ebeacaad`.

On this exact source hash, a fresh full-weight CUDA FP32 fused-AdamW run stopped
at step one and resumed the same run identity through step two. Its final
resumable checkpoint SHA-256 is
`23ca1ac8f550b896313bc4b20d52740a009e4e36aa3fe9d9f896557816967b15`,
parent step-one SHA-256
`3c148289d38fb24566f7c55e902973c194ef22fa58b230282aab1d3861c68b78`.
Final loss and gradient norm were finite; peak PyTorch CUDA allocation was
8,473,674,752 bytes under the 70% cap. Telemetry SHA-256 is
`c842405b51efc7c5e1bb4d9dbd5d4a2701cd701fb02e6c94a4c6f27d34d2d52c`.
An independent checkpoint reload verified resumable state and produced finite
short FP32 Choice/Noul/Score logits; private receipt SHA-256
`66f71155a617dd5f3a63526a0a5c4ad3775bd24a159b3e893578e8247e7e37c1`.
These were two short examples and 288 useful forward tokens, not long-context
training exposure or predictive-quality evidence.

## Complete FP32 native readout on merged source

On merged source `7e207d0c488466615a295385efea439a9c228d60`, the same
pinned actual-weight checkpoint completed all six English/Spanish
beginning/middle/end cases at exactly 65,536 logical tokens each. All six passed
the unchanged combined absolute and relative 1e-5 numerical comparison and
both structured overflow checks. Maximum logit and probability errors across
the matrix were `1.9073486328125e-05` and `9.822982885293596e-07`;
optimized latency ranged from 77.84 to 87.37 seconds and independent full-row
reference latency from 482.64 to 522.24 seconds under the 70% RTX 3060
PyTorch allocation cap. The short pinned Transformers oracle passed, and the
existing training-exposure chain verified 32,768 branch and 65,536 aggregate
observation.

**The overall report failed**: only 1 of 18 diagnostic marker decisions was
correct, or 5.6%, below the frozen 80% threshold. The private composite report
SHA-256 is
`50cbd14f93f7e8e92343d7cac6fce079d89f9ef783c89db5814c182fc263b919`.
The original process stopped after five durable cases. A one-case continuation
evaluated only the missing Spanish end case; an assembly retry hash-verified all
six request/result pairs and recomputed the short oracle and exposure chain.
The recovery is explicit in the report. A temporally adjacent NVIDIA driver
event 153 does not establish why the original process stopped. The first
assembly attempt had a private import error after writing the sixth result;
it did not require another long inference.

The separate complete BF16 validator on the previous main source passed 6/6
numerical and overflow cases but **failed overall** with 0/18 diagnostic marker
decisions correct; private report SHA-256
`0cba4a5181ecd55176a73fe32166952a7411913b12adbfecee4fb8a6c756df`.
On current merged source `f2589b9978e24cdcd4170a152d713370e8b08870`,
a separate generated BF16 LoRA and activation-checkpointed CUDA run completed
two optimizer steps with a durable stop/resume lineage. Both examples reached
exactly 32,768 maximum branch and 65,536 aggregate tokens, for 131,072 useful
forward tokens. The final resumable checkpoint SHA-256 is
`99f797ffd0521adbee964d130d73eb381c9b6bd3b7896ce61b5007886d19ccf9`,
parent step-one checkpoint SHA-256
`98fef20b90429707e5ce58237e8fd6b7b0927c8c4d222b303cec6844feab4822`,
and telemetry SHA-256
`98f84b03b0a9d53711d09d5039d49f80e921b8db04cb759e1c2d36fbcbc87f9e`.
Exposure verification confirmed both exact-limit examples and both optimizer
steps. Final loss and gradient norm were finite; peak PyTorch CUDA allocation
was 8,647,482,368 bytes under the 70% RTX 3060 cap. Monitoring export
failures were zero.

A separate SHA-verified CUDA reload of that checkpoint produced finite short
Choice, Score and Noul outputs with shapes `[2]`, `[10]`, `[2]` and 237 logical
tokens. Its private receipt SHA-256 is
`429fc084b9546190b660278662b68c899a88a9610110b956db002daf9d0bb0f9`.
The checkpoint reports `unfitted-after-weight-training`. This run establishes
current-source generated long-context backward and checkpoint mechanics for
BF16 LoRA with activation checkpointing. It does not establish long full-weight
backward, calibrated long-run serving, representative quality, or Jev-relative
evidence. The frozen diagnostic marker failure and observed `case=99999`
quality regression remain open; neither numerical matrix passed overall.
