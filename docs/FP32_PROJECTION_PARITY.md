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

## Current-source short mixed backward diagnostic, 2026-09-29

On clean merged source `66b17b354c4ce53fbb143bfb1f7326474640269e`,
the exact long-trained checkpoint above was reloaded for one generated
201-logical-token Choice/Score/Noul request. CUDA FP32 batched, serial cached
and independent full-row paths passed unchanged 1e-5 logit/probability and
2e-5 gradient tolerances across all 100 trainable gradients. Batched/full-row
maximum logit error was `4.291534423828125e-06`; worst gradient tolerance
ratio was `0.57034`. Private FP32 receipt SHA-256 is
`e19f27d644907fb59b365b2317ee3827cc7838c615f8d946f024f751f4e28af6`.

The otherwise identical BF16 short trial **failed**: batched/full-row maximum
logit error was `0.6494655609130859`, and the worst gradient tolerance ratio
was `12165.25`. Private BF16 receipt SHA-256 is
`2af3abd91eafa37b0e8e086eac617562aef742403028362386a30d94384a72ed`.
The short forward difference persisted in inference mode and with the earlier
calibrated checkpoint. A layer probe first observed a paired-versus-serial
backbone difference at layer 7; private receipt SHA-256
`a0af21e541bd2b285e1c000e076c4905087a433576cc5118b0b9a420d878481c`.
A source-free double-accumulation short-attention intervention made forward
logits equal but left gradients failing (worst ratio `2574.53`); private receipt
SHA-256 `59580b4b8f487eb676ba326921392cc8c5159794680e1f785609d68b978c62d3`.
No production arithmetic or tolerance was changed. These are single short
diagnostics; the earlier full-context BF16 numerical pass covers a different
shape and does not prove BF16 backward parity.

## Generated marker curriculum on current main, 2026-09-29

On clean main `dfaed82c69c4848f579297e020c25e9eab5fc39b`, a frozen
generated short curriculum with group-disjoint train, development, calibration
and test partitions ran 120 resumable BF16 LoRA optimizer steps and 166,610
useful forward tokens. Its terminal checkpoint SHA-256 was
`8c44d74dc0b3dd05ea2c004033999b23965996798df5123e605511f2b6619890`;
the frozen suite manifest SHA-256 was
`40e9d6b5bd43c962c26f3ef1544ea332f6048f6a7a84d7f2c2a0143d44b45963`.
The separate long curriculum then completed two parent-linked BF16 LoRA and
activation-checkpointed optimizer steps on two distinct generated examples at
exactly 32,768 maximum branch and 65,536 aggregate tokens. Its final resumable
checkpoint SHA-256 was
`f939a66c1f722f256665441d79c016c3acd1b7d35ce968443ef039c6b45f8035`;
the long suite manifest SHA-256 was
`984ee9c10ecd39df94bb0b8e261ec8ed311b6ecc117148fea15f10d6989a184b`.
The earlier 20-question long design exhausted the frozen 70% RTX 3060
PyTorch allocation cap before step one. A separately frozen 19-question
resource revision completed the two steps, totaling 131,072 useful forward
tokens with finite loss and gradient and no monitoring export failure.

A separate generated calibration partition produced checkpoint SHA-256
`7a53dffc893bb12aa4db4ffbabd82e01f5cc15aa70bbafad40da58aabfe80dd0`.
The exposure verifier passed its parent chain and exact-limit observations.
The unchanged BF16 native validator then completed all six English/Spanish
beginning/middle/end cases on this checkpoint. Every case had 32,768 maximum
branch and 65,536 aggregate tokens, passed the unchanged numerical comparison
with zero maximum logit and probability difference, and passed structured
over-limit rejection. The short pinned Transformers oracle also passed.
Optimized latency ranged from 14.32 to 14.75 seconds; the independent full-row
reference took 82.94 to 86.48 seconds. The private report SHA-256 is
`0fba22628bd1c114d5ea39b19fbd7f50f273782a9d9ad4e0e7306aa044a9f73b`.

**The overall BF16 report failed:** only 2 of 18 diagnostic marker decisions
were correct (11.1% against the frozen 80% minimum). The generated development
split scored 50/72 (69.4%); its receipt SHA-256 is
`a9c6e0953e5bf633cce9e24ffeee9c68678c8d3bd514934126109dd8749d29be`.
The fixed validator had been observed during development, so it is not an
untouched quality test. These generated results do not establish reviewed
representative quality or Jev parity. FP32 full-context validation on this
new checkpoint is unmeasured, and the separate short real-weight BF16 gradient
parity failure remains. Issue #4 stays open.

## Further generated exact-limit diagnostics on current main

On clean main `c7c6d26d80534035c67618e1f245af5f2fdbab9a`, a new
24-case generated long curriculum varied language, evidence position and
route/urgency/escalation labels, excluding the fixed validator's exact
`(billing, 7, true)` combination from training. Its manifest SHA-256 was
`7c515c4678e8b372e7c499a0a8249ebcc66b24066101ce1682c52eefd90370db`.
The first v4 attempt exhausted the unchanged 70% RTX 3060 PyTorch allocator
cap during its second backward pass before a checkpoint; failed telemetry
SHA-256 was `dc4ff6555012d731bb91e266239650088c94f4427812038a7ad616990b262f86`.
The separately frozen v5 run kept the same suite, config, parent and cap and
cleared unused CUDA cache after gradient cleanup. It completed 24 exact
32,768-branch/65,536-aggregate optimization examples and 1,572,864 useful
forward tokens. Its resumable checkpoint SHA-256 was
`799959df7fe48a54b82c5e7fd87fe706760645517327001a8baeb7050a2eaa36`;
all saved parent hashes matched their manifests. Peak PyTorch allocation was
8,601,058,304 bytes, and monitoring export failures were zero. This does not
prove that cache cleanup caused the v4/v5 feasibility difference.

The v5 separately calibrated artifact SHA-256 was
`c1b415beb5c5d498103fe1d7595fd4eb2b9ed60768235c5c665f91a9003ea60d`.
Exposure verification passed its calibration-to-training chain. The unchanged
six-stratum BF16 native validator passed numerical and overflow checks in all
six cases with zero maximum logit/probability error, but **failed overall**:
2/18 marker decisions were correct against the unchanged 80% diagnostic
minimum. Report SHA-256 was
`83e53328d1d123b12f8a7fbae1013fff3a304a15a661bf7113ee56f9118649ca`.
On its generated development split, accuracy was 48/72 (66.7%); receipt
SHA-256 was `849c93a95bf4c577d80d19b55d7b97522465fb96786a8eccf343443399a5bddd`.

A further, separately frozen generated v2 short curriculum included 160
group-distinct train cases with 255 route options each, varied distractor
descriptions and option order, plus separate development/calibration/test
splits. Its suite manifest SHA-256 was
`2a10bfdd8368e61da0cdeadf180e0ec6b3e127a1afc1ade9acccb0daf4825c2f`.
From the v5 terminal parent it completed 160 short steps and 961,078 useful
forward tokens; intermediate checkpoint SHA-256 was
`12b172f41124a71438a574e3bd529b8e41bbad77d1b464682182267a45901413`.
That intermediate run observed only 6,087 maximum branch and 6,269 aggregate
tokens. A separately frozen v6 continuation then optimized two distinct exact
32,768-branch/65,536-aggregate examples. The final training checkpoint
SHA-256 was `8472f22b8f02b14f27ccded353c2f378d71579a927324a25d9e5c818d4aaad90`;
its parent manifests and native exposure verified. Its calibrated artifact
SHA-256 was `a731903c8e91adc6206ecf60325b6ebbf4cdedd1349f0ab831c85069cf37df3d`.

On that v6 artifact, the unchanged BF16 native validator again passed 6/6
numerical comparisons and 6/6 structured overflow checks with zero maximum
logit/probability error, but **failed overall** at 6/18 marker decisions
(33.3% against 80%). Route became correct in both languages at beginning and
middle, while both end routes failed; urgency was correct only at the end,
and escalation predicted false in all six cases. Report SHA-256 was
`9b3f05edb105c2a09f80d0acf6cb28178ae64dd3329ba93415b505fcc240bc60`.
The separate v2 generated development split scored 52/72 (72.2%), with
Choice 11/24, Score 24/24 and Noul 17/24; receipt SHA-256 was
`fb69e4f47a7207505aa25994af62930279e417b56898ceddcfb7e64bdb204a19`.
Its split differs from v5's, so the development percentages are not paired.

The fixed validator had been observed before both new curricula were designed.
These are diagnostics, not an untouched quality test or representative decision
evidence. The 80% marker gate, BF16 short backward parity, broader native
precision/shape matrix, representative quality and Jev comparison remain open.
