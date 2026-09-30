# FP32 canonical numerical repair — September 29, 2026

Refs #3 and #4. This is partial numerical evidence on pinned Qwen weights,
not closure of either acceptance issue. The portable [receipt](short-fp32-canonical-v10-20260929.json)
records checkpoint, source-file, protocol-script and private-result hashes.
[Public reproduction commands](FP32_REPRODUCTION.md) retain the generated
requests, fixed precision settings and numerical tolerances.
The source parent is recorded with a dirty candidate tree; delivery commit and
post-merge verification must be recorded separately by the delivery owner.
The [verified delivery readback](../../RELEASE_READINESS.md) records those revisions and checks.

CUDA FP32 projections use fixed 64-row calls. Attention dispatch follows absolute
query bands: positions 0–255 use fixed 256-row eager arithmetic, positions
256–8191 use SDPA, and later positions use grouped fused SDPA. Canonical CPU
RoPE frequencies are stored as nonpersistent integer bits to survive dtype moves.
Shared K/V and LoRA accumulation use double precision; the short query derivative
retains the rounded eager graph. CUDA cache planning accounts for eight-byte
materialized GQA storage, and its version participates in resume identity for
both FP32 and BF16. Checkpoint schema and telemetry v1 are preserved.

The independent Transformers 4.57.1 eager oracle passes all 96 Q/V LoRA
parameter gradients at 62 tokens; hidden states are exact. Worst gradient
combined-tolerance ratio is 0.706173. Four generated two-question cases cover
short full rows, prefix lengths 255 and 256 crossing the short/medium band, and
prefix length 257. All 100 trainable gradients pass in each batched/serial/full
comparison; tolerances remain 1e-5 for logits/probabilities and 2e-5 for gradients.
These tests use the pinned two-step checkpoint, not a newly trained quality model.

Earlier failed candidates remain preserved in the private execution ledger,
including v9's prefix-255 one-gradient failure (receipt SHA-256
`302f96c18369879e2ae96ae975b6664e9f9351c406f209a6fa39866393d5146e`).
No passing historical long run is promoted to evidence for this candidate.
Fresh current-source 8191/8192 state-prefix boundary cases and the 10520-token
state case pass all 100 gradients in each comparison, with exact logits and
probabilities. Their checkpoint/source/protocol hashes, timings and peaks are
recorded separately in the receipt. These are generated forward/backward probes.
Exact-limit resources, optimizer/resume and serving qualification remain
separate gates. Generated
probes establish no representative quality or Jev benefit; issue #4's diagnostic
marker threshold remains unmet. Overwatch remains archived by user instruction.

The separate FP32 exact-limit training resource probe failed with CUDA OOM during
forward on the RTX 3060 under the unchanged 70% allocator cap. The frozen row
has maximum branch length 32768 and aggregate length 65536. Worker elapsed time
was 148.098410 seconds; peak allocation/reservation was 8,382,291,968 /
9,017,753,600 bytes. Backward did not run. This measured hardware limitation is
not an inference numerical failure or a passing exact-limit training gate.
