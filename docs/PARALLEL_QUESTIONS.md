# Parallel questions — execution contract and verification

## Scope and implementation

Implementation version: `parallel-questions-v1`. Package: `0.1.0+parallel1`.
The default `DecisionEngine.forward` now performs a state pass followed by real
`[B, L]` tensor batches of independent question suffixes. This path is used by
training, calibration/evaluation, and serving. It is not HTTP concurrency,
threaded question evaluation, or batched tokenization around serial model calls.

`execution.py` plans bounded microbatches before any backbone work. `model.py`
executes them and vectorizes the pointer head across the real option positions.
The planner sorts by suffix length by default, never by question ID. It restores
the request's original order using original indices. Mixed Choice/Score/Noul,
per-type temperatures, confidence, model aliases, JSON shapes, serialization and
question-ID round trips retain their existing definitions.

The parent prefix is a request-local tuple of per-layer K/V tensors. Expansion
uses a read-only batch view followed by functional concatenation, not mutation,
`detach`, a global cache, or a copied model. Autograd sums branch contributions
back into the original prefix. Full-weight, LoRA (including nonzero adapter
matrices), activation checkpointing, accumulation and resumed optimization are
covered by CPU tests. No training runtime is imported into Overwatch.

## Padding and attention proof

Let `S` be the state length, `Li` a branch's real length, and `L` the longest
suffix in its microbatch. Each row contains its own real suffix followed by right
padding up to `L`. All real rows start RoPE at offset `S`. Cached attention uses
lower-right causal alignment on queries of length `L` and keys of length `S+L`.
Real query `i < Li` can see exactly the complete state and its own suffix keys
`j <= i`. Padding keys occur strictly after `Li-1` and cannot affect a real query.
Different questions occupy different tensor batch rows, so no cross-question
attention exists. Padding queries have valid causal keys (including self); they
are not all-masked and their outputs are discarded. The pointer readout gathers
`Li-1` and the actual option-end indices, never `L-1` for shorter rows.

Padding therefore needs no dense per-row key-padding mask for this strictly
causal, right-padded layout. A test replaces padding with two different token
values and verifies identical real hidden states, finite padding outputs, and
zero gradient into a padding-only embedding. This argument does not apply to
left padding, bidirectional attention, or recurrent/hybrid models; those are not
silently enabled. Grouped-query attention retains each layer's K/V head count and
repeats K/V only for its query heads.

There is no dense attention mask over the sum of all request branches. The
existing fused-kernel-only guard for long CUDA rows is retained. Native CUDA and
pretrained-Qwen parity remain unverified; CPU fixture tests are not substitutes.

## Bounded execution policy

```json
{
  "version": "parallel-questions-v1",
  "max_branches": 16,
  "max_padded_tokens": 65536,
  "max_cache_bytes": 536870912,
  "sort_by_length": true
}
```

`max_branches` bounds rows per backbone suffix pass. `max_padded_tokens` bounds
`B * (S + L)`, including the prefix footprint for each row, not merely the sum
of unpadded suffix lengths. Cache admission estimates:

```
parent KV bytes = S * kv_rate
batch cache bytes = parent KV bytes + B * (S + L) * (kv_rate + gqa_rate)
kv_rate  = 2 * layers * KV_heads * head_dimension * element_size
gqa_rate = 2 * layers * query_heads * head_dimension * element_size
```

The GQA term conservatively includes materialized expanded keys/values across
all layers. Element size uses parameter storage and is conservative for the
supported FP32-parameter autocast profile. It is an admission estimate, not a
measurement or a total-RAM promise: weights, attention workspace, activations,
allocator reserves and optimizer state are additional. Training autograd can
retain activations/caches from several microbatches until backward; use activation
checkpointing and deployment-specific measurement. Per-microbatch budgets do not
claim to bound the entire training graph's peak memory.

Every singleton must fit before the state pass starts. Otherwise a
`BatchBudgetExceeded` identifies `padded_tokens` or `cache_bytes`, actual value,
limit and question ID. HTTP maps this to 422. There is no truncation, implicit
budget enlargement, or hidden serial fallback. Singleton batches are legitimate
when only one question exists or the explicit branch/token/cache policy requires
one. Strict context limits from `encoding.py` and the backbone remain enforced.

For training, put the policy under `execution` in the normal training JSON.
Serving uses the checkpoint policy unless explicitly overridden:

```sh
python -m kev_laya serve --checkpoint trained.pt --no-auth \
  --batch-policy configs/parallel-default.json
```

## Reference paths and diagnostics

`model(encoding, reference=True)` independently recomputes each complete
state+suffix row. `model(encoding, serial_reference=True)` preserves the old
single-prefix, serial-suffix computational schedule for benchmarking. Neither
path is selected as an error fallback or exposed as the normal serving mode.
Both use the same weights and pointer formulation. The serial comparison now
includes the common admission planner/diagnostics, so its timing is a scheduling
comparison, not an exact replay of every old orchestration instruction.

Every call returns request-local diagnostics, not mutable module attributes:
`prefix_passes`, `branch_passes`, `effective_batch_sizes`, original branch indices,
padded lengths, real prefix bytes, estimated cache bytes, and token counters.
The HTTP API adds `X-Kev-Laya-Execution`, `X-Kev-Laya-Prefix-Passes`,
`X-Kev-Laya-Branch-Passes`, `X-Kev-Laya-Batch-Sizes`,
`X-Kev-Laya-Compute-Tokens`, and `X-Kev-Laya-Padding-Tokens` headers without
changing its JSON response schema. `GET /v1/models` exposes `question_batching`.
These counters describe scheduled forward passes; activation-checkpoint backward
recomputation and optimizer operations are not included.

* Logical input tokens: `S + sum(Li)`; the state is counted once.
* Existing `forward_tokens`: useful, unpadded model tokens; unchanged for the
  shared-prefix path. The independent full-row oracle counts its repeated state.
* New `compute_tokens`: `S + sum(B * L)` for the shared-prefix path.
* New `padding_tokens`: `compute_tokens - forward_tokens`.

Output-token accounting still tokenizes the actual serialized answers and does
not claim generated tokens. Tiny numerical differences can change serialized
floating-point text length; accounting reflects that text rather than inventing
identical output usage across different rounded values.

## Checkpoints and resume

No weight keys, tensor shapes, tokenizer serialization, or checkpoint format tag
changed. New checkpoints carry an optional top-level `execution` dictionary.
Loading an old checkpoint defaults inference to the new batch policy and retains
its digest-based model identity and calibration. The unchanged reviewed trained
checkpoint is exercised in the benchmark.

For new training runs, execution policy is persisted in the checkpoint, training
configuration and config hash. Changing branch/token/cache limits, length sorting
or execution version during resume is rejected. Legacy serial checkpoints can
initialize a new batched run with `--init`; exact optimizer resume from a legacy
checkpoint lacking an execution record is explicitly rejected. This avoids
silently claiming the old and new numerical training trajectories are identical.
Within an unchanged CPU execution configuration, interrupted continuation is
bit-for-bit equal, including LoRA, activation checkpointing, reward training,
choice permutation, and gradient accumulation. CUDA reproducibility is not claimed.

## Telemetry and Overwatch

The telemetry v1 contract, validators, units, adapter, collector/cache/report
integration and frontend helper files are byte-for-byte unchanged. Training's
`microbatches` still counts accumulated training examples, not question batches.
Execution counters live in checkpoint state and in checksummed
`execution-<step>.json` artifacts using the existing `evaluation` artifact kind.
A failed diagnostic artifact write increments monitoring failures without
invalidating the optimizer checkpoint or crashing training. Existing Overwatch
artifact links can expose the diagnostics; no new telemetry schema or live
Overwatch modification was necessary for this task.

The original support files are also retained under `integrations/overwatch` so
the model checkout's tests and miniature script do not require an external sibling
directory. This is a relocation/copy of delivery support, not a second dashboard
or a modification of the user's Overwatch primary checkout.

## Verification and reproduction

The original code's baseline suite passed 120 tests with one SDK skip. An
instrumented 16-question request made 17 backbone calls, all batch size one.
The new full suite passed 185 tests with three skips: the same missing SDK and
two unavailable CUDA checks. Actual tensor batching is asserted with a seven-row
suffix pass after one prefix pass. Tests include unequal lengths, padding-value
independence, question permutation/addition/renaming, cross-microbatch behavior,
concurrent requests, cache identity and immutability, explicit overflow, 255
Choice options, 10 Score levels, and all relevant training modes.

Frozen FP32 tolerances are `atol=rtol=1e-5` for logits/probabilities and
`atol=rtol=2e-5` for gradients, using the normal absolute-plus-relative test.
Across 24 fixture configurations the maximum differences were 1.78814e-7 in
logits, 5.96046e-8 in probabilities, and 2.10421e-8 in parameter gradients.
The old trained checkpoint's benchmark reached 8.58307e-6 versus the independent
full-row oracle and 1.52588e-5 versus the serial-cached path. The latter is above
1e-5 in absolute terms but well inside the unchanged combined tolerance (worst
ratio about 0.128). A separate audit decomposes backbone batching and head
reduction rounding; the targets were not loosened. The old zero-tolerance
composition assertion was replaced with the task's explicit 1e-5 tolerance;
identical-layout ID renaming and same-configuration resume remain exact tests.

```sh
python -m pytest -q
python -m compileall -q src tests scripts
PYTHONPATH=src python scripts/verify_parallel_numerics.py --out new-numerics.json
PYTHONPATH=src python scripts/benchmark_parallel.py \
  --checkpoint ../evidence/benchmark-input/calibrated.pt --out new-benchmark.json
PYTHONPATH=src python scripts/smoke_experiment.py --out new-smoke \
  --steps 40 --extension-steps 8 --choice-permutation
PYTHONPATH=src python scripts/verify_parallel_http.py \
  --checkpoint new-smoke/added-pgps/calibrated.pt --out new-http
```

PowerShell: set `$env:PYTHONPATH = "src"` first, then run the `python` commands
without the leading `PYTHONPATH=src`. Use new output locations; measurement scripts
refuse to overwrite existing evidence. Benchmark subsets are supported by
`--counts 1 4` and `--distributions uniform` for bounded runs.

The benchmark uses identical checkpoint bytes and token IDs across three modes,
1/4/16/32 questions, equal and unequal lengths, three warmups, ten measurements,
two CPU threads, and a fresh process for every mode/case. Tokenization is outside
timing; admission, collation, prefix, suffix batches and head readout are inside.
RSS is sampled every 2 ms after warmup; short allocation peaks may be missed.
The script synchronizes CUDA timings and records GPU allocator peaks when run
on supported hardware, but no CUDA execution occurred here.

The delivered miniature experiment trained 40 supervised steps and two 8-step
continuations, calibrated on its held-out calibration split, and produced
snapshots/checkpoints/evaluations that the unchanged adapter ingested. Its
recorded process wall time was 5.37 seconds and OS peak RSS 336,260 KiB. The
installed-wheel live server passed 12 requests, nine with multiple questions,
and produced valid content-free telemetry. These are engineering fixtures, not
new predictive-quality evidence. General quality, 32k/64k context, pretrained
native parity, live Windows/Overwatch integration, and paid cloud checks remain
outside this task's verified results.


## Stage-3 compatible diagnostics

Parallel kernels and tolerances are unchanged. Training/serving now also expose counters in telemetry v2 `extensions.execution`: `prefix_passes`, `branch_passes`, `questions`, `max_batch_size`, `batch_size_histogram`, `useful_forward_tokens`, `padding_tokens`, `compute_tokens`. Historical v1 remains readable. Additional preprocessing/exposure evidence makes legacy optimizer resume explicit rather than silently migrating it; see `STAGE3_CORRECTNESS.md`.
