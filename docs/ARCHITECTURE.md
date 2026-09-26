# Architecture decision: a single functional-prefix pointer model

**Decision date:** 2026-09-25. **Status:** implemented on CPU fixtures; pretrained-kernel feasibility pending.

## Reused, adapted, and replaced

Kev at `3d9973b80b34d187f9fc8ce81de940b5767eb624` supplies the initial architecture: a causal state prefix,
independent question branches and a learned pointer readout over option boundary states. The new
`PointerHead` adapts its query/key projections and scaled dot product. Its five delimiter conventions
are retained for the Qwen tokenizer. The current upstream hybrid recurrent kernels, training loop,
server, truncation behavior and released weights are **not** silently inherited.

Laya at `4066d5d5fbf08b66c6757ddeedbd797bd7655bc0` supplies proper-scoring reward, typed handling,
post-hoc calibration and inference-hook concepts. These are adapted into this same pointer model.
Its bidirectional encoder, independent per-question server, truncating preprocessing and unvalidated
auxiliary action head are not copied. The new method is named PGPS-v1, not proprietary RLCD.
The parent revisions remain explicit baseline definitions; no parent baseline evaluation ran here.

The new Qwen2-compatible backbone is independently implemented from inspected architecture sources.
It is not a wrapper around an upstream inference server. It accepts the pinned 0.5B weight layout;
`native_validation.hf_parity` compares its hidden states to the pinned Transformers oracle after
merging adapters. Successful strict key loading alone is not numerical equivalence or quality proof.

## Why all-attention rather than a hybrid

A purely attention-based backbone makes all carried state explicit as each layer's keys and values.
The selected Qwen configuration has no recurrent/DeltaNet/convolution state to accidentally share.
Hybrid, sliding-window and unsupported RoPE configurations are rejected rather than assuming an
attention mask controls them. The native model card/config and Qwen2 implementation were inspected,
but actual native memory/latency feasibility was not measured. Selection therefore remains a
candidate architecture, not proof of its advertised context performance.

## Execution

A request serializes `[state delimiter, state tokens]` once. Each branch serializes its type,
instructions, option labels, rubrics, explicit option-end markers and a decision marker. IDs and the
requested model alias never enter model tokens. Branch positions start at the same state length.

The state pass yields a tuple of per-layer KV tensors. Right-padded question rows run in real tensor
microbatches, each expanding the immutable parent prefix by a view and concatenating its own KV
functionally. Branch, padded-token and cache-byte budgets bound each forward pass. See
`PARALLEL_QUESTIONS.md` for the exact memory estimate and training-autograd caveat. The attention kernel uses a **lower-right causal bias** for cached suffix queries;
an upper-left triangle would be wrong. There is never an attention matrix for an aggregate request.
For long CUDA inputs, the code requires a fused flash/efficient SDPA backend and refuses silent dense
math fallback. Native support for the selected hardware/kernel is a measured gate, not assumed.

The training graph retains state-prefix gradients. Multiple branches contribute to one shared-prefix
backward graph. Full-weight and LoRA paths both exercise this behavior in tests, including activation
checkpointing. The reference execution recomputes complete state+branch rows independently; CPU logits
and all trainable parameter gradients are compared against it. Numerical equivalence is not evidence
of task correctness. Question order/addition invariance within declared FP32 tolerances and exact ID invariance are checked separately from measured
option-order sensitivity: option order is **not** structurally invariant in this architecture.

There is no cross-request cache. Concurrency is bounded at the API before parsing/model execution.
Request contents are not logged, saved, embedded for retrieval or included in telemetry. Tokenizer
configuration, RoPE settings, dimensions, LoRA configuration and calibration state are checkpointed.
No pretrained or native claim is inferred from a larger maximum-position constant.

## Monitoring separation

The model emits validated, bounded, atomic snapshots and checksummed artifact references. Overwatch's
new provider reads explicitly registered sources. Only its collector writes `global/kev_laya/runs-v1.json`.
A cache transaction lock prevents overlapping collectors from overwriting a newer sequence with an
older read. `cached_metrics`, `report`, and `report_service` read the cache, and the existing React
resources page gains a model-run section. This does not fabricate cloud resources for local runs.
SkyPilot's resource inventory and existing Flow records retain their own types and behavior.

## Source inspection evidence

Overwatch: `CompleteDotTech/Overwatch@501bd99a0cb0c6b321feb022f5747d6dcbd5e9c4`.
The required provider, collector, raw cache, cached metrics, report service, report, constants, frontend
types/App and AGENTS files were inspected through GitHub. Full local checkouts were not available.
The Windows primary checkout is untouched. The supplied installer, not an imagined live edit, is the
integration deliverable. The research alignment/report/45-column feature matrix files were not found
in the uploaded files, accessible Library or mounted runtime; their contents are not reconstructed.

Primary source URLs and inspected revisions are recorded in `configs/sources.json` and the frozen
compatibility JSON. License texts in `licenses/` match the observed upstream Git blob hashes.
