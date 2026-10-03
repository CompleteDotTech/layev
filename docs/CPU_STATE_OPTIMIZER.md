# Explicit CPU optimizer state

## Trained rank0 derivative readout (October 2, 2026)

[Unfavorable derivative evidence](evidence/native/rank0-derivatives-failed-20261002.md) records one passing comparison out of five. Both HF comparisons and native shared-prefix/full gradient comparisons fail frozen tolerances. This independent derivative failure establishes no CPU-state optimizer arithmetic result; native acceptance remains incomplete.

`cpu_state_streaming_v1` is an experimental, explicitly selected CUDA AdamW
backend. It requires PyTorch `2.10.0+cu128` and trainable CUDA FP32 parameters,
including FP32 parameters used under BF16 autocast. Default and fused optimizer
selection and their recorded configuration hashes are preserved.

Select it in the training configuration with an explicit persistent state budget:

```json
{
  "training": {
    "optimizer_backend": "cpu_state_streaming_v1",
    "optimizer_state_max_bytes": 4000000000
  }
}
```

The budget bounds two CPU moment tensors and one FP32 CPU scalar step per
trainable parameter. It does not bound total host RAM, checkpoint deserialization,
model parameters or gradients. The backend rejects an insufficient budget before
construction. Each update stages one parameter's moments on CUDA and calls the
pinned stock single-tensor AdamW functional. Global gradient clipping remains in
the trainer. There is no automatic fallback or allocator-cap increase.

Algorithm version and ordered trainable parameter names, shapes and dtypes enter
the opt-in configuration identity. Resume across default, fused and CPU-state
backends is refused. State loading validates the complete payload before changing
the optimizer, keeps moments on CPU, and rejects malformed or aliased state.
Live parameter replacement or reordering is also refused. The trainer releases
temporary checkpoint/model objects after restoring optimizer, scheduler, sampler
and RNG state; the CLI independently releases its unused checkpoint payload.

## Qualification scope

The reviewed external primitive candidate completed eight CUDA unit tests with
no skips under an owned 120-second watchdog and a 70% allocator cap. This covers
stock AdamW primitive equivalence, CPU state residency, malformed-state refusal
and optimizer restore. Its unchanged optimizer math is distinct from the trainer
integration: the integration additionally fixes canonical parameter-schema
serialization to use JSON lists. CPU configuration tests preserve default/fused
hashes and exercise cross-backend resume rejection using simulated device metadata.

The integrated source at `b5d8299b768564c09c541e3dfc04b3d28eef2fd6` subsequently
completed another owned primitive run: eight tests passed, zero skipped. The
[portable tiny qualification receipt](evidence/optimizer/tiny-cuda-offload-v4-20260930.json)
records all eight FP32/BF16 production-trainer process phases passing with no
skips: uninterrupted two steps, durable step1 exit, fresh-process resume to
step2, and reload/compare. All 33 production Python hashes and source HEAD
matched at each process start/end. Every model and optimizer tensor element,
scheduler, sampler, RNG and the whole training exposure matched exactly; resume
attempt lineage and each checkpoint's own parent chain were checked separately.

This is a four-row synthetic byte-tokenizer fixture with 30 parameters and
28128 model tensor elements, not a pretrained fullweight result. Live persistent
CPU optimizer state was 225144 bytes; comparison covered 56286 optimizer tensor
elements. A zero tensor-element count for scalar-only fields means those fields
were asserted equal, not omitted. Effective matmul precision was highest,
TF32 was off, deterministic algorithms were enabled and the allocator cap
was 70%. The earlier external candidate primitive receipt remains distinct.

A separate tiny **LoRA rank4 with activation checkpointing** run completed all
8/8 FP32/BF16 trainer phases with zero skips on the same frozen source. The
[portable LoRA receipt](evidence/optimizer/tiny-cuda-offload-lora-checkpoint-v2-20260930.json)
records exact model/optimizer/RNG/exposure and resume state comparisons,
12 trainable parameters, 26 unchanged frozen states, and observed LoRA/head
updates. Live CPU optimizer state was 15664 bytes. The integrated v4 primitive
proof was reused; this was not another primitive test run. These are synthetic
fixture results, with no native fullweight or acceptance claim.

Actual pretrained full-weight step/save/exit/resume/reload and the
captured-gradient stock optimizer oracle remain pending. Their fixed 16 GiB
available-host-RAM admission requirement is currently unmet. Existing default
backend OOM receipts remain failures. Primitive or CPU fixture checks establish
no independent full-weight derivative parity, trained long context,
representative quality, Jev advantage or issue closure.
