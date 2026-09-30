# Explicit CPU optimizer state

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

Production trainer uninterrupted versus fresh-process resume checks in FP32 and
BF16 are pending. Actual pretrained full-weight step/save/exit/resume/reload and
the captured-gradient stock optimizer oracle are also pending. Their fixed 16 GiB
available-host-RAM admission requirement is currently unmet. Existing default
backend OOM receipts remain failures. Primitive or CPU fixture checks establish
no independent full-weight derivative parity, trained long context,
representative quality, Jev advantage or issue closure.
