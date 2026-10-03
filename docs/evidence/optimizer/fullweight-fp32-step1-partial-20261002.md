# Actual full-weight FP32 optimizer step: partial evidence

The [portable receipt](fullweight-fp32-step1-partial-20261002.json) records one actual pretrained rank0 FP32 update using `cpu_state_streaming_v1`. The checkpoint reloaded on CPU, with 294 optimizer states totaling 3,952,722,584 bytes. The worker took 47.776 seconds; CUDA peaks were 6,167,243,776 allocated and 6,805,258,240 reserved bytes.

A separate owned stock CUDA AdamW comparison covered all 294 parameters and found exact equality of updated weights, moments and step counters. All 294 expected-shard hashes and the saved checkpoint hash were checked before publishing this receipt. The training-worker receipt remains `numerical_oracle_status=not_run`; the later oracle has its own receipt and identity.

## Resource limitation

The original oracle reported 14,821,785,600 available host bytes at completion, below its 17,179,869,184-byte floor. Its checks covered admission and periodic generation, leaving comparison unchecked. Thus the exact arithmetic result does not prove complete resource-floor compliance. Later versions enforce comparison boundaries and preserve their failures; no floor was lowered and no training update was repeated.

Resume and BF16 remain unproved. The frozen examples are development mechanics data, not representative reviewed deployment decisions. This is not native derivative acceptance, trained context, quality, Jev parity, or issue closure. The independent [trained derivative failure](../native/rank0-derivatives-failed-20261002.md) remains unresolved.
