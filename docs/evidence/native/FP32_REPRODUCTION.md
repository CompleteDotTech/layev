# Reproduce the scoped FP32 numerical evidence

Use the CUDA model environment with PyTorch 2.10.0+cu128 and Transformers 4.57.1.
Supply the locally retained trained native LoRA checkpoint; weights are not
included in this repository. Both scripts record runtime source identity and
refuse overwriting result artifacts. Output paths below must be fresh.

```shell
python scripts/verify_native_fp32_gradients.py --checkpoint CHECKPOINT.pt --expected-checkpoint-sha256 99f797ffd0521adbee964d130d73eb381c9b6bd3b7896ce61b5007886d19ccf9 --out fp32-hf.json
python scripts/verify_native_fp32_seams.py --checkpoint CHECKPOINT.pt --expected-checkpoint-sha256 99f797ffd0521adbee964d130d73eb381c9b6bd3b7896ce61b5007886d19ccf9 --out-dir fp32-seams
python scripts/verify_native_fp32_seams.py --checkpoint CHECKPOINT.pt --expected-checkpoint-sha256 99f797ffd0521adbee964d130d73eb381c9b6bd3b7896ce61b5007886d19ccf9 --out-dir fp32-long --cases boundary_8191 boundary_8192 long_10520
```

The first command compares hidden states and all Q/V LoRA parameter gradients
against independent eager HF parameters. The second runs four exact-token
cached/batched/full numerical cases. The third constructs the same generated
8191/8192 boundary and 10520-token state requests and applies the same loss and
three-mode backward protocol. It requires substantial GPU time. The 70% CUDA allocator cap and disabled
TF32 are fixed. HF hidden tolerance is 1e-4, logits/probabilities 1e-5, and
trainable gradients 2e-5 (combined absolute/relative). Generated requests
establish no representative quality, long-context or Jev benefit. The seams
protocol retains its explicit 16-GiB cache-policy ceiling; this is a protocol
planning ceiling, not allocator budget growth or a total-memory claim.
