# Bounded pinned-weight BF16 CUDA training receipt

This is partial evidence for issue #3. The two-step run used the actual pinned
`Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`
weight file, SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
The source gate had verified those bytes. The tokenizer JSON SHA-256 was
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
No weights or training records were copied into Git.

The run used Python 3.13.12, PyTorch 2.10.0+cu128, Transformers 4.57.1,
tokenizers 0.22.1, and an NVIDIA RTX 3060 with 12 GiB VRAM. The runtime
reported BF16 support. The training source provenance recorded a clean Git
checkout at signed commit `34c7775fdecee28e3d184c075a53f501bb8edf77`
(later merged in PR #25); source-tree SHA-256
`e81ab8b9cbb81340c1cf9530e7c29944881b2c666f0bbae5c2957b03d70ef50b`.
The model/trainer bytes are the same as merged native PR #24. The configuration
is the two-step FP32 smoke configuration in
[the earlier receipt](two-step-train-resume-serve-20260927.md), changing only
`training.precision` to `bf16`; file SHA-256
`4cb55e9e5aaaf784cb1c445e48589e31b7729f456dee55c5ace74848b67e7c69`.
The same generated 64-row suite was reused, with 41 train, 5 development,
9 calibration, and 9 test rows. This is synthetic smoke data.

From the existing pinned-weight LoRA-rank-8, activation-checkpointed initial
checkpoint, training was stopped after step 1 and resumed from its exact
checkpoint with the same config, suite, experiment ID, and run ID. Both CUDA
optimizer steps completed. The step-1 checkpoint SHA-256 was
`0dc53a5278844ffbd5d845d6f23dff6fa10f97dd5a1b35628ce033995001e81c`;
step 2 was
`dd996659ff2453bd2995258b71a40c5ab8d4fbc42e0d73a32d901bbca1559ec5`.
`verify-exposure` returned `verified`, two optimizer steps, maximum branch 69
and aggregate 144 tokens, and `native_32k_64k: false`. Its
[raw receipt](bf16-exposure-20260927.json) SHA-256 is
`04b41cef29e16ce5205d5f95d658dbae85fa6ef1fe13afbe0f1620d8e8cbfb5e`.
The telemetry SHA-256 is
`ac98373b6e476bea69130d69dbf5da645908d4180417d0b346e40508e511c203`;
it records BF16 precision, two steps, five-second optimizer elapsed time
across attempts excluding process startup and checkpoint I/O, 288 useful
forward tokens, and sampled GPU peak allocated memory 4,037,493,760 bytes.
RSS was unavailable.

The step-2 checkpoint reloaded and passed the opt-in independent short
Transformers 4.57.1 hidden-state comparison at the unchanged combined
absolute/relative tolerance on a 62-token request. The maximum absolute
difference was `0.00018310546875`. The
[raw short-probe receipt](bf16-trained-short-native-20260927.json) SHA-256 is
`c69da3440608691b41376c9f109fb7e7c990f5d0b56b538396f0706821ab1f25`.
That comparison runs FP32 inference on the BF16-trained checkpoint; it does
not constitute a BF16 inference parity test.

## Full-weight memory boundary observed separately

An actual pinned-weight rank-0 initialization with activation checkpointing
produced a zero-step checkpoint SHA-256
`7b89e1082e449cd92c64072ef4af64fc6b0255dd8a0bd40d287155d71321769a`.
At an explicitly capped 8.64 GiB CUDA process budget, the merged AdamW path
ran out of memory in its foreach optimizer update before step 1 committed.
An uncommitted scalar-AdamW diagnostic also ran out of memory in its update.
An uncommitted fused-AdamW diagnostic saved one step, but resume ran out of
memory in backward after restoring optimizer state. No rank-0 change was
merged or claimed as a passing train/resume path. The failed runs remain in
private local artifact directories outside Git. This is a limit of the tested
RTX 3060, shape, optimizer, and cap; it is not a proof that full-weight
training is impossible on other authorized hardware or settings.

These small synthetic runs do not establish full-weight resume, broader
gradient and branch-isolation parity, trained 32k/64k context, calibrated
quality, or Jev comparison. Keep issue #3 open.
