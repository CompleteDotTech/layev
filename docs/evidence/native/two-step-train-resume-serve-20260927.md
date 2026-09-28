# Layev bounded native train, resume, and serving probe

Date: 2026-09-27/28, America/New_York. Scope: two optimizer steps on generated marker data. This is implementation evidence, not representative quality, calibration, long-context, or Jev comparison evidence. No remote job, W&B/S3 publication, or paid provider was used.

## Identity and environment

- Source package tree SHA-256: `25becf4a38f494a38337b946aff2506b45fb88836187ca5e476e9601a0717e9e`; all 27 recorded `src/kev_laya` file hashes match committed short-CUDA source commit `28e6c82f0ca2200f658b6b751311e7158ea9e2c5`. The run imported an isolated copy of those bytes, so its automatic Git status is `error` and commit is null; byte comparison establishes the correspondence.
- Python 3.13.12; PyTorch 2.10.0+cu128; CUDA build 12.8; Transformers 4.57.1; tokenizers 0.22.1; NVIDIA RTX 3060 12,288 MiB, driver 617.14.
- Pinned Qwen/Qwen2.5-0.5B revision `060db6499f32faf8b98477b0a26969ef7d8b9987`; `model.safetensors` 988,097,824 bytes, SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`. Source gate returned verified; auxiliary file origin remains local-manifest-only.
- Tokenizer JSON SHA-256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`; preprocessing identity: `qwen-tokenizer-sha256:c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`, `kev-laya-prefix-v2`, `literal-bpe-v4:002e76a7328e372b4ca78a015cc203a522994eb150be050f834a136572d4965e`. The same identity appears in initialization, both trained checkpoints, and the service model response.
- Synthetic suite from `freeze-smoke --count 64`: 41 train, 5 development, 9 calibration, 9 test rows. Manifest SHA-256 `29cbc2e7541cc0b8a9ec92f6b8583740b91fa33d1e4340e47b6ff9f0e17c84e0`; probe config file SHA-256 `d83f715550c1c7b15fb37da368f4d8a427642fa4f5c56e2170f13ba6a1327c26`. Configuration freezes two FP32 optimizer steps, accumulation 1, seed 412, LoRA rank 8, activation checkpointing, branch/aggregate limits 512/2048, and one-step checkpoint interval.

## Reproduction commands

Use a separate CUDA Python environment and fresh private paths outside Git. `<source>` is the pinned five-file snapshot; `<init>` is the actual initialized Qwen checkpoint; `<root>` is a new empty local directory. Set `PYTHONPATH` to the source bytes identified above.

```text
python -m kev_laya.cli freeze-smoke --out <root>/suite --count 64
python -m kev_laya.cli train --init <init> --suite <root>/suite --config <root>/config.json --out <root>/run --device cuda:0 --experiment-id layev-native-probe-20260927 --run-id layev-native-probe-20260927-a --stop-after 1
python -m kev_laya.cli train --resume <root>/run/checkpoint-000001.pt --suite <root>/suite --config <root>/config.json --out <root>/run --device cuda:0 --experiment-id layev-native-probe-20260927 --run-id layev-native-probe-20260927-a
python -m kev_laya.cli verify-exposure --checkpoint <root>/run/checkpoint-000002.pt --parent <init> --parent <root>/run/checkpoint-000001.pt --expected-sha256 b6e0db32a65bbf240a5cdec8c938040054a45284e5d4cf35cd310160262b6efb --out <root>/exposure-verification.json
python -m kev_laya.cli serve --checkpoint <root>/run/checkpoint-000002.pt --host 127.0.0.1 --port <free-loopback-port> --device cuda:0 --no-auth --telemetry <root>/serve-telemetry.json --experiment-id layev-native-probe-20260927 --run-id layev-native-probe-20260927-serving
GET http://127.0.0.1:<free-loopback-port>/v1/models
POST http://127.0.0.1:<free-loopback-port>/v1/systemone
Content-Type: application/json
{"state":"A short real-weight reference state.","model":"kev-laya-preview","questions":{"q":{"type":"noul","instructions":"Is the state short?"}}}
```

The probe `config.json` is:

```json
{"training":{"steps":2,"accumulation":1,"learning_rate":0.00001,"weight_decay":0.0,"seed":412,"gradient_clip":1.0,"save_every":1,"keep_checkpoints":2,"precision":"fp32","choice_permutation":false,"resource_sample_seconds":0.1},"objective":{"ce":1.0,"reinforce":0.0,"ordinal":0.0},"limits":{"branch":512,"aggregate":2048,"max_questions":3},"execution":{"version":"parallel-questions-v1","max_branches":3,"max_padded_tokens":2048,"max_cache_bytes":67108864,"sort_by_length":true}}
```

Save that JSON as one UTF-8 line with a trailing newline to match the recorded file hash. These commands cannot reproduce the same byte hashes unless the exact source, frozen suite, config, runtime, and RNG state are used.

## Observations

- Initialization checkpoint SHA-256 `ead1228e11d0415716541208745df23a8a938dcbca76b24e9d91b53495a3705d`, zero training steps.
- Step 1 checkpoint SHA-256 `73745f4f95ef24d5f37016d6e390b13fc5a5c04047e48eaec5d77f327ef21c7a`, parent init hash, resumable. Step 2 checkpoint SHA-256 `b6e0db32a65bbf240a5cdec8c938040054a45284e5d4cf35cd310160262b6efb`, parent step 1 hash, resumable. Both checksum manifests match measured checkpoint bytes.
- One optimizer step before interruption and one after resume; experiment/run IDs stayed fixed. Resume attempt index advanced from 0 to 1 with linked attempt IDs/receipt hashes; two examples, two microbatches, 288 useful forward tokens, 330 compute tokens, 42 padding tokens, and zero monitoring export failures. Telemetry phase completed.
- Sampled CUDA peak allocated memory 4,018,332,672 bytes during training; optimizer elapsed time 2.064 seconds across the two attempts, excluding process startup, checkpoint load/save, and service startup. RSS was unavailable.
- Exposure verification returned `status: verified`, `evidence_class: pretrained-backbone`, two optimizer steps, maximum branch 69 and aggregate 144 tokens, and `native_32k_64k: false`.
- Loopback GET `/v1/models` and POST `/v1/systemone` both returned HTTP 200. The model resolved to `kev-laya-0.1.0-b6e0db32a65bbf24`; response was one finite Noul answer (`q.noul = 6.490285338407294e-9`), input 62, output 33, forward 62 tokens, with the same v4 input identity and `unfitted-after-weight-training` calibration status. The temporary server was stopped after the readback.
- The original trained-checkpoint oracle run failed unchanged hidden-state tolerance: max absolute error `0.000293731689453125`. A separate outside-repository reference-adapter candidate passes on the same checkpoint with error `0.00011444091796875`; that candidate has not been merged. Preserve the failed report and do not close trained oracle parity from the successful training or serving results alone.
