# Correctness and integration stage 3

This release extends the verified parallel-question package (`b78c6991604c86f6ceca817135a6667199f342643de42b76a5f7dcecfd3c04a5`). It is not a completed Jev replacement. Software contracts, real pretrained execution, long-context training/execution, predictive quality, calibration, live Overwatch integration and operating cost are separate gates. See the delivery's acceptance and validation reports.

## Lossless native literal content

New Qwen tokenizer instances use `literal-special-v2` and `kev-laya-prefix-v2`. `tokenizers==0.22.1` is configured once with `encode_special_tokens=True`: literal special-token spellings pass through ordinary tokenization instead of becoming added structural tokens. There is no character replacement, escaping heuristic, normalization, truncation, or silent fallback. Every literal encoding is checked for forbidden special IDs and decoded back to the identical original string. This decoder-left-inverse check establishes injectivity **for accepted literal inputs**; a lossy backend fails explicitly. Startup probes use the real loaded backend when available. Structural delimiters are inserted as IDs only by the serializer.

State is a typed canonical JSON envelope (`{"state": ...}`), so the object `{}` differs from the string `"{}"`. Instructions and options retain typed canonical JSON serialization. Question IDs never enter these tensors. Logical input usage is the length of the actual serialized state once, plus the actual suffixes, including delimiters and every rubric/option. Overflow is checked after actual tokenization, not an estimated character count.

Old Qwen checkpoints explicitly recording `special-spelling-v1` retain that old, lossy rule. They are flagged in model metadata and cannot qualify for the v2 native acceptance gate. Unknown or contradictory preprocessing versions fail to load. There is no automatic weight migration to a new serialization. The UTF-8 byte **fixture** tokenizer remains v1 for its existing checkpoint compatibility; its literal byte encoding was already reversible. This does not turn it into a native Qwen tokenizer or retroactively make the legacy full serialization injective.

Executed tests use the byte fixture and explicitly labeled recording backend doubles. Real Rust BPE and pinned-Qwen tokenizer tests are included but were **skipped because tokenizers and the pinned tokenizer files are unavailable**. The native serialization fix is implemented and contract-tested, not empirically validated on the real tokenizer in this environment.

## Durable attempts and uncertainty

`attempt-lineage.json` stores up to 64 consecutive, SHA-256-linked receipts independently of optimizer save frequency. Each receipt binds experiment/run identity, attempt ID/index, parent attempt ID and predecessor receipt hash. It is written before the attempt's first export and copied into v2 snapshots/checkpoints. A restart from an old optimizer save reconciles overlapping receipts against that save, then advances the newest durable local chain. Contradictory identities are rejected, not guessed.

The collector verifies receipt continuity from an observed anchor. Missing attempt 1 no longer strands a collector at attempt 0 when the actual snapshot is attempt 2. Rehashed contradictory anchors, arbitrary larger indexes, conflicting transports, regressing sequence/counters and identity changes are rejected or quarantined. A proof gap beyond the retained window, conflicting transports, or lost proof is represented as a **new unverified observation** beside the **last verified attempt**. It is never silently presented as the current verified run. A collector with no prior verified root can show an explicitly unverified observation. Cache persistence retains these distinctions across reader/collector restart.

Receipts are integrity evidence from a trusted local producer/authorized transport, not signatures establishing the truth of arbitrary attacker-written histories. Concurrent writers to one training output directory are unsupported. Corrupt or contradictory identity ledgers may require operator recovery rather than silently inventing a lineage. Operational ledger write/read failures are recorded; ordinary snapshot/diagnostic export failures do not invalidate optimizer checkpoints.

## Collection admission budgets

Registry validation finishes before snapshot payload I/O. Defaults per pass: 512 sources, registry 128 KiB plus one counted registry-size probe, 16 MiB aggregate exposed payload, 512 request slots, eight W&B pages, 4,096 stream reads and a 15-second cooperative deadline. Local files must be regular files; supported platforms reject symlinks and FIFO races. Stream reads request at most 32 KiB and never more than the remaining allowance. Unknown-length overflow probes count against the budget. Invalid JSON bytes count too. Once a budget or deadline is exhausted, subsequent sources are not fetched/read.

Local/S3 body bytes and the **entire W&B GraphQL response body**, including its envelope, are charged. TLS, response headers, credential-provider traffic and other SDK-hidden transport overhead are explicitly unmeasured. W&B uses explicit streamed pages rather than eager SDK iterator hydration; redirects and implicit pagination are disabled. S3's internally created client has one total attempt and bounded connection/read timeouts. Injected transports must honor the bounded-read/no-hidden-retry contract. Production W&B/S3 exchanges were not performed; network lifecycle checks use instrumented doubles.

The deadline is checked before each new request/read, with network timeouts capped at the remaining allowance at call admission. **It is not hard cancellation of an already in-flight SDK call or credential-provider setup.** The result records this limitation. Work stops after a slow call returns; a strict end-to-end wall-clock cap on opaque SDK internals is not established.

All configured source IDs are retained even when unvisited. Partial/error/skipped sources keep their last cached records and last successful refresh timestamp. Missing registry means unavailable, not deregistration; only an explicit valid empty registry revokes all registrations. Structured status distinguishes a valid registry, a complete collection, and a partial collection. An early full-benchmark subprocess run hit the tool execution deadline; its partial log is retained separately from the complete rerun in bounded measurement groups.

## Exposure through calibration

`training_exposure` is independent of `training_state`, optimizer and RNG state. The trainer observes actual encoded branch/aggregate lengths and counts at optimization boundaries. Calibration/export copies this immutable identity/counter record and binds an explicit parent checkpoint basename and SHA-256. `verify-exposure` checks the artifact hash, available manifest, parent hashes, tokenizer/serialization/backbone identity, unchanged derived evidence, and a root matching the trainer's observed optimization-boundary record. It follows only sibling filenames or explicit `--parent` files, at most 64 derivations.

Calibrated checkpoints remain inference-only (`training_state`, optimizer, scheduler and sampler are absent). Legacy/missing/malformed/tampered evidence is **unknown**, not reconstructed from context configuration. A native 32k/64k exposure result additionally requires a pinned, manifest-bound Qwen base/tokenizer, actually loaded weight hash, native evidence class and observed examples reaching both thresholds. Tiny counters cannot satisfy it. Local artifact hashes are not a cryptographically signed training attestation; expected hashes must come from a trusted source.

Same-policy CPU resume is exact in tests and a fresh interrupted/uninterrupted run. Changing preprocessing, batching, frozen data, optimization configuration, explicit run/experiment ID or W&B identity is rejected. Historical checkpoints remain usable for inference/new-run initialization. Historical optimizer saves without this release's configuration/exposure evidence are not silently resumed; use `--init` for a new, explicitly distinct run. This conservative compatibility boundary is intentional.

## Provenance, resources, publishing and counters

Source metadata records actual source-module hashes and a source-tree hash. When Git exists, bounded startup queries record the actual revision and dirty state and sanitize the repository origin. Archive-only code has unknown Git identity; no commit is invented. An explicitly supplied archive can be hashed. The measured experiments record actual module/source hashes; their archive hash is unknown at runtime, with the final delivery archive independently hashed afterward.

Actual configuration bytes, canonical resume configuration, data manifest and every split have distinct recorded hashes. Tokenizer/serialization, model/backbone, precision and measured hardware labels persist. RSS and available CUDA allocations/peak allocations are sampled at an explicit minimum interval with timestamp and byte units. Samples are activity-driven and may miss brief peaks; unsupported measurements are null. The implementation does not infer utilization, billing or energy cost.

W&B publication is opt-in via a complete entity/project/run ID plus `--wandb-publish`. It updates **an existing remote run's** config and summary via the public API; it does not create a run or call `finish()`. The scheduler/application owning that remote run retains lifecycle ownership. Disabled publishing imports/calls no remote SDK. Remote write errors increment monitoring failures without aborting optimization; real publication remains unverified.

Optimizer steps, gradient-accumulation microbatches/examples, prefix passes, question-batch backbone passes, question counts, useful forward tokens, padding tokens and total forward compute are separate. Forward counters exclude backward activation-checkpoint recomputation; they are not FLOP or billing measurements. The parallel kernels and FP32 acceptance tolerances are unchanged. Serving uses the same path and exposes execution headers plus bounded aggregate v2 telemetry. Request/response content is not retained or added to training data.

## Reproduce local software validation

In the model's independent environment, from `kev-laya`:

```sh
python -m pip install --no-deps -e .
python -m pytest -q
python -m compileall -q src tests scripts integrations/overwatch
python scripts/verify_parallel_numerics.py --out fresh-numerics.json
python scripts/verify_stage3_local.py --out fresh-software-run
python scripts/verify_parallel_http.py --checkpoint fresh-software-run/run/calibrated.pt --out fresh-http
python scripts/stage3_experiment.py --out fresh-counterfactual-run
python -m kev_laya verify-exposure --checkpoint fresh-software-run/run/calibrated.pt --out fresh-exposure.json
```

Do not overwrite recorded evidence. The counterfactual script freezes one seed, splits, source hashes, all training budgets, thresholds, calibration-only fitting and terminal-checkpoint readout before outputs. The prior 50%/70% failed smoke remains visible; it is not a controlled comparison with this new diagnostic. The new diagnostic contains all six color/level combinations within each case/language/decoy/option-order group, with group-disjoint splits. Marker languages are not multilingual comprehension. The known case 99999 regression is excluded from all suite partitions and reported separately. Reward training did not outperform the supervised controls; Noul and Score still fail the exact known regression after calibration. Do not use these fixture checkpoints for consequential decisions.

Native prerequisites can be inspected without downloads or jobs:

```sh
python scripts/check_prerequisites.py --out fresh-prerequisites.json
# Only after independently obtaining verified pinned files and sufficient authorized hardware:
python -m kev_laya native-init --backbone-dir /verified/qwen --out runs/native/init.pt
# Train with a deliberate fitting execution/cache policy and actual long examples; calibrate held-out data.
python scripts/validate_native.py --checkpoint runs/native/calibrated.pt --out native-fp32 --precision fp32
python scripts/validate_native.py --checkpoint runs/native/calibrated.pt --out native-bf16 --precision bf16
```

The default 512 MiB cache admission policy need not fit a Qwen 32k parent cache. Select a hardware-measured policy explicitly in the training config; a larger policy is not evidence of native support. The original software-stage snapshot did not include native weight, tokenizer or CUDA runs. Subsequent real-weight results are recorded in [the native evidence readout](FP32_PROJECTION_PARITY.md): exact-limit BF16 LoRA backward completed, while both full-context validators failed overall marker accuracy and the short BF16 mixed backward diagnostic failed numerical parity. Representative licensed datasets, pinned parent baselines and an authorized Jev comparison remain required for broad quality/parity claims.
