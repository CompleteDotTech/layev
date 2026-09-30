# Rank-8 noncheckpoint CUDA mechanics — September 29, 2026

Refs #3; partial evidence only. The [portable receipt](noncheckpoint-rank8-mechanics-v10-20260929.json)
records separate FP32 and BF16 cells with actual pinned Qwen backbone and
tokenizer provenance. Activation checkpointing is disabled in both cells.

Each cell initializes its own pointer head, completes generated short training
step 1, reloads the checkpoint for a complete finite forward, resumes the same
run/attempt lineage through step 2, and reloads again. Step-2 parent checkpoint
and attempt IDs point to step 1. All 390 parameter tensors are finite on reload;
the generated three-question request has 230 logical tokens. No monitoring
export failures were recorded. The CUDA allocator cap remains 70%.

| Precision | Step-1 peak allocation | Step-2 peak allocation |
| --- | ---: | ---: |
| FP32 | 2,295,416,832 bytes | 2,300,203,008 bytes |
| BF16 | 4,246,000,640 bytes | 4,250,786,816 bytes |

These are independently initialized cells. Loss values and memory observations
are individual measurements, not a matched precision accuracy/performance
comparison. Passing resume mechanics does not establish equality with an
uninterrupted run, installed serving, representative quality, long-context
training, or completion of the applicable native matrix. Full-weight rank-0
cells remain separate, unclaimed work at this readback.

The mode orchestrator remains private. Its v3 runner hash, source hashes,
configuration identities and immutable checkpoint/reload/telemetry receipt
hashes are retained; this document provides no public runnable mode command.
The separately published [FP32 numerical reproductions](FP32_REPRODUCTION.md)
cover different protocols and must not be presented as mode orchestration.
