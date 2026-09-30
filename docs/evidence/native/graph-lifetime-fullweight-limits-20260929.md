# BF16 graph lifetime repair and full-weight resource limits

Refs #3; incomplete acceptance. The [portable receipt](graph-lifetime-fullweight-limits-20260929.json)
separates historical numerical source from the subsequent training graph repair.
Previous numerical and rank-8 receipts retain their original source hashes.

Earlier BF16 branch backward retained graph owners through optimizer allocation.
Deleting the completed local logit, gradient, list and loss owners after backward
releases those activations before AdamW. The CUDA regression checks weak references
at optimizer entry: one pass; the proper old-source control fails that assertion.
All 14 selected BF16 tests pass, including the lifetime case; these counts are
not additive. The GPU owner recorded these terminal results; the reviewer
verified the observation receipt and did not rerun the GPU tests. This demonstrates graph ownership repair without
changing gradient order, optimizer parameters, allocator cap or resume identity.

| Generated noncheckpoint rank-0 cell | Step 1 / reload | Resume step 2 |
| --- | --- | --- |
| Historical FP32 | Passed / passed | Backward CUDA OOM |
| Historical BF16 | First AdamW moment allocation CUDA OOM | Not reached |
| BF16 same-init replay after graph repair | Passed / passed | Short attention backward CUDA OOM |

The repaired BF16 replay uses the exact original initialization hash recorded in
the receipt. A preliminary replay failed tokenizer setup before GPU execution;
it is preserved separately. The corrected replay's first step takes 39.363267
seconds, allocating/reserving peaks of 7,971,341,824 / 8,552,185,856 bytes.
Its resume fails after 51.652650 seconds with peaks of 8,638,769,664 /
9,017,753,600 bytes, at a 2-MiB short-attention gradient allocation. Historical
FP32 resume fails at a 520-MiB backward allocation. A single same-run FP32
allocator-option retry also fails; PyTorch explicitly warns that expandable
segments are unsupported on this Windows runtime. This observation does not
establish that all hardware or supported execution configurations are unsuitable.

The tested graph repair improves the observed BF16 first-step result. Complete
rank-0 train/resume mechanics remain unverified on this measured configuration.
The private mode orchestrator is not published here; no public runnable mode
command is claimed. The graph ownership test is reproducible with the CUDA model
environment:

```shell
python -m pytest -q tests/test_parallel_questions.py::test_bf16_branch_activations_released_before_optimizer
```

These generated mechanics establish no installed serving, quality, Jev benefit
or closure of #3/#4. The separate exact-limit training resource failure remains
recorded in the [FP32 numerical evidence](short-fp32-canonical-v10-20260929.md).
