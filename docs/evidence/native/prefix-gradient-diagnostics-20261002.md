# Prefix gradient diagnostics — October 2, 2026

The [portable receipt](prefix-gradient-diagnostics-20261002.json) records three rejected actual-weight candidates. The original [six-mode failure](rank0-derivatives-failed-20261002.md) remains the baseline. Runtime source and frozen numerical criteria were preserved throughout these external diagnostics.

| Candidate | Cached/full failed tensors | Native/HF task failed tensors | Batched/serial failed tensors |
| --- | ---: | ---: | ---: |
| FP32 attention products, FP64 shared sums | 38/294 | 194/294 | 0/294 |
| FP64 per-call projection parameter products | 21/294 | 194/294 | 0/294 |
| Branch-owned prefix VJPs | 27/294 | 193/294 | 0/294 |

The first two candidates ran six fresh modes. The branch-owned candidate ran fresh batched and serial modes and compared all 294 gradients with hash-verified preserved full-row and independent-HF task receipts for the same checkpoint and production source. It is a diagnostic execution version with no resume compatibility claim. Passing small operator tests did not predict full-weight acceptance.

## Localization and correction

In a separate unmodified actual-model trace, suffix inputs and layer-input gradients were exact between serial-cached and full-row execution across all 24 layers. Shared-prefix input gradients differed; layer 0 exceeded the frozen combined tolerance by a ratio of 3.7371. This localizes the observed activation-gradient difference without proving that every parameter-gradient error has the same cause.

Normalization inputs also feed residual paths. Their tensor hooks therefore capture combined gradients, not an isolated normalization VJP. The initial replay comparison was superseded. The corrected normalization-only replay still fails with the actual incoming prefix gradients: layer 0 input normalization has a stock ratio of 3.2461 and an FP64 candidate ratio of 3.2555; post-attention normalization has ratios 8.5656 and 8.6086. Matching incoming gradients makes the local candidate comparisons pass. Improving normalization precision alone is insufficient.

The frozen cache budget groups Noul/Choice together and Score separately: widths 57/135 and flattened suffix projection rows 114/135, plus one 25-token prefix. These are budget-driven groups, not explicit length buckets. Projection parameter-sum rounding and inherited shared-prefix gradient differences were both observed; neither tested intervention passed the complete gate.

## Acceptance status

Production code, existing acceptance flags, the independent reference, and declared tolerances are unchanged. These receipts establish no completed native gate, resume/BF16 qualification, predictive-quality benefit, continuous resource compliance, or issue closure. Original checkpoint provenance, failed candidates and the corrected analysis remain preserved locally. No model artifacts or private local paths are included here.
