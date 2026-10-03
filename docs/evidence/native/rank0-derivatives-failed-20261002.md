# Trained rank0 derivative failure — October 2, 2026

The [portable receipt](rank0-derivatives-failed-20261002.json) records all six completed phases and five comparisons under frozen tolerances. Only batched/serial passes. Native/HF backbone fails 132/290 gradients (worst ratio 46.2653); native/HF task fails 194/294 (94.1157); serial/full and batched/full each fail 22/294 (5.2974 and 5.2984). Native mode logits and probabilities agree exactly, but that does not establish gradient equivalence. HF hidden, Choice/Score logits and Score probabilities also fail.

CPU metadata verified 294 tensors / 494,090,176 scalars, including 290 backbone tensors / 494,032,768 scalars. The earlier run stopped on an incorrect inventory assertion; that failure and original artifacts remain preserved. Correcting the assertion did not relax numerical criteria.

The checkpoint retains historical stock-optimizer birth provenance; evaluation used the reviewed 33-file runtime. This probe constructs no optimizer and establishes no CPU-state optimizer arithmetic result, quality benefit, or issue closure. Existing acceptance flags remain unchanged.

Later [prefix-gradient diagnostics](prefix-gradient-diagnostics-20261002.md) rejected three additional candidates and document the corrected normalization-only replay. They do not change this baseline or acceptance status.
