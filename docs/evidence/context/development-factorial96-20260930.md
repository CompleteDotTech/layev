# Observed development diagnostic — September 30, 2026

The [portable receipt](development-factorial96-20260930.json) records a fixed
separately calibrated v8 checkpoint on 96 generated cases / 288 tasks. It
preserves report, data, checkpoint, source and forensic hashes plus every reported
slice and factorial cell. This is observed synthetic development evidence.

| Task | Correct | Main failure pattern |
| --- | ---: | --- |
| Choice | 79/96 | All 17 errors predict `unrelated-00`; 14 have first-ranked gold |
| Noul | 48/96 | All 96 predictions are true; false facts score 0/48 |
| Score | 90/96 | All six errors have gold zero; zero succeeds 4/10 |

Choice first/middle/last ranks score 18/32, 30/32 and 31/32. Counts 3/6 score
42/48 and 37/48; short/medium and en/es each score 39/48 and 40/48. The receipt
includes all eight slice families for all three tasks and every factorial cell,
with counts, accuracy and calibration metrics. Counts agree with raw records.
The independent label readback finds no gold/ordered-option/one-hot mismatch.

The generator correlates Noul truth with Score parity (true for even, false for
odd) and Choice route with Score modulo five. The historical proposal shorthand
`truth=index%2` has the wrong boolean direction; its original bytes remain
preserved. The parity-correlation conclusion remains valid. Development-marker wording,
rubrics and distractors differ from historical training. Those confounds permit
explanations involving state use, wording transfer or lexical/rank bias; these
observations do not establish one cause. Earlier training included false and
zero targets. Positive temperature calibration cannot change argmax.

The original native-context-v3 marker gate remains failed at 3/18 against its
unchanged 80% threshold. This short/medium readout establishes no 32768/65536
performance, untouched acceptance quality, representative deployment quality,
or Jev advantage. A separately proposed balanced paired repair is development
work. It was pending at the original readback; the subsequent attempt
timed out with durable96/144. The separate [timeout receipt](paired144-timeout-20260930.md)
records that outcome; there is no completed repair, calibration or readout result.
