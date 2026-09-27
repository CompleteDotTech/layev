# Representative-quality protocol preparation

Tracking issue: [Layev #5](https://github.com/CompleteDotTech/layev/issues/5).
This opt-in, read-only preflight is independent of the tokenizer repair and does
not change the training entry points, HTTP schemas, telemetry, existing runs,
model implementation, quality thresholds, or Overwatch integration.

## What the preflight establishes

It binds the exact bytes of a proposed protocol, data-review declaration and
existing suite manifest with SHA-256. It calls the existing `data.load_suite` to
validate the real dataset format, partition hashes/counts, record IDs, group
separation and normalized-state duplicate rejection. The current loader's
semantic near-duplicate limitation is unchanged. Manifest paths must additionally
be portable basenames; symlinked inputs, unbounded documents, empty partitions,
Boolean row counts, unknown protocol fields and malformed JSON are rejected.

The protocol must declare at least three distinct integer seeds in the trainer's
nonnegative range `0..2**64-1`, one pinned parent, matched supervised/reward
optimizer-step and useful-token budgets, and a
sufficient total step budget. Positive wall-time and peak-memory bounds must be
specified. Paid use is zero for this local-only preflight. These are declarations,
not a resource scheduler or enforcement by the existing trainer.

Selection uses development data only; calibration fits on calibration data only.
The planned untouched-test readout is once per selected arm/seed, with raw and
calibrated results paired. Kev/Laya source revisions and artifact hashes, all six
metric families, five slicing dimensions, eight independent variation dimensions,
and group-level 95% uncertainty with at least 1,000 bootstrap replicates must be
specified. Actual baseline execution, resampling and metric calculations are
separate work, not performed by this script. Hash syntax is not artifact existence.

The retained synthetic accuracy threshold stays **0.70**. The exact
`color=red; level=1; case=99999` state and Choice/Noul/Score readouts must remain in
the planned regression, with failures retained and reported. No regression result
is generated or changed by this preflight.

## What a successful receipt does NOT establish

`status=preflight_validated` means structural preparation and local byte checks
only. The receipt explicitly keeps review authenticity and preregistration
chronology unverified, representativeness unestablished, quality unmeasured,
regression/baseline/native/CUDA/context outcomes untested and Jev parity unknown.

A reviewer's name or a license string is not authentication, legal verification,
permission to move data, or proof that data is representative. An authorized
review must genuinely establish these facts separately. Pin actual approved
artifacts; do not copy placeholder hashes from tests. The known smoke and
counterfactual fixture IDs are rejected, but this is not an exhaustive synthetic
or semantic-data detector. Arbitrarily relabeling a fixture does not make it
representative. The tests use artificial review declarations and synthetic data
solely to test software boundaries; they are not a publishable quality benchmark.

This command does not reserve budgets, authenticate a reviewer, verify baseline
files, prevent future test-set reuse, train models, call Jev, or establish that a
protocol predates training. Preserve an externally witnessed/signed protocol
receipt before training, then bind all run, selection, calibration and evaluation
receipts to that immutable digest. Retrospective local timestamps are not proof.
The full issue remains open until its real data, execution and measured-result
gates are demonstrated.

## Running on existing approved data

Use the isolated Layev environment and a complete checkout. Keep CUDA and
Overwatch's environment on the authorized workstation. First complete a proposed
protocol and data-review record with real reviewed values. The two templates in
`docs/quality-protocol-templates/` deliberately contain null prerequisites and
cannot pass as delivered.

```powershell
python scripts/verify_quality_protocol.py `
  --protocol "C:\approved-research\protocol.json" `
  --data-review "C:\approved-research\data-review.json" `
  --suite "C:\approved-research\suite"
```

The paths above are examples, not discovered workstation files. Preserve the
content-free JSON stdout in a **new** evidence file, not an existing run record.
No endpoint, cloud credentials or paid budget is needed by this script. It does
not write or copy the dataset. Keep inputs immutable during the gate; final
readback detects ordinary concurrent edits, not malicious ABA filesystem races.
Do not run validation simultaneously with dataset edits.

Exit 0 is structural preflight only; exit 2 is missing local prerequisites; exit 1
is rejection or execution failure. JSON documents are bounded to 1 MiB each and
partition data to 256 MiB in total by default. The Python API supports an explicit
positive `max_suite_bytes` when the reviewed suite genuinely needs a larger
bound. The gate's memory footprint includes the existing loader's decoded suite;
this byte bound is not a precise RAM limit or a substitute for native memory tests.

```powershell
python -m pytest -q tests/test_quality_protocol.py tests/test_quality_protocol_templates.py
python -m pytest -q
python -m compileall -q src tests scripts
```

Run the unchanged required hosted/source checks before signed delivery. Isolated partial-export tests are not full-repository or hosted validation.
No acceptance box may be checked from a preflight receipt alone.
