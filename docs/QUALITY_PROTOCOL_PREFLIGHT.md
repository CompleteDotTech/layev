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
specified. Paid use is zero for this local-only preflight. The opt-in quality
training path binds this protocol to an exclusive durable ledger and reserves
the exact encoded useful tokens before each optimizer step. It enforces
aggregate and per-seed/arm step and useful-token ceilings, checkpoints each
completed step, and refuses unreconciled in-flight steps or a run/checkpoint
mismatch. Use all three `--quality-protocol`, `--quality-data-review`, and
`--quality-budget-ledger` arguments with an explicit new `--run-id` for
`train` or `reward-train`. The configured seed and optimizer-step count must
match the frozen protocol.

The ledger starts the declared wall clock before model loading, persists that
start across runs, and checks the deadline before each step and between
microbatches. On CUDA the quality CLI caps this process's PyTorch allocator
before loading weights and checks its measured peak around optimizer work.
An overrun discovered inside a reserved step leaves that step pending for
audited reconciliation. These are cooperative checks: an in-flight CUDA kernel
can pass the wall deadline before returning, and the allocator cap does not
cover all driver or external GPU allocations. A process-level hard wall
deadline and complete GPU-process accounting remain outstanding. This source
gate alone does not prove that a review is authentic or the data is
representative.

For an approved quality training run, invoke the opt-in `kev-laya-quality`
entry point with the frozen protocol, reviewed data declaration, suite,
budget ledger and a new receipt path, followed by `-- train` or
`-- reward-train` and the normal training arguments. The supervisor starts
the ledger wall clock before launching a dedicated training child. It stops
that child at the declared deadline and writes a content-free receipt;
there is no automatic retry. On Unix, it stops the owned process group.
On Windows, it stops the direct child; a child-created descendant is not
covered. A supervisor crash can also leave the child running. Review any
pending ledger step against its checkpoint before resuming. This wrapper
does not measure non-PyTorch GPU allocations or establish the full quality
acceptance gate.

Selection uses development data only; calibration fits on calibration data only.
The guarded untouched-test readout is once per selected arm/seed, with raw and
calibrated results paired. Kev/Laya source revisions and artifact hashes, all six
metric families, five slicing dimensions, eight independent variation dimensions,
and group-level 95% uncertainty with at least 1,000 bootstrap replicates must be
specified. Actual baseline execution, resampling and metric calculations are
separate work, not performed by this script. Hash syntax is not artifact existence.

The evaluator carries optional `meta.option_order` and named
`meta.variations` strata into per-slice reports. Missing strata are reported as
`unknown`, never inferred from question IDs or answer labels. These fields let
approved suites supply independently reviewed variation labels; the evaluator
does not establish that the variations were independently sampled. The full
multi-seed quality experiment and budget enforcement remain outstanding.

The retained synthetic accuracy threshold stays **0.70**. The exact
`color=red; level=1; case=99999` state and Choice/Noul/Score readouts must remain in
the observed regression, with failures retained and reported. No regression result
is generated or changed by this preflight.

To score that previously observed case on an existing pinned checkpoint, run
`python scripts/report_quality_regression.py --checkpoint <checkpoint.pt>
--expected-sha256 <checkpoint-sha256> --out <new-private-receipt.json>`.
The report checks both option orders and all six Choice/Noul/Score decisions.
Its `all_six_correct` field requires 6/6. It is a diagnostic on a known
synthetic case, cannot select a checkpoint, and does not independently verify
training exposure or representative quality. Keep the receipt private until
its permitted fields and source/checkpoint hashes are reviewed.

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

The read-only preflight command does not reserve budgets, authenticate a reviewer, verify baseline
files, prevent future test-set reuse, train models, call Jev, or establish that a
protocol predates training. Preserve an externally witnessed/signed protocol
receipt before training, then bind all run, selection, calibration and evaluation
receipts to that immutable digest. Retrospective local timestamps are not proof.
The full issue remains open until its real data, execution and measured-result
gates are demonstrated.

## Claimed untouched-test readout

The quality CLI now requires a single-use claim for `evaluate --split test`.
Ordinary development evaluation remains available. Existing fixture experiments
must explicitly request `--diagnostic-test`, which the CLI accepts only for
known synthetic fixture IDs. Their report and telemetry mark them ineligible
for representative quality claims. The direct Python evaluator
also requires an active claim for test scoring, unless called as an explicit
diagnostic. This is an ordinary software guard, not a filesystem access-control
boundary against code that deliberately reads the suite directly.

After development-only checkpoint selection and calibration-only fitting, write
an immutable JSON selection record with exactly these fields:
`schema_version=layev-quality-selection/1`, `protocol_sha256`,
`suite_manifest_sha256`, `arm`, `seed`, `development_report_sha256`,
`raw_checkpoint_sha256`, and `calibrated_checkpoint_sha256`. The development
report must bind the selected raw checkpoint and frozen development split. The
calibrated checkpoint manifest must name the raw checkpoint as parent. Preserve
the selection record before opening the test partition, ideally under an
externally witnessed preregistration receipt; a local file timestamp alone does
not prove selection chronology or that the development rule chose the best arm.
The selected raw checkpoint must record positive training steps, the declared
seed, all frozen split hashes, unfitted after-training calibration status and
unit temperatures. These recorded fields provide a local consistency check;
they are not independent proof that an optimization run actually occurred.

```powershell
python -m kev_laya evaluate --split test `
  --checkpoint C:\approved-research\selected-raw.pt `
  --calibrated-checkpoint C:\approved-research\selected-calibrated.pt `
  --suite C:\approved-research\suite `
  --quality-protocol C:\approved-research\protocol.json `
  --quality-data-review C:\approved-research\data-review.json `
  --quality-selection C:\approved-research\selection.json `
  --development-report C:\approved-research\selected-development.json `
  --quality-readout-ledger C:\approved-research\test-readout-ledger.json `
  --out C:\approved-research\paired-test.json
```

The ledger claims `arm:seed` under an exclusive lock before preflight or the
suite loader reads test rows. Its in-process permit is consumed exactly once for
the selected raw checkpoint and once for its calibrated descendant, before each
model forward pass. Both results are written to one paired output. The claim
permanently blocks another evaluation for that arm and
seed under the frozen protocol, even after a failure. If the process crashes
after writing a complete paired output but before committing its receipt,
reconcile that existing output without reading the test data again:

```powershell
python -m kev_laya reconcile-test-readout `
  --quality-protocol C:\approved-research\protocol.json `
  --quality-readout-ledger C:\approved-research\test-readout-ledger.json `
  --arm supervised --seed 42 --out C:\approved-research\paired-test.json
```

An incomplete or missing output stays claimed and cannot be retried. The ledger
checks the output's protocol, selection, arm/seed, test split and both checkpoint
hashes. It cannot authenticate a data-use review, independently prove the
development selection rule, or prevent an operator from bypassing the library
by directly reading data with unrelated code. Those remain acceptance review
requirements; this source guard does not establish representative quality.

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
