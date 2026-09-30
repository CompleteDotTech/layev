# Native context acceptance: report protocol v3

This harness is for Layev issue #4, with native prerequisites from issue #3.
A green source test cannot establish trained native 32k/64k context or predictive
quality. The code still requires an actual pinned pretrained checkpoint, a trained
head, held-out calibration, verified training exposure, a pinned Transformers
reference, and supported CUDA hardware on the authorized model workstation.
No cloud workload, download, private-index change or remote telemetry operation is
started by this change. Existing model, tokenizer v4, checkpoint, training, API,
telemetry, context ceilings and batch policy are not changed.

## Observed failure and development evidence (September 30, 2026)

The [observed development diagnostic](evidence/context/development-factorial96-20260930.md)
and [portable receipt](evidence/context/development-factorial96-20260930.json)
retain the fixed v8 marker failure: 3/18 against the unchanged 80% requirement.
This differs from the earlier v6 6/18 historical failure. The separate 96-case
short/medium readout scores Choice 79/96, Score 90/96 and Noul 48/96; every
Noul prediction is true. It does not establish exact-limit correctness, a
representative quality result, or a passing native-context-v3 gate.

The label readback finds no mismatch. Target correlations and wording/rubric/
distractor differences leave multiple explanations for these failures. The
separately frozen paired144 attempt subsequently timed out with 96/144
durable updates; see the [timeout receipt](evidence/context/paired144-timeout-20260930.md).
No completed repair, calibration or readout result is claimed. A new run initialized from weights resets
current-run exposure; resumed training preserves it. A separately fixed
exact-limit continuation is required before final validation.
Protocol ceilings, numerical tolerances, marker threshold and issue status
remain unchanged.

## What changes in the acceptance report

The protocol is now `native-context-v3`; preserve prior v2 reports as historical,
not equivalent reruns. A v3 result requires all of the following:

- Exactly 32,768 state-plus-branch tokens and 65,536 aggregate logical tokens,
  counting state once, with every requested question present and ordered.
- Explicit `ContextOverflow` evidence for both ceilings reduced by one token:
  branch 32,767 and aggregate 65,535. The error's code, limit name, actual count,
  maximum and applicable question ID must match. Ordinary `ValueError`, another
  exception, a different overflow, or silent acceptance cannot become a pass.
- One one-dimensional floating-point logit vector per branch on each execution
  path, with exactly one value per option. Shape, dtype and device must match;
  all outputs and calibrated probabilities must be finite. Python zip truncation
  and tensor broadcasting cannot conceal absent branches or wrong shapes.
- The installed Transformers distribution must be exactly 4.57.1 and agree with
  the imported package version. The distribution check precedes checkpoint
  loading; the imported-package check occurs when the oracle is constructed.
  This is a version-consistency check, not a cryptographic attestation of every
  installed package file.
- The requested CUDA index is resolved before loading weights. BF16 capability is
  checked on that selected device; synchronization and peak allocation calls use
  the same explicit device. Autocast construction and execution use the selected
  device context, restoring the previous current device afterward.
  CPU/unavailable-device requests fail without a model load or output-directory
  creation. No fallback device is selected.

Per-case receipts add `overflow_evidence` and `measurement_device`, retaining the
existing `overflow_rejected` field. It is true only after both exact rejection
checks complete. Reference receipts include the checked Transformers version.

The original tolerances remain unchanged: logits/probabilities use absolute and
relative tolerance 1e-5, the protocol's gradient tolerances remain 2e-5, and oracle
hidden states use 1e-4. BF16 does not relax them. The marker threshold is still
0.80; it is separate from the retained 0.70 historical smoke-quality threshold.
The known `color=red; level=1; case=99999` failure is not repaired by this harness.

## Run on the authorized local model workstation

Use the repository's separate supported model environment, current main plus the
reviewed change, existing pinned source artifacts and the actual trained,
calibrated checkpoint. Inspect the PC's GPU and runtime rather than assuming a
CPU-only PyTorch installation proves the absence of CUDA-capable hardware.
Do not move CUDA work to a remote machine for this gate.

```powershell
python -m pytest -q
python -m compileall -q src tests scripts
python scripts/validate_native.py --checkpoint <trained-calibrated-checkpoint> --out <new-empty-output-directory> --device cuda:0 --precision fp32
```

Execute a separate BF16 measurement only when the selected device supports it;
use another fresh output directory. The CLI's nonzero exit/raised prerequisite
error is a blocked or failed gate, never a passing skip. Missing weights,
calibration, exposure or reference dependencies must be resolved before asserting
a native result. Existing nonempty output directories are not overwritten.

Peak allocation is the PyTorch allocator's peak on the selected device, not total
GPU process memory or billed cost. Reference measurement may include still-live
optimized outputs; these are absolute peaks, not isolated incremental memory
claims. Timing still describes the one synchronized invocation in each case,
not a production latency distribution. The eager oracle checks a short sequence,
while long-case reference execution remains the independent model path. The
constructed marker cases are diagnostic, not representative quality or Jev parity.

## Source regressions and scope

`tests/test_native_acceptance_guards.py` exercises actual byte serialization and
CPU tensor comparisons. Its explicitly injected model/oracle/CUDA/exposure
collaborators test the real harness's control flow only. Simulated native reports
remain in memory inside tests and are not exported as acceptance evidence.
The tests prove rejection behavior, not authentic model provenance or hardware
measurements. In particular, exact byte-fixture lengths do not establish Qwen
lengths, training exposure, native inference, calibration quality, or UI readback.

This change does not supply enforced main-branch checks, an approved
representative dataset, official TypeSafe SDK acceptance, authenticated Overwatch
backend access, the original 45-feature matrix reconciliation, populated Overwatch
UI, live S3/W&B transport or Jev access.
Those acceptance gates remain independently required.
