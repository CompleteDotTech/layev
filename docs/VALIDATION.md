# Validation gates and reproduction

## Executed in this session

The evidence directory at the bundle root contains JUnit test results, the actual miniature protocol,
immutable split manifests, complete per-question evaluation reports, real checkpoints and trainer
snapshots, adapter presentation output, runtime measurements and environment metadata. Unit tests
cover serialization/API semantics, boundaries, isolation, reference logits/gradients, finite PGPS
and LoRA gradients, Monte Carlo estimator agreement, exact interrupted continuation, calibration
partition refusal, integrity, quotas, overload, retention-disabled behavior and transport/cache cases.

TypeScript client/helper typechecking and helper runtime assertions are separate from a full
Overwatch React typecheck/build. Byte-code compilation is not a substitute for lint. The optional
official TypeSafe SDK gate is skipped when its package or explicit local target is absent.

Reproduce all independent work from `kev-laya`:

```sh
python -m pytest -q --junitxml=../evidence/retest.xml
python -m compileall -q src tests scripts ../overwatch-integration/added/src
cd clients/typescript && tsc --noEmit -p tsconfig.json
# Back in kev-laya:
python scripts/smoke_experiment.py --out runs/new-miniature
```

To test the official TypeSafe SDK, install it in a separate client environment, start the local
Kev-Laya server, and set `KEV_LAYA_BASE_URL`, `KEV_LAYA_MODEL` and `KEV_LAYA_TEST_API_KEY` explicitly
before running `tests/test_clients_and_installer.py::test_native_typesafe_sdk_gate`. Never point that
test at a paid external API without a supplied authorization and budget.

## Native backbone/context gate — not executed

Acquire the pinned Qwen files through the explicit script and verify `source.json`. Run native-init,
then meaningful SFT/reward training on a licensed, frozen long-context suite. Use LoRA or full-weight
configuration according to measured memory feasibility; do not substitute a raised config constant.

`python scripts/validate_native.py --checkpoint ... --out ... --device cuda` freezes its diagnostic
protocol before inference, rejects the fixture tokenizer/backbone, compares against a pinned HF
oracle, and constructs serialized 32,768-token branches and 65,536-token aggregate requests. It
measures actual forward tokens, GPU peak allocated memory and latency, preserves complete state,
checks question-independent reference outputs, records positions/languages and rejects overflow.
Its first native run and any debugging it reveals remain outstanding. Passing this diagnostic still
would not establish broad quality or Jev equivalence.

The training checkpoint records maximum branch/aggregate exposure. The native context gate requires
both, not just a successful serving pass. No native length exposure exists in the delivered fixture.
A calibration-only checkpoint is inference-only and should not be used to conceal missing training
provenance; run the training-exposure gate on the actual resumable training checkpoint.

## Real quality and baseline gates — not executed

The intended external acceptance protocol needs licensed frozen datasets, meaningful domains and
languages, semantic near-duplicate grouping, source provenance, actual length buckets, realistic
large candidate sets, and deployment-specific acceptance thresholds established before reading
results. Keep fixed test data locked during model/recipe selection. Report calibrated risk/coverage
and uncertainty, not just one accuracy number. No such external dataset was supplied or acquired.

Retain separate environments for the exact Kev and Laya revisions in `configs/sources.json` and
`evals/baselines.json`. Do not install those packages into this model or Overwatch environment to
hide version conflicts. Compare identical serialized semantic questions, label/tokenizer differences
and rejection populations. No parent baseline results or Jev comparison was fabricated.

The missing research package's original 45-column matrix has not been reconstructed from its broad
“Yes” descriptions. `ACCEPTANCE.csv` is a prompt-derived engineering matrix, **not** a mapping of
unseen research columns. Recover the named original files and map each actual column before closing
that gate. Integration evidence and functional feature coverage remain different questions.

## Overwatch and cloud gates — not executed

Run the revision-checked installer on the actual primary checkout; execute the new pipeline test,
all unchanged legacy tests, backend lint and the existing frontend typecheck/tests/build in its
supported environments. Confirm the running application actually displays local model runs,
curves, artifact links and fresh failure/recovery states through normal refresh. The provided pure
adapter tests do not replace this check.

A bounded SkyPilot pilot requires an explicit authorization, valid budget, compatible machine image,
checkpoint/artifact persistence and cloud identity. The candidate YAML is not a proven deployment.
Record provisioning/runtime cost, request latency distributions, memory, quotas and actual recovery.
Local costs remain unknown without a power/billing meter. Do not equate “no hosted API fee” with zero
operating cost or inherit TypeSafe's latency, price, quotas, reliability or enterprise terms.
