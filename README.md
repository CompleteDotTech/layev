<p align="center"><img src="https://raw.githubusercontent.com/Jev-Engineering/.github/main/profile/assets/layev-banner.jpg" alt="Parallel question lanes converging through a calibration dial and scorer gauge into one chosen option" width="100%"></p>

# Layev

Layev is an open research implementation combining Kev's learned option scorer and shared-state question execution with Laya-inspired calibration and reward training. The Python package and CLI retain the `kev-laya` / `kev_laya` names. The code provides typed Choice, Score, and Noul decisions, bounded parallel question batches, training, evaluation, calibration, telemetry, and an Overwatch integration payload.

**Status:** The source has passed CPU fixture tests in its delivery environment. It is a research candidate, not a validated Jev replacement. Native Qwen/tokenizer and CUDA checks, trained 32k/64k context, general predictive quality, Jev-relative quality/cost/latency, and a complete live Overwatch pipeline remain open. See [the next-stage implementation prompt](docs/NEXT_STAGE_PROMPT.md) for the acceptance plan.

## Unreleased source-hardening work

The handoff recorded **280 passed, 5 skipped** on Linux CPU. A fresh Windows run
here also passed **280 tests with 5 skips**, including the real Rust byte-level
tokenizer test after correcting its trainer input type. These are separate from
the historical archive result. The strict 24-configuration audit,
exact interrupted/resumed continuation, real 16-row question batching, fresh
calibration/export and installed-wheel loopback checks have been exercised.
The guarded integration upgrade, source/index checks and group-aware uncertainty
reporting are documented in [release readiness](docs/RELEASE_READINESS.md).
See [current validation](docs/VALIDATION.md) and [environment boundaries](docs/ENVIRONMENTS.md).

These results do not close pinned Qwen/CUDA, trained 32k/64k context, representative
predictive quality, Jev comparison, or the full Overwatch backend/report/UI pipeline.
Remote CI and merge must be checked separately from local validation.
The repository's existing lock is retained unchanged.

## Layout

- `src/kev_laya/`: model, training, serving, telemetry, and checkpoint code.
- `tests/`: software and numerical regression tests.
- `integrations/overwatch/`: guarded adapter payload and instructions for the existing Overwatch application.
- `docs/`: architecture, telemetry, correctness, and validation notes. [The archive README (link-normalized copy)](docs/ARCHIVE_README.md) is retained for provenance; its relative evidence paths refer to the separate delivery ZIP.
- `licenses/` and `NOTICE`: upstream attribution and license texts.

The source came from `Kev_Laya_Correctness_Integration.zip` (delivery version `0.1.0+stage3`). That ZIP contains the larger historical evidence and trained fixture artifacts. Those artifacts and any private environment values are not part of this source repository. The [original 45-feature research matrix](docs/research/FEATURE_MATRIX.original.json) is preserved byte-for-byte with its [Layev evidence mapping](docs/research/layev-feature-mapping.json). The mapping records separate source, test, native, quality, and Jev evidence states. It does not establish Jev parity.

## Local setup

Use Python 3.12 or 3.13 in an environment separate from Overwatch. `requirements-cpu.lock` records the delivery's Linux CPU closure; it is not a validated Windows or CUDA lock.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e '.[dev]'
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m compileall -q src tests scripts
```

On POSIX, use `.venv/bin/python` in place of `.venv\Scripts\python`. Native requirements and prerequisites are described in [stage-three correctness](docs/STAGE3_CORRECTNESS.md). Do not run native training or cloud jobs without checking the required hardware, licensed data, and cost boundary.

The [Overwatch integration guide](integrations/overwatch/README.md) documents its separate supported Python and private dependency environment. The current Windows payload uses `ModelRunsView.tsx` to avoid a case-insensitive filename collision with `modelRuns.ts`.

## License

Apache-2.0. See [NOTICE](NOTICE) and the retained [Kev and Laya licenses](licenses/).
