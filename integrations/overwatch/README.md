# Revision-checked Overwatch integration, stage 3

Target: `CompleteDotTech/Overwatch@501bd99a0cb0c6b321feb022f5747d6dcbd5e9c4`.
The published baseline README recorded a Windows preflight/application, eight
Vitest tests, a frontend build and local provider-to-presentation verification on
2026-09-26. It also recorded a private-index 401 blocker and an uninspected UI.
Those are retained historical reports, not results independently rerun here.

This source-hardening overlay tests guarded fresh/v1/stage-three migrations on
Linux fixtures, including LF/CRLF files and rollback. On the primary Windows
checkout it passed preflight and updated one known frontend test. Eight frontend
tests and the production build passed. Full backend and populated UI verification
remain open.

## Apply in the primary checkout

From the model project, in PowerShell:

```powershell
python integrations/overwatch/apply.py --root 'C:\path\to\Overwatch' --check
python integrations/overwatch/apply.py --root 'C:\path\to\Overwatch' --apply
```

Read applicable `AGENTS.md` in the primary checkout first. The installer checks the exact Git revision and every modified base file's normalized Git blob SHA. It accepts the exact base, the exact known v1 integration, or its own exact current delta. Reverse patching must reconstruct the verified base bytes: arbitrary edits are never treated as an upgrade. New-file replacements require a known v1/stage-three hash or the current bytes. Historical `./ModelRuns` and current `./ModelRunsView` imports are recognized only by exact reverse-to-base proof. An obsolete `ModelRuns.tsx` is removed only after verifying its known hash; changed or unknown obsolete files are refused. Backups and rollback include deletions. Unknown target edits, review paths, symlinks and Git worktrees are refused. Unrelated files are untouched. Writes are backed up under `.kev-laya-stage3-backup-*`, atomically replaced and rolled back on failure. A different revision requires deliberate review/rebase, not `--force`.

The delta extends the eight inspected base files: collector, raw cache, cached metrics, report, report service, constants, frontend types and App. Added modules contain the versioned provider, pure adapter, standalone contract/lock, model-run frontend components/tests and native pipeline test. This payload preflights as applied against the current primary checkout; future revisions require a deliberate rebase.

## Real data path

`trainer atomic export -> collector.collect_raw_metrics -> versioned raw cache -> cached_metrics -> report/report_service -> existing React resources page`

Default cloud collection, Sky resource inventory and both Flow `_id_ == "TrainConfig"` filters remain unchanged. The new provider is explicit, never disguised as Flow. Report `resources` remain independent of `model_runs`. The trainer never writes this cache. Model rendering uses the cached versioned record, including attempts, uncertainties, losses/reward/validation, compact calibration fits, source/config/split hashes, artifacts, measured resources, parallel counters and separate provider status.

`--local-models-only` (collector CLI) or `OVERWATCH_LOCAL_MODELS_ONLY=1` (application environment) explicitly skips cloud collection. It is useful for isolated local testing and does not delete previously cached resource records. It does not make missing/private import dependencies available. Without that opt-in, original provider behavior remains in effect.

## Native application verification

Overwatch requires **Python >=3.13.12,<3.14**. Its private TypeSafe/TSPath package-index environment, SkyPilot/W&B dependencies and actual npm dependency lock must be available. Do not install the training runtime into Overwatch or relax the Python constraint to claim success. The private package-index authentication and pinned native tokenizer/runtime were unavailable to this application verification.

First generate a fresh run in the separate model environment:

```powershell
python scripts/verify_stage3_local.py --out runs/stage3-native-e2e-input
python -m kev_laya register --registry C:\temp\overwatch-stage3\registry.json --snapshot runs/stage3-native-e2e-input/run/telemetry.json --source-id local-proof
```

Then, in Overwatch's **supported** environment and primary checkout:

```powershell
$env:OVERWATCH_RAW_CACHE_ROOT = 'C:\temp\overwatch-stage3\raw-v1'
$env:OVERWATCH_KEV_LAYA_REGISTRY = 'C:\temp\overwatch-stage3\registry.json'
$env:OVERWATCH_LOCAL_MODELS_ONLY = '1'
$env:KEV_LAYA_SNAPSHOT = 'C:\path\to\layev\runs\stage3-native-e2e-input\run\telemetry.json'
uv run python -m overwatch.collector --local-models-only
uv run pytest tests/test_kev_laya_pipeline.py -q
uv run pytest -q
uv run ruff check src/overwatch/providers/kev_laya.py src/overwatch/kev_laya_adapter.py src/overwatch/kev_laya_contract.py src/overwatch/kev_laya_lock.py
npm ci
npx tsc --noEmit
npm test -- --run
npm run build
uv run overwatch --no-open
```

These are reproduction commands, **not reported successes**. Use the existing authorized package-index authentication; never place credentials in a registry or telemetry. The test calls the actual top-level collector, real cache readers, real billing defaults and real report service. It then removes a copied export and installs a negative I/O trap to prove cache-only rendering. It does not replace the initial readers/providers with stubs to obtain a pass. All caches are pytest-local. It skips outside the declared Python version or without a real fresh training/evaluation export. Any skip leaves the native gate open.

In an environment with Playwright and its Chromium browser installed, inspect the real running UI:

```powershell
python integrations/overwatch/verification/inspect_ui.py --url http://127.0.0.1:8765 --run-id resume-proof --out C:\temp\overwatch-stage3\ui-evidence
```

The script reads the actual `/api/report`, verifies the model-run card and parallel counters in the existing React application, captures page errors and saves a screenshot/report. It accepts only loopback URLs and never serves an imitation page. It has not run here. The published baseline reported frontend checks; this session exercised only the TypeScript client/helper and helper runtime assertions, not the full React application.

## Remote transport and deadline boundaries

S3 and W&B are optional and were tested with instrumented doubles, not live publication. W&B collection uses bounded explicit GraphQL pages to enforce pre-read bytes; public-SDK eager iterators cannot provide that contract. The production endpoint/query remains a live acceptance gate. An in-flight SDK call or credential discovery may outlast the cooperative deadline; no hard SDK cancellation claim is made. See `docs/STAGE3_CORRECTNESS.md` and `docs/TELEMETRY.md` in the model project for exact counters, migration, lineage and budget semantics.
