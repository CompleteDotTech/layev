# Release readiness — intentionally incomplete

This is an **unreleased source-hardening change**, not a completed Jev replacement
or an approved model deployment. Public name Layev and package/import/CLI names
`kev-laya`/`kev_laya` remain unchanged. The existing package version and lock are
not changed by this overlay; no package release was published.

## Changes and boundaries

The guarded Overwatch payload now reverses both historical component import
spellings only when the original base hash is recovered exactly. The obsolete
`ModelRuns.tsx` is deleted only when its normalized SHA-256 matches a known v1 or
stage-three payload. Modified obsolete files, unknown case variants, symlinks,
junctions, worktrees and changed base revisions are refused. A backup includes
deleted files. Transaction preimages are checked again before/during application;
rollback is tested. Concurrent editors/installers still require operator control;
this is not a filesystem-wide transaction lock.

`known-added-hashes.json` cites the historical archive SHA-256 and inspected public
revision. Linux fixture transactions cover fresh, v1 and stage-three paths, LF and
CRLF. A later primary-Windows preflight and one-file installer update passed;
eight Overwatch frontend tests and its production build passed. Full backend and
populated UI verification remain open.
The original archives and their evidence remain unchanged.

The numerical audit now asserts probabilities as well as recording them, retaining
`atol=rtol=1e-5` for logits/probabilities and `2e-5` for gradients. Parallel kernels,
cache policies and budget defaults are unchanged.

Evaluation records source groups. Group-percentile bootstrap intervals are
available for accuracy, NLL, Brier, ECE and ordinal error through
`scripts/summarize_uncertainty.py`. Historical records without explicit groups
remain unknown. These intervals cover the supplied groups, not model-seed or
deployment variation, and do not repair leakage. Calibration resets an unsampled
type to the same unfitted `1.0` temperature recorded in its receipt instead of
silently retaining an earlier fit.

## Publication and verification status

The Linux handoff sandbox lacked Git push/administration permission, signing,
and access to the primary Windows checkouts. Those statements describe that
handoff, not this checkout. Here the source patch was applied in the primary
Layev checkout, the guarded installer updated the primary Overwatch checkout,
and local Windows software/frontend checks passed. Hosted CI, signed PR/merge
state and live UI readback require separate readback before any completion claim.

After actual CI execution, required checks must include
`source-integrity-and-critical-lint` and all four `model-<os>-py<version>` jobs.
Configure normal review/signature/check requirements through authorized repository
administration; never use an admin/bypass merge to skip them. Then verify signatures,
remote merged state, clean primary checkouts and divergence `0 0` before calling
the Git lifecycle complete. The integration must first be reconciled with the
actual primary Overwatch revision and its unrelated local changes.

## Missing acceptance sources and quality

The original companion `overwatch-model-research/FEATURE_MATRIX.json` and
independent review/probe files exist in the primary workspace, although the
Linux handoff could not retrieve them. They have not yet been reconciled into
this repository. Prompt-derived acceptance files are not that matrix. No
replacement set of 45 labels is invented here. The mapping, native
validation, representative multi-seed quality, pinned Kev/Laya baselines and
Jev-relative quality/cost/latency remain open.

The historical stage-three CPU result was 235 passed/5 skipped. Its synthetic
calibrated reward arm scored 83.33% on 270 questions, but its separate known
regression scored 42.59% on 108. The earlier smoke scored 50% against the retained
70% threshold. These failures and the known red/level-1/case-99999 regression must
remain visible. They are not results from this source checkout or general-quality
proof. There was no authorized Jev call, protected-data upload or paid workload.
