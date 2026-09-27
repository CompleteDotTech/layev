# Release readiness — intentionally incomplete

The source-hardening change is **merged**, not a completed Jev replacement or
an approved model deployment. Public name Layev and package/import/CLI names
`kev-laya`/`kev_laya` remain unchanged. A source merge is not a package release,
trained-model release or native acceptance result. This readback is dated
September 27, 2026; later changes require fresh verification.

## Current branch policy

Main protection was enabled later on September 27, 2026. It now requires pull
requests, the five stable checks, signed commits and administrator enforcement.
The [policy and failing-check receipt](RELEASE_POLICY.md) records the live
readback and the closed, unmerged negative probe. Historical `protected=false`
and HTTP 403 statements below describe earlier revisions and credential contexts;
they are not the current policy state.

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

[Layev PR #1](https://github.com/CompleteDotTech/layev/pull/1) merged on
September 26, 2026 into `080aa7183bb5d8f6576bde9d41e3c5115c7ab132`.
[Hosted run 36224185965](https://github.com/CompleteDotTech/layev/actions/runs/36224185965)
reports all five source/model jobs completed successfully:
`source-integrity-and-critical-lint`, `model-ubuntu-latest-py3.12`,
`model-ubuntu-latest-py3.13`, `model-windows-latest-py3.12`, and
`model-windows-latest-py3.13`. This is evidence for that existing revision,
not for subsequent unpublished source patches or pinned Qwen weights.

The existing merge's GitHub verification is `verified=true`, `reason=valid`,
with a GitHub PGP merge signature. It is not a new SSH-signed execution-session
commit. The branch listing reports `protected=false`; the connected integration's
branch-protection read returned HTTP 403, `Resource not accessible by integration`.
No rule change or failing-required-check merge-block experiment was performed.
[Layev #8](https://github.com/CompleteDotTech/layev/issues/8) remains open.
The owner's repository role does not establish this integration's admin API access.

[Layev PR #11](https://github.com/CompleteDotTech/layev/pull/11) also merged on
September 27, 2026 at 08:18:33 UTC. Main is now
`b4cb9df97be61729ebd68b8e0f5ee9bb2d193c73`; its five existing check names all
passed in [run 36305789586](https://github.com/CompleteDotTech/layev/actions/runs/36305789586).
The package commit `6ad6ee8770eaebf9e68f2ee30fd5c33bf328e145` has a verified SSH
signature; the merge has a verified GitHub PGP signature. That PR delivers the
TypeScript package portion, not all of
[interface issue #6](https://github.com/CompleteDotTech/layev/issues/6).
The issue records an eight-step calibrated CPU byte-tokenizer fixture served over
actual loopback HTTP, with two packaged-client model/request checks. These are
reported workstation interface results, not native Qwen or official SDK proof.
The issue also reports clean primary main and divergence 0/0. This Windows
checkout independently confirmed main at that merge, clean before this
documentation edit, and 0/0 against origin/main. The final branch listing
still reports `protected=false`.

[Overwatch PR #1](https://github.com/CompleteDotTech/Overwatch/pull/1) remains draft
and unmerged, now at `ed825bf1c2c4c19eb3de99ad46564c5803bb4a39` on
`feat/layev-model-runs`. Its added cache-guard and CI commits both have verified
SSH signatures. Overwatch main remains
`501bd99a0cb0c6b321feb022f5747d6dcbd5e9c4`.
[Hosted run 36305664862](https://github.com/CompleteDotTech/Overwatch/actions/runs/36305664862)
has two completed jobs: `frontend-typecheck-vitest-build` succeeded, while
`backend-lint-and-locked-tests` failed. The backend log records
`BLOCKED: missing_authorized_tscore_index_token` and exit code 2, before the
locked backend suite. The PR records focused 44/44 cache/CI tests under isolated
Python 3.13.12, Ruff, and frontend checks on the primary workstation; those
reported results do not replace the full authenticated locked backend or
populated UI gate. Do not reapply the old cache/CI patches already on this branch.
Use the existing PR, not a second integration PR. Tracking:
[backend/CI #2](https://github.com/CompleteDotTech/Overwatch/issues/2),
[fresh visible pipeline #3](https://github.com/CompleteDotTech/Overwatch/issues/3),
[cache boundary #4](https://github.com/CompleteDotTech/Overwatch/issues/4), and
[live transports #5](https://github.com/CompleteDotTech/Overwatch/issues/5).

These later remote readbacks supersede earlier reports of the old heads and zero
Overwatch checks. The package merge and two Overwatch commits predate this
documentation update. This Windows workstation independently confirmed the
Layev main and Overwatch draft branch heads above, clean checkout states before
this edit, and divergence 0/0 against their respective origin branches. It has
not verified the running Overwatch service checkout, populated report, or UI.
The archive author's isolated session could not inspect either primary checkout;
its access limitation is historical, not a limitation of this readback. No WSL
process, cloud CUDA environment, paid Jev call, or live telemetry run was changed
for this documentation update.

Configure review/signature/required-check policy only through authorized
administration; never use a bypass merge. Run the repository-required checks on
each candidate revision and verify hosted results before accepted merge. Then
fetch and safely synchronize the actual primary checkout without discarding
unrelated work; verify signatures, remote merged state and divergence `0 0`.
For Overwatch, verify the actual service's checkout and populated report/UI too.

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
