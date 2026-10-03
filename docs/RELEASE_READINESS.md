# Release readiness — intentionally incomplete

## Actual full-weight FP32 step: partial evidence (October 2, 2026)

[One actual optimizer step and a separate exact stock comparison](evidence/optimizer/fullweight-fp32-step1-partial-20261002.md) are verified. The original oracle left its comparison memory floor unchecked; complete resource compliance, resume and BF16 remain unproved. Native derivative acceptance remains incomplete.

## Trained rank0 derivative readout (October 2, 2026)

[Unfavorable derivative evidence](evidence/native/rank0-derivatives-failed-20261002.md) records one passing comparison out of five. Both HF comparisons and native shared-prefix/full gradient comparisons fail frozen tolerances. This independent derivative failure establishes no CPU-state optimizer arithmetic result; native acceptance remains incomplete.

## Explicit CPU-state optimizer: partial qualification (September 30, 2026)

The opt-in backend at signed source `b5d8299b768564c09c541e3dfc04b3d28eef2fd6`
passed the [tiny rank0 trainer qualification](evidence/optimizer/tiny-cuda-offload-v4-20260930.json)
and a separate [tiny LoRA rank4/activation-checkpointing qualification](evidence/optimizer/tiny-cuda-offload-lora-checkpoint-v2-20260930.json).
Each completed 8/8 FP32/BF16 uninterrupted/save/exit/fresh-resume/reload phases,
with zero skips and exact state comparisons. All 33 source hashes and HEAD
were unchanged at each phase start/end. The LoRA run reused the integrated v4
8-pass primitive receipt; it did not rerun those tests. This is partial delivery
using synthetic byte-tokenizer fixtures. Actual pretrained fullweight optimizer
qualification and the captured-gradient stock oracle remain pending under their
separate 16GiB host gate. Default OOM receipts and issues #3/#4 remain unchanged.

## Observed synthetic context decisions (September 30, 2026)

The [development diagnostic](evidence/context/development-factorial96-20260930.md)
and [portable receipt](evidence/context/development-factorial96-20260930.json)
record 96 short/medium generated cases on the fixed separately calibrated v8
checkpoint. Choice scores 79/96, Score 90/96 and Noul 48/96. All Noul predictions
are true; all six Score errors have gold zero. Choice first-ranked gold scores
18/32, compared with middle 30/32 and last 31/32. Label/option/target readback
finds no mismatch. Correlated generated targets, marker wording, rubrics and
distractors limit causal interpretation; false and zero labels existed in prior
training. Positive temperature calibration cannot change argmax.

The original v8 native-context-v3 marker result is 3/18 against the unchanged
80% threshold. The earlier v6 result of 6/18 remains a separate historical
failure. The short/medium development scores establish neither a passing
32768/65536 gate nor representative deployment quality or Jev benefit.
The separately frozen paired144 attempt subsequently timed out with 96/144
durable updates; see the [timeout receipt](evidence/context/paired144-timeout-20260930.md).
No completed repair, calibration or readout result is available. A new run
initialized from weights resets current-run exposure; resuming a run preserves
its exposure. A later fixed exact-limit continuation is required before final
context acceptance. Existing numerical receipts and
measured rank-0 resume/resource failures remain unchanged; issues #3/#4 stay open.

## Verified numerical/source delivery (September 29, 2026)

[PR #56](https://github.com/Jev-Engineering/layev/pull/56) merged at
`4f31cc76e38ef9848950e1667875571f8e190903`, from verified SSH-signed source
`4ffd9f1f1263d225e4f130f174cf6070c635d235`. All five required
[PR checks](https://github.com/Jev-Engineering/layev/actions/runs/36656915788)
and all five [post-merge checks](https://github.com/Jev-Engineering/layev/actions/runs/36657405767)
passed. The primary checkout synchronized cleanly with main. The CUDA graph
ownership regression passed again on the merged checkout (one pass, 80 deselected).

This completes delivery of the canonical FP32 numerical and BF16 graph lifetime
repairs, public reproducers, and scoped receipts. Historical numerical and rank-8
receipts retain their precommit source identities; the graph repair records its
later training source separately. Complete rank-0 resume and exact-limit FP32
training still fail on the measured configuration. Issues #3/#4 remain open;
representative quality, Jev benefit and remote publication remain unverified.
Overwatch remains archived by user instruction.

## Full-weight mechanics and graph lifetime supplement (September 29, 2026)

The [graph lifetime repair and measured rank-0 limits](evidence/native/graph-lifetime-fullweight-limits-20260929.md)
permit the same-init BF16 first step and reload, with a passing CUDA ownership
regression and failing old-source control. Rank-0 FP32/BF16 resume backward still
exhausts the unchanged allocator cap on this configuration. These results do not
close #3/#4 or alter the source identity of earlier numerical/rank-8 receipts.

## FP32 candidate numerical readback (September 29, 2026)

The [canonical FP32 repair](evidence/native/short-fp32-canonical-v10-20260929.md)
passes the short independent HF adapter oracle, four generated cached/batched/full
cases, 8191/8192 state-prefix boundaries, and a 10520-token state numerical gate.
This is candidate source evidence; the exact-limit FP32 training resource probe exhausted the fixed allocator cap
during forward, leaving backward unverified. Delivery is recorded above; these receipts retain their original source identities. Issues #3/#4 remain open; representative quality and Jev benefit
are unknown. Overwatch remains archived by user instruction. Older readbacks
below are historical and do not override these scoped receipts.

## Current delivery readback (September 28, 2026)

Layev main is `a6926f7b4585e840b36f1d9157c3af3907210ef5`, after protected
[PR #33](https://github.com/CompleteDotTech/layev/pull/33). Its signed head
passed all five required checks; the merge is verified, and the primary checkout
was clean and synchronized at divergence 0/0. The exact-merge main run
`36384521208` was queued at this readback. Earlier [PR #34](https://github.com/CompleteDotTech/layev/pull/34)
also passed all five PR checks and merged at `821ffeaeb7914755969c64c6d7aee56cc1ce203e`,
but its post-merge run `36383174403` was canceled by the newer main push before
all five jobs finished. Do not count that canceled run as a post-merge pass.
Main still has [enforced protection](RELEASE_POLICY.md).

The pinned Qwen weights and literal tokenizer were verified, and a two-step
trained LoRA checkpoint was reloaded and served locally. Its corrected
independent Transformers comparison passed the unchanged `1e-4` combined
hidden-state tolerance at 62 tokens; the earlier failed probe remains in
[native evidence](NATIVE_SHORT_PROBE.md). Strict actual-weight FP32 and BF16
cached/batched/full-row numerical comparisons still fail. The exact 32,768/65,536
context request and overflow negatives were constructed with the pinned
tokenizer, but trained and calibrated long-context execution has not passed.
No licensed representative multi-seed quality, repaired case-99999 result, or
live Jev comparison is established. PR #33 adds an offline scorer only; PR #34
adds seed-variation reporting only.

[Issue #8](https://github.com/CompleteDotTech/layev/issues/8) and the 45-feature
mapping delivery are closed with their separate evidence boundaries. Overwatch
[PR #1](https://github.com/CompleteDotTech/Overwatch/pull/1) remains draft at
`aef193862499734af552860744db667b64afebd1`: frontend checks pass, while
locked backend CI stops before installation because its `overwatch-ci`
environment lacks the required CodeArtifact token. Live W&B/S3 targets and
approved data transfer are also missing. These source and interface deliveries
are not a completed Jev replacement or approved model deployment. Older
revision and policy statements below are historical readbacks.

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

The original companion matrix is now preserved byte-for-byte as
[FEATURE_MATRIX.original.json](research/FEATURE_MATRIX.original.json), with its
[45-feature Layev evidence mapping](research/layev-feature-mapping.json). The
historical Linux handoff lacked the original; that access limitation is retained
as history. Current source/test inventory is recorded separately from native
validation, representative multi-seed quality, pinned Kev/Laya baselines and
Jev-relative quality/cost/latency, which remain open.

The historical stage-three CPU result was 235 passed/5 skipped. Its synthetic
calibrated reward arm scored 83.33% on 270 questions, but its separate known
regression scored 42.59% on 108. The earlier smoke scored 50% against the retained
70% threshold. These failures and the known red/level-1/case-99999 regression must
remain visible. They are not results from this source checkout or general-quality
proof. There was no authorized Jev call, protected-data upload or paid workload.
