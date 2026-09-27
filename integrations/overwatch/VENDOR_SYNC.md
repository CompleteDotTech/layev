# Overwatch guard payload synchronization

This synchronizes source-level repairs; it is not native model, live remote,
locked Overwatch backend, populated UI, or Jev comparison acceptance.

## Provenance

The bundled adapter, provider and their two focused regression files are copied
without semantic changes from Overwatch commit
`5a7e4418bcebee9c9ceccca4837e04f881d57ecf` on the existing integration PR:
https://github.com/CompleteDotTech/Overwatch/pull/1

At the recorded September 27, 2026 readback that source was SSH-verified and
pushed, but PR #1 was draft and unmerged. Its frontend check passed; its backend
check exited at missing authorized private-index credentials. Source identity
must not be promoted to an accepted deployment. Recheck delivery and checks
before merging any dependent change.

The previous Layev payload at
`6a99afde9df06be616a2ac75fb63e4e933bdfd96` still contained the malformed cache
version defect and the two W&B pagination/response-ownership defects. Published
sources and exact file histories remain available at:

- https://github.com/CompleteDotTech/layev/tree/6a99afde9df06be616a2ac75fb63e4e933bdfd96/integrations/overwatch/added
- https://github.com/CompleteDotTech/Overwatch/blob/5a7e4418bcebee9c9ceccca4837e04f881d57ecf/src/overwatch/kev_laya_adapter.py
- https://github.com/CompleteDotTech/Overwatch/blob/5a7e4418bcebee9c9ceccca4837e04f881d57ecf/src/overwatch/providers/kev_laya.py

The adapter uses `None` for a cold start with no prior cache. An existing empty
object (`{}`) lacks a readable cache version and now produces an explicit
`unsupported_prior_cache_version` warning. Layev's historical cold-start tests
were updated to pass `None`; the malformed-cache regression retains `{}` as a
deliberately invalid prior envelope.

`tests/test_overwatch_vendor_sync.py` freezes the four reviewed Git blob IDs,
using LF canonicalization so Windows CRLF checkouts remain supported. The small
fixture diff reconstructs the exact published predecessors and checks their
SHA-256 values before exercising an upgrade.

## Preserved installer boundary

No change is made to `apply.py`, either historical hash manifest, the legacy
core-file transformations, accepted base revision, or telemetry/cache schemas.
Both exact predecessor hashes were already in `known-added-hashes.json`; the
allowlist is not broadened. Unknown target edits remain refusals. Matching new
files remain idempotent, and unrelated files are not overwritten.

Do not run the base-pinned installer forcibly against the current PR branch.
Its primary-checkout and revision safeguards remain intentional. Review/rebase
against the actual primary checkout and use the existing PR for Overwatch work.
Do not reapply the historical Overwatch cache or W&B patches; those repairs are
already on its integration branch. This change updates the Layev vendor copy.

## Reproduction and evidence limits

From a complete Layev checkout run:

```sh
python -m pytest tests/test_overwatch_vendor_sync.py -q -ra
python -m pytest -q
python -m compileall -q src tests scripts
```

The focused test uses temporary added-file fixtures and the real installer
prepare/transaction functions. It deliberately excludes the eight core-file
transformations and an actual primary-checkout revision. It verifies exact
predecessor upgrades, CRLF, backups, idempotence, unknown/concurrent-edit refusal,
rollback and existing revision/worktree refusal conditions. The fixture's
installed output runs 94 synthetic cache/transport cases in an external process
whose imports are checked to resolve to that output, not the Layev checkout.
These are not 94 additional distinct cases beyond those same two source suites.
No cloud, credentials, model weights, running service or WSL operation is used.

Full supported backend, official SDK, genuine producer-to-visible-UI and live
S3/W&B qualification remain in Overwatch #2–#5 and Layev #6/#10. The original
45-feature matrix, native/CUDA/context/quality gates and Jev parity remain
separate and unverified by this source synchronization.
