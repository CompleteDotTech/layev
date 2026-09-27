# Main branch release policy

Readback date: September 27, 2026. The effective policy is GitHub's live branch
protection, so recheck it before relying on this receipt.

Main requires a pull request and these five successful checks from the GitHub
Actions app, with strict up-to-date status checks:

- `source-integrity-and-critical-lint`
- `model-ubuntu-latest-py3.12`
- `model-ubuntu-latest-py3.13`
- `model-windows-latest-py3.12`
- `model-windows-latest-py3.13`

Protection applies to administrators. It requires signed commits, disallows
force pushes and deletion, and requires zero approving reviews. Ordinary
maintainers cannot bypass the required checks. The live branch readback reported
`protected: true`; the GraphQL branch rule reported
`requiresStatusChecks: true`, `isAdminEnforced: true`,
`requiresCommitSignatures: true`, and exactly the five check contexts above.
The REST required-signatures readback reported `enabled: true`.

The [disposable negative probe](https://github.com/CompleteDotTech/layev/pull/21)
used signed head `4131693914a0a3e2582cf3d45a0e3b0a5244dbe8`, whose GitHub
verification was `verified: true`, `reason: valid`. It deliberately added invalid
Python syntax. In [Actions run 36359784989](https://github.com/CompleteDotTech/layev/actions/runs/36359784989),
the required `source-integrity-and-critical-lint` job concluded `FAILURE`.
GitHub reported the PR's `mergeStateStatus: BLOCKED` while the failing job was
listed among the effective required contexts. The PR was closed without merge;
its `mergedAt` was null. The probe demonstrates rejection of that failing head
through the ordinary merge path. Other model jobs were still running at the
recorded readback and are not claimed as passes or failures here.

Earlier Layev merges, including PRs #1 through #20, predate this policy and
must not be described as protected deliveries. Their passing hosted checks and
verified signatures remain evidence for those revisions only. For new changes,
read back the effective protection and exact PR head, wait for all five checks,
merge normally, and verify the merged revision and post-merge checks.
