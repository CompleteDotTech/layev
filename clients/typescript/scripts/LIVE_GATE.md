# Versioned loopback interface gate

Related acceptance: `CompleteDotTech/layev#6`. This is a validation-tool repair,
not a native-model, official-SDK, trained-context, quality or Jev acceptance pass.
The runtime client API and its `layev-client/1` contract are unchanged.

## What was corrected

The earlier verifier accepted mere presence of several objects. An envelope with
empty usage/probabilities, missing decision values and an arbitrary confidence
identity could produce `status: passed`. It also required the response model to
match the requested alias, although `InferenceRuntime` advertises aliases and
returns the concrete `resolved_model` from the same model inventory.

The receipt now includes `gate_version: live-loopback-interface/2`. The original
`gate`, client contract identity, request payload, model argument, content hashes,
exclusive receipt creation and four false native/quality/SDK/Jev flags remain.
The request still includes `color=red; level=1; case=99999` and decomposed Unicode.
No answer is scored against that example and no regression-quality result changes.

The gate checks exact answer/question coverage, the current strict response
fields, dynamic choice keys, finite bounded probabilities and confidences,
probability normalization (absolute tolerance 1e-6), score level keys/string
legend values, finite in-range scores/Noul, positive integer usage for this
nonempty probe, one positive branch count per question and zero generated tokens.
The confidence-definition identity must match `entropy-concentration-v1`; a
calibration-status string must be present. Model inventories require uniquely
named metadata rows. An advertised
alias may resolve only to a concrete ID also in that same inventory. Minimal
inventories without the Layev `resolved_model` extension require exact identity.
Unknown/duplicate resolutions fail before the decision request. Requests still
use the caller's explicit model, never a substituted alias or external endpoint.

This helper is scoped to this probe's string-valued score levels. It intentionally
does not recreate Python canonical JSON rendering for arbitrary structured score
criteria. The unchanged transport suite separately covers structured request
round trips. Response probabilities and usage are server-reported values; this
helper does not independently tokenize a response, measure model execution,
recompute confidence/expected-score formulas, prove calibration, or authenticate
model provenance. The 1e-6 wire normalization check does not modify any native
oracle tolerance, the 0.80 diagnostic threshold or the retained 0.70 fixture gate.

## Tests and real service separation

From `clients/typescript` in a reviewed full checkout:

```sh
npm ci --ignore-scripts
npm run typecheck
npm test
npm run test:installed
```

`npm test` retains the original transport suite and adds CLI subprocess tests.
Those subprocesses use the unchanged compiled client over real loopback TCP,
but ALL model responses in the tests are synthetic. They test the verifier, not
an installed Layev service. No SDK, cloud target, paid account or native model is
called. Source-module checks are not an installed-tarball or full-repository pass.

With a separately authorized running loopback Layev service, explicit
`LAYEV_BASE_URL` and `LAYEV_MODEL`, and the existing secret-safe `LAYEV_API_KEY`
route when authentication is needed, the actual gate remains:

```sh
npm run verify:live -- path-to-a-new-interface-receipt.json
```

Missing configuration or non-loopback destinations exit 2 without requests.
Invalid responses/authentication failures exit 1. Successful interface validation
exits 0 and still sets all four higher-level acceptance flags false. Errors use
fixed local codes and do not echo credentials, endpoint URLs or response bodies.
An existing receipt is not overwritten. A `passed` receipt is NEVER proof of
native weights, representative quality, official TypeSafe SDK behavior or Jev
parity; those remain separate open acceptance gates.
