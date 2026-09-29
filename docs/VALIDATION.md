# Validation and reproduction

## Source-hardening exercise — Linux handoff and Windows checkout

The base is public `CompleteDotTech/layev` revision
`480005c722027a254f1129bd83dbc09bf9f0a496`. Source was recovered from the reviewed
archive plus GitHub blob/tree checks in the Linux handoff. The primary Windows
checkout has the existing `uv.lock`, which the patch leaves unchanged. The
Windows results below were rerun against that real checkout. See
[environment details](ENVIRONMENTS.md).

| Gate | Current evidence |
|---|---|
| Full model/source suite | Linux handoff: 280 passed, 5 skipped; Windows primary checkout: 280 passed, 5 skipped |
| Numerical audit | All 24 CPU FP32 fixture configurations pass unchanged tolerances |
| Exact resume | Same-policy interrupted/resumed parameters equal the uninterrupted control bit-for-bit |
| Actual question scheduling | 16 questions: one state-prefix call and one 16-row suffix call |
| Calibration/export | Fresh trained fixture, calibration, checkpoint reload and verified exposure; inference-only calibrated artifact |
| Local monitoring | Actual local-file provider and adapter recover missed attempt 1, read persisted cache, show verified attempt 2 after input export deletion |
| Packaging | Windows wheel and source distribution built; 25 runtime modules and four license/notice files match source |
| Installed wheel | Windows wheel install and isolated import/CLI smoke passed; handoff recorded 12 real loopback requests, nine multi-question, exactly matching direct responses |
| TypeScript | Client/helper typechecks and nine helper runtime assertions; not full React/Vitest/UI |
| Source integrity | Original 359-file archive manifest verified; source/index/license/contract checks and relative links checked |
| Remote CI/protection/Git completion | CI prepared; hosted execution and merge require separate readback |

The Linux handoff's FP32 maximum absolute differences were
`1.7881393432617188e-7` in logits, `5.960464477539063e-8` in probabilities
and `2.1042069420218468e-8` in gradients. The fresh Windows audit measured
`1.7881393432617188e-7`, `8.940696716308594e-8` and
`2.9802322387695312e-8`, respectively.
The checks retain `atol=rtol=1e-5` for logits/probabilities and `2e-5` for gradients.
No execution budgets, serial-fallback behavior or kernel tolerances were relaxed.

The Windows run skips the official TypeSafe SDK, CUDA FP32, CUDA BF16, Windows
symlink creation without privilege, and the pinned Qwen tokenizer because its
verified snapshot is absent. The real Rust byte-level tokenizer test passed with
the pinned `tokenizers` runtime. This test does not establish pinned Qwen behavior.
Five formerly skipped archive-installer tests now run from exact retained source.

The Linux handoff first failed an isolated wheel CLI smoke because that interpreter
could not import Torch; a corrected environment passed. In the Windows checkout,
`uv sync --locked --extra dev --extra native` resolved and installed the declared
dependencies, `uv build --no-build-isolation` built both distributions, and an
installed-wheel import/CLI smoke passed. This is a local checkout check, not yet
an independent clean-clone or hosted CI result.

## Reproduce without replacing reviewed evidence

From a source clone in the model environment:

```sh
uv run --no-sync python -m pytest -q -rs
uv run --no-sync python -m compileall -q src tests scripts integrations
uv run --no-sync python scripts/check_source.py
uv run --no-sync python scripts/verify_parallel_numerics.py --out verification-output/numerics.json
uv run --no-sync python scripts/verify_stage3_local.py --out verification-output/software
uv run --no-sync python scripts/verify_parallel_http.py --checkpoint verification-output/software/run/calibrated.pt --out verification-output/http
uv run --no-sync python scripts/summarize_uncertainty.py --evaluation verification-output/software/software-evaluation.json --out verification-output/uncertainty.json
```

New evidence destinations must not already exist. `check_source.py` checks actual
Git-index bytes; use `--snapshot` only for an explicitly identified source export.
The software exercise uses generated miniature data. Its evaluation is a
**development fixture**, not an untouched final quality experiment.

## Historical evidence is not a current result

The stage-three archive SHA-256 is
`f5eab7178cde0f8bbb8e7d13b98d8ff17ff203b88e5387b337a8e5d25e0bad20`.
It contains the historical 235-pass/5-skip Linux CPU result, a synthetic calibrated
reward-arm result of 83.33% on 270 questions and a known-regression result of
42.59% on 108 questions. The earlier short smoke failed with 50% against 70%.
The original archives, checkpoints, failures and acceptance/report bytes are
preserved outside this source overlay. No historical ZIP or weights are added
to Git to repair documentation links.

## Still-open native, quality and integration gates

The pinned Qwen weight and tokenizer bytes, pretrained initialization/reload,
short independent Transformers oracle, CUDA FP32/BF16 LoRA and short full-weight
training, and source-backed BF16 LoRA/activation-checkpointed training at exact
32,768-branch/65,536-aggregate limits have now been exercised on the local RTX
3060. See the [native evidence readout](FP32_PROJECTION_PARITY.md) for hashes,
resource measurements and the recovery history. Both six-stratum full-context
FP32 and BF16 validators passed numerical and structured-overflow checks but
**failed overall** on their frozen diagnostic marker threshold (1/18 and 0/18
correct) on the earlier checkpoint. A later generated marker curriculum on
current main completed 120 short and two exact-limit BF16 training steps,
separate calibration and another six-case BF16 validator. Numerical and
overflow checks again passed 6/6, but the overall diagnostic still failed at
2/18 against the unchanged 80% minimum. Its generated development score was
50/72 and does not establish representative quality. Two later, separately
frozen generated extensions on clean main preserved exact-limit training and
separate calibration. The 24-step long extension's BF16 validator again passed
6/6 numerical and overflow checks but failed at 2/18 marker decisions. A
further 160-step, 255-option short curriculum followed by two exact-limit
long steps reached 6/18 on the same fixed six-case validator, still below 80%.
Its separate generated development split scored 52/72; development splits
across these runs differ and are not paired. The fixed validator had already
been observed during design, so none of these readouts is untouched quality.
The [native evidence readout](FP32_PROJECTION_PARITY.md) records the artifact
and report hashes. A fresh actual-weight
short mixed FP32 forward/backward comparison passed one generated request. A
subsequent focused short CUDA BF16 attention change on signed source `9406026`
made batched, serial cached and full-row logits/probabilities match exactly on
two generated actual-weight mixed requests (3 and 4 questions). The same
source still **fails** BF16 gradient parity across all 100 trainable tensors.
The unchanged six-stratum BF16 long validator on the calibrated v6 artifact
passed 6/6 numerical and overflow checks with zero maximum cached/full error
and the pinned short FP32 Transformers oracle, while still failing its overall
marker gate at 6/18. Its source-bound private report SHA-256 is
`3bba0d6d9f75c2c42ec03976438d8bd7e88e912faaa6e93ecb580ccc9e55c62e`.
Long full-weight backward, general native precision/shape/gradient acceptance,
calibrated representative long-context behavior and quality remain unproved.
CPU fixtures and configuration constants cannot fill these gaps.

Representative licensed, group-disjoint datasets and a protocol fixed before
training are still required for the supervised/reward/calibration multi-seed
comparison. Group-aware intervals now exist, but no representative quality or
pinned Kev/Laya baseline run was executed. Jev version/quality/latency/billed-cost
comparison requires authorized access, approved inputs and budget; it is unknown.

The original 45-feature research matrix was unavailable to the historical Linux
handoff. It is now preserved byte-for-byte in [the repository](research/FEATURE_MATRIX.original.json),
with a [per-feature evidence mapping](research/layev-feature-mapping.json) at the
inspected Layev revision. Source and test paths are inventory evidence; they do
not establish passing current tests, native execution, representative quality,
or Jev comparison. The [acceptance status](acceptance-status.json) retains its
historical workstream and failure record.

The [Overwatch guide](../integrations/overwatch/README.md) retains its primary-
checkout-only workflow and separate Python/private-index environment. The
hardened installer was applied to the primary Windows checkout after a passing
preflight; its frontend passed eight Vitest tests and the production build.
The full backend suite remains blocked by private-index authentication. The
Overwatch repository is archived by user choice, so its draft integration PR
and issues are held. No populated UI screenshot or browser-error record is
claimed here.
The local provider/adapter exercise is not the full collector/report/UI pipeline.
