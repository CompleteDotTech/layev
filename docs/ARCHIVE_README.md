# Kev–Laya correctness stage 3

Version `0.1.0+stage3` preserves actual parallel questions and adds literal serialization versioning, attempt-proof recovery, pre-I/O collection budgets, calibration exposure lineage, measured provenance and a checked Overwatch v2 delta. **Native Qwen/tokenizer/CUDA/32k/64k and live Overwatch integration remain unverified. General predictive quality and Jev parity are not established.**

Start with [the stage-3 change and reproduction guide](docs/STAGE3_CORRECTNESS.md), [telemetry v2](docs/TELEMETRY.md), and [the primary-checkout integration instructions](integrations/overwatch/README.md). The delivery report records the fixed-budget counterfactual diagnostic and its remaining Noul/Score failures. The original reviewed archives are unchanged.

## Inherited project documentation

The following describes the earlier project baseline. Where its telemetry or checkpoint notes differ, the explicitly versioned stage-3 documents above govern the new implementation. Historical evidence is not a result of this release.

# Kev-Laya 0.1.0+parallel1 — independent research candidate

**Status: implemented and exercised on CPU fixtures; NOT a completed or validated Jev replacement.**
The archive contains executable source, actual tiny trained checkpoints, frozen fixture data, evidence,
and a revision-checked Overwatch integration installer. It does not contain pretrained Qwen weights,
a modified Windows primary checkout, or a successful native 32k/64k evaluation. Read `docs/PARALLEL_QUESTIONS.md` and the delivery `../REPORT.md`
and `../ACCEPTANCE.json` before using any performance or compatibility claim.

## Parallel-question update

Real bounded tensor batches now evaluate question branches in both training and serving.
The complete state executes once, gradients remain connected, and padding/caches are explicitly
accounted for. See [parallel execution documentation](docs/PARALLEL_QUESTIONS.md) for configuration,
strict resume behavior, measured evidence, and all remaining gates. Integration support files are
included under `integrations/overwatch`; no live Overwatch checkout is edited by model commands.

## What runs

One causal-attention backbone, one learned pointer head, one checkpoint format and one serving path.
The state is encoded once per request. Each question gets its own functional KV branch, with no
cross-question or cross-request state. Decisions come directly from option logits: no generated
answer text, parser, ensemble or second model server. Supervised training, PGPS-v1 reward training,
per-type temperature fitting, evaluation and serving are implemented. Full-weight and LoRA parameter
paths share the same model. The included miniature experiment actually trained and saved its head.

The API supports text/object/array state; structured or null instructions and criteria; Choice,
probability-weighted ordinal Score, Noul, mixed questions, full categorical distributions and
question-ID round trips. The public confidence is explicitly **entropy concentration**, not a
probability of correctness. The exact Jev confidence formula is not reproduced.

## Install the independently tested model environment

Use Python 3.12 or 3.13 for this project. The recorded execution used Python 3.13.5 and CPU PyTorch
2.10.0. Keep this environment separate from Overwatch, whose existing requirement remains
**Python >=3.13.12,<3.14**. The upstream Kev baseline has a different environment again.

```sh
cd kev-laya
python -m venv .venv
# POSIX: . .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.lock
python -m pip install --no-deps -e .
python -m pytest -q
```

`requirements-cpu.lock` pins the installed dependency closure used for offline validation, not
arbitrary unrelated sandbox packages. Network access is needed on a machine without those wheels.
Native CUDA/reference dependencies have a separate, explicitly unvalidated requirements file.

## Reproduce the actual miniature experiment

From `kev-laya`, choose a destination that does not already exist:

```sh
python scripts/smoke_experiment.py --out runs/miniature --choice-permutation
```

The script locates the adjacent `overwatch-integration/added/src` modules when run from this bundle.
It freezes train/development/calibration/test partitions and the experiment protocol before looking
at results, trains SFT, trains compute-matched SFT and PGPS extensions, fits temperatures only on the
calibration partition, writes reports/checkpoints/snapshots, registers the runs, and exercises the
new adapter with an isolated fixture cache. It does **not** claim the original Overwatch report
service or React frontend has executed. No cloud jobs, hosted inference or publishing occur.

The marker task is deliberately easy. Perfect fixture accuracy is a software-path check, not evidence
of broad reasoning, real multilingual comprehension, calibration under distribution shift, or parity
with Jev. Inspect `evidence/miniature-run` at the bundle root for the actual retained session results.
The original unaugmented checkpoint exhibited an option-order failure (20/34 flips);
`evidence/permutation-v2` retains the permutation-trained engineering regression (0/34 flips).
The supplied serving command uses that later checkpoint. A subsequent out-of-fixture probe still fails: `color=red; level=1; case=99999` is answered as blue. The case-number/color correlation in the fixture is a potential shortcut; no generalization or calibration claim is justified. These are not independent real benchmarks.
Its recorded absolute artifact URIs describe this session; generating a new run creates valid URIs
on your machine without modifying the archived evidence.

## Serve the included trained fixture

```sh
python -m kev_laya serve --checkpoint ../evidence/permutation-v2/added-pgps/calibrated.pt --no-auth
```

This binds loopback by default. Public binds require API keys, provided through the
`KEV_LAYA_API_KEYS` JSON-array environment variable. Never place real keys in command history or
committed configurations. `--no-auth` refuses a non-loopback host.

```python
from kev_laya.client import KevLayaClient

with KevLayaClient(base_url="http://127.0.0.1:8009") as client:
    response = client.system_one(
        state="color=red; level=1; case=99999",
        questions={
            "color": {"type": "choice", "instructions": "color?",
                      "criteria": {"red": None, "blue": None}},
            "level": {"type": "score", "instructions": "level?",
                      "criteria": ["0", "1", "2"]},
        },
    )
    print(response.model, response.answers, response.usage)
```

The example above is a retained **failing semantic probe**, not a promise of a correct color answer. It exercises the interface; see `evidence/installed-http-verification/report.json` at the bundle root.

The included checkpoint has a **512-token branch budget with the fixture byte tokenizer**, not native
32k support. Its immutable response identity is derived from the actual checkpoint hash. Preview is
the default alias; `--stable-alias` is an explicit operator promotion, not a scientific certification.
TypeSafe SDK base-URL/model workflows have a runnable compatibility test but were **not executed**
because that SDK was unavailable. The project's own Python client and TypeScript types were checked.

## Individual reproducible stages

```sh
python -m kev_laya freeze-smoke --out runs/data
python -m kev_laya train --suite runs/data --config configs/smoke.json --out runs/sft
python -m kev_laya evaluate --checkpoint runs/sft/checkpoint-000160.pt --suite runs/data --split test --out runs/sft-test.json
python -m kev_laya calibrate --checkpoint runs/sft/checkpoint-000160.pt --suite runs/data --out runs/sft-calibrated.pt
```

Evaluation and calibration create independent `.telemetry.json` snapshots. Serving can publish a
bounded aggregate snapshot with `--telemetry runs/serve.json --run-id local-serving-01`. No request
or response text is supplied to those hooks. All stages refuse conflicting output destinations.

For interrupted training use the **same total-step configuration**, `--stop-after N`, and then
`--resume path/to/checkpoint-N.pt` against the same output directory and frozen suite. Do not change
learning-rate scheduling, accumulation, architecture or the dataset manifest on resume.

## Native candidate and remaining mandatory gates

The selected candidate is Qwen/Qwen2.5-0.5B at
`060db6499f32faf8b98477b0a26969ef7d8b9987`: an all-attention, non-recurrent architecture. Its source/config
was inspected; its actual weights were **not** downloaded or executed here. Model-card context size
alone is not acceptance evidence. No long-context training has occurred.

```sh
# Explicit free public-weight acquisition; not a paid job or hosted evaluation.
python -m pip install -r requirements-native.in
python scripts/download_backbone.py --out backbone/qwen-0.5b
python -m kev_laya native-init --backbone-dir backbone/qwen-0.5b --out runs/native-init.pt --lora-rank 8 --checkpointing
# licensed-long-suite must already contain frozen, group-disjoint labeled partitions.
python -m kev_laya train --init runs/native-init.pt --suite licensed-long-suite --config configs/qwen-sft.json --out runs/native-sft --device cuda
python -m kev_laya reward-train --init runs/native-sft/checkpoint-001000.pt --suite licensed-long-suite --config configs/qwen-reward.json --out runs/native-pgps --device cuda
python scripts/validate_native.py --checkpoint runs/native-pgps/checkpoint-000500.pt --out runs/native-context-validation --device cuda
```

Those native commands are supplied for reproduction but have **not passed in this environment**.
The harness requires an actually pretrained, trained checkpoint, independently compares the backbone
to the pinned Transformers implementation, constructs complete serialized 32,768/65,536-token
requests, retains evidence at beginning/middle/end, measures GPU memory/latency, tests overflow and
checks recorded training-length exposure. It will not turn a small-model result into a native pass.
It is itself awaiting its first native execution. A licensed real long-context dataset, sufficient
training, real-language benchmarks and controlled Kev/Laya baselines remain necessary. It is not
appropriate to deploy this fixture for consequential decisions.

## Overwatch

See `../overwatch-integration/README.md`. The installer checks the exact inspected base revision and
all touched Git blob hashes, refuses changes to dirty target files, permits unrelated dirty files,
and creates rollback backups. It edits the existing primary checkout only when explicitly run.
No worktree, competing dashboard or model-runtime dependency is introduced into Overwatch.

## Contracts and evidence

`docs/compatibility-2026-09-25.json` is the frozen compatibility decision record.
`docs/ARCHITECTURE.md`, `LEARNING.md`, `TELEMETRY.md`, `CHECKPOINTS.md`, `VALIDATION.md` and
`OPERATIONS.md` document implementation and limitations. JSON schemas are generated into `schemas/`.
`NOTICE` and `licenses/` preserve upstream attribution and byte-verified license texts.
