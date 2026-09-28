# Environments and verification boundaries

## Current readback (September 28, 2026)

Main at `a6926f7b4585e840b36f1d9157c3af3907210ef5` has [enforced main
protection](RELEASE_POLICY.md). All five PR-head checks passed for its merge;
the exact-merge post-merge run `36384521208` was queued at this readback.
The public source environment remains separate from an independently created
local CUDA environment using Python 3.13.12, PyTorch 2.10.0+cu128,
Transformers 4.57.1 and tokenizers 0.22.1. That environment verified actual
pinned Qwen weight and tokenizer bytes and can see an RTX 3060. The initial
short hidden-state probe failed, then a corrected independent oracle passed
the unchanged `1e-4` combined tolerance on the trained two-step checkpoint.
Strict FP32/BF16 parallel numerics, long-context native training and
representative quality remain open. The supported Overwatch `tscore` index URL
is present, but its hosted `overwatch-ci` environment has no authorization
token secret; its locked backend install and tests remain unrun. Historical
environment statements below describe their earlier source-hardening scope.

The supported model range remains **Python >=3.12,<3.14**. The separate Overwatch
range remains **>=3.13.12,<3.14**. A Python 3.13.5 model run is not a supported
Overwatch backend run. No dependency requirement was lowered in this change.

## Source-clone environment

The existing `uv.lock` belongs to the public source revision. It is retained
unchanged, not regenerated from a different platform or represented as a new
validated Windows/CUDA environment. From a clean clone, use a separate environment:

```sh
python -m pip install uv==0.10.0
uv sync --locked --extra dev --extra native
uv run --no-sync python -m pytest -q -rs
uv run --no-sync python scripts/check_source.py
```

`--locked` fails rather than silently resolving a different environment. These
commands require accessible public package indexes and suitable platform wheels.
The CI matrix exercises Windows/Linux and Python 3.12/3.13. Its current hosted
execution and required-check protection are documented in the readback above.

`requirements-cpu.lock` is the unchanged historical Linux CPU closure, not a
cross-platform wheel-hash lock. The local source-hardening exercise used the
available Python 3.13.5/PyTorch 2.10.0+cpu environment with inherited installed
dependencies. That is not a successful dependency installation from a clean clone.
The standalone source recovered for this session did not materialize `uv.lock`;
the delivered patch intentionally leaves the repository's actual lock untouched.

## CI coverage

The workflow checks source/index bytes, licenses, matching telemetry contract
copies, local Markdown links, Python syntax and trailing-whitespace formatting.
Ruff is pinned at 0.13.2 and selects critical errors (`E9,F63,F7,F82`). This is not
a whole-tree Ruff-format or full-style-lint claim. Whole-tree formatting/style
remains an open gate until the normal environment can run those tools and review
its changes. The original checkout had no formatter configuration to relax.

The model matrix requires the actual pinned Rust tokenizer package. Missing
CUDA, pinned Qwen files and the official SDK remain explicit skips; a successful
CPU job never certifies those gates. Jobs also run numerical/resume checks, build
and inspect a wheel, install it, and exercise real loopback HTTP outside the
source tree. Source distribution contents include tests, scripts, schemas and
integration support without model weights.

## Native and application prerequisites

Run `scripts/check_prerequisites.py` to record presence/absence without downloads,
provider calls or paid work. Exit 2 denotes an open prerequisite gate, not a pass.
Native Qwen files and their pinned hashes, actual trained/calibrated artifacts,
CUDA FP32/BF16 hardware and licensed representative data are still required.
TypeSafe/TSPath private-index authentication belongs to the existing authorized
Overwatch environment and must not be copied into source or telemetry.
