# Stage-3 checkpoint additions

The checkpoint tensor format remains `kev-laya-checkpoint/1`; additive preprocessing, source provenance and immutable training-exposure records are versioned independently. Calibration remains inference-only and now binds a verifiable sibling/explicit-parent hash chain. Legacy missing exposure is unknown. New native tokenization uses explicit v2; recorded legacy v1 is never silently reinterpreted. Exact CPU continuation is supported only for the same current execution/preprocessing/data/optimization configuration; legacy optimizer saves lacking the new evidence require explicit new-run initialization. See [STAGE3_CORRECTNESS.md](STAGE3_CORRECTNESS.md).

## Historical checkpoint documentation

# Checkpoints and resume

Format `kev-laya-checkpoint/1` contains the complete backbone and pointer state, LoRA parameters and
configuration where enabled, tokenizer configuration/checksum, positional encoding, temperatures,
calibration provenance, immutable source/data/config identities, trained-step count and parent
checkpoint hash. A separate manifest stores SHA-256, size and resume capability.

Resume-capable checkpoints additionally contain optimizer moments/groups, scheduler state, Python
and Torch RNGs (including all visible CUDA RNG states), sampler permutation/cursor/epoch/independent
RNG, example/microbatch/forward-token/step counters, accumulated elapsed time, run/attempt lineage and
observed maximum training branch/aggregate lengths. Files are saved only after a complete optimizer
step with gradients cleared. There is no unresolved gradient-accumulation state. BF16 autocast does
not use an FP16 gradient scaler. The world/device topology must remain compatible.

Writes use a private temporary file, fsync and atomic replacement. SHA verification precedes loading;
Torch `weights_only=True` avoids arbitrary pickled classes. A trusted source and resource limits are
still required for checkpoint loading; this is not a hostile-file sandbox. A manifest checksum is an
integrity check, not a signature proving who trained a model. The weight-acquisition manifest pins the
source revision and all selected files; independent publisher authenticity remains an operator concern.

Byte tokenizer checkpoints are self-contained. Native checkpoints retain an adjacent `tokenizer.json`
with a checked hash; a mismatched existing tokenizer is never overwritten. Calibration-only outputs
contain no misleading optimizer state and are marked non-resumable. Continuing training invalidates
previous calibration. The checkpoint hash determines the serving model identity.

Tests compare uninterrupted versus interrupted/reloaded training for both CE and PGPS, including
all model tensors, optimizer state, scheduler and sampler. Both continuations are bit-for-bit equal
on the recorded CPU environment. GPU/kernel/topology reproducibility has not been established.

`keep_checkpoints` bounds checkpoints created by one training invocation. Existing checkpoint files
from earlier attempts are not deleted automatically: cross-attempt/global storage retention remains
an operating-policy gap. Do not delete referenced lineage artifacts before applying your retention
policy. Artifact URIs in the evidence bundle are historical session paths; rerun the miniature script
to create a locally valid artifact manifest and registration file.

## Parallel execution addendum

New checkpoints record `execution` and training config hashes bind the full batch policy.
Old weights load unchanged for batched inference or `--init`. Legacy serial optimizer resume
is rejected; any batch-policy change on resume is rejected. See `PARALLEL_QUESTIONS.md`.
