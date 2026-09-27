# Native weight loading: integrity and failure boundaries

This change supports the native initialization acceptance in Layev issues #2 and
#3. It is not a native-model, CUDA, long-context, quality or Jev-parity receipt.
The existing native-context harness and tokenizer-v4 contracts are unchanged.

## Load the bytes whose hash is recorded

`DecisionEngine.load_qwen_weights(path, expected_sha256=...)` reads the source
once into an immutable byte buffer, computes SHA-256 on that buffer, and gives
those same bytes to `safetensors.torch.load`. A supplied expected digest must be
64 lowercase hexadecimal characters. A mismatch is rejected before tensor
parsing. The successful `loaded_backbone_sha256` describes exactly the bytes
parsed, not a later read of a pathname that might now name a replacement file.

Direct callers may omit the expected digest for backward API compatibility.
That only records which local bytes were loaded. The `native-init` command always
passes the weight digest from its checked five-file source manifest. Neither a
local digest nor a caller-supplied manifest authenticates upstream origin:
independent acquisition, reviewed pinned revision and license provenance remain
required before native acceptance. No network download is performed by the loader.

The original `model.` prefix mapping is retained. Only backbone tensors are
imported; unrelated output-head tensors are not imported. LoRA keeps its existing
q/v base-weight mapping, trainable adapter parameters and independent pointer head.

## Reject malformed tensors before mutation

The loader checks the complete backbone key set and every tensor's exact shape,
floating dtype, finiteness and destination-dtype range before copying parameters.
This prevents ordinary missing-key, extra-key, shape and nonfinite-input failures
from partly replacing an already loaded backbone. A finite wider-range tensor
that would overflow the destination dtype is rejected rather than copied.

A checksum, parser or validation rejection leaves the prior model and its source
receipt unchanged. Immediately before the copy, the loader clears
`native_weights_loaded` and `loaded_backbone_sha256`; they become successful only
after the copy returns successfully. An unexpected copy/device failure may still
partly change parameters, but no stale native-success receipt is retained.
**This is not transactional rollback.** Discard or explicitly reload such a model;
do not continue serving it based on the old receipt.

Use this operation on an exclusively owned model during initialization. It does
not provide synchronization with concurrent inference, training, other loaders,
or custom state-dict hooks. It is not a general-purpose hostile-checkpoint sandbox.
`native_weights_loaded` records successful backbone import, not verified upstream
identity, training, calibration, predictive quality or release approval.

## Native initialization order

`native-init` refuses an existing checkpoint or sidecar manifest, including a
symlink at either output path, before reading source inputs. It then validates the
source declaration, checks and retains the config bytes, checks the auxiliary
file digests, initializes the tokenizer, verifies its exact source identity and
embedding-vocabulary compatibility, and only then allocates the model and loads
weights with the expected digest. The config is parsed from the retained verified
bytes, not from a second read. The same tokenizer object is passed to checkpoint
export; no preprocessing version, API or telemetry shape is changed here.

The output paths are checked again before export to avoid replacing a result
that appeared during loading. Exclusive destination ownership is still required:
the check and subsequent writes are not an atomic filesystem transaction.
Existing checkpoint serialization and tokenizer-copy validation continue to apply.

## Memory and performance tradeoff

The immutable byte-based path intentionally replaces memory-mapped source loading.
It can hold the serialized file, parsed CPU tensors and destination parameters
concurrently. Peak host-memory and load-time costs for real Qwen weights have not
been measured by the source tests. This change makes no low-memory or speed claim,
and does not silently fall back to the old pathname-based provenance behavior.
Keep the model environment separate from Overwatch and measure the real native
workload on the authorized PC before considering its hardware gates satisfied.

## Verification

In the actual complete checkout and existing authorized model environment:

```text
python -m pytest tests/test_native_weight_loading.py -q
python -m pytest -q
python -m compileall -q src tests scripts
python -m kev_laya.cli native-init --backbone-dir <verified-pinned-snapshot> --out <new-private-directory>/init.pt
```

The source tests use the actual Safetensors library and miniature synthetic CPU
weights. Command-order tests explicitly use a tokenizer double and checkpoint
writer double. Synthetic source declarations are test inputs, not authenticated
Qwen snapshots. Test model flags are not native acceptance evidence.

The original review package records 52 focused synthetic cases and a partial
historical supporting suite. Those results do not substitute for tests in the
current checkout or hosted CI. Actual native initialization, reload, serving,
CUDA, and independent-model comparison remain separate acceptance gates. Leave
the native acceptance issues open until their full evidence is demonstrated.
