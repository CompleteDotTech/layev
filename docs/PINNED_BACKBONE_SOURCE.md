# Pinned backbone source-byte gate

This opt-in, read-only preflight supports the source-provenance part of
[Layev #3](https://github.com/CompleteDotTech/layev/issues/3). It is **not a native
model test**, and it does not close #2, #3, or any dependent acceptance issue.
It adds no dependency and leaves the existing downloader, native-init command,
loader, checkpoint, tokenizer, training, service, and telemetry code unchanged.

## Independent weight anchor

The primary source is Qwen's [Git LFS pointer at the exact pinned revision](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors),
inspected on 27 September 2026. It identifies:

- Repository: `Qwen/Qwen2.5-0.5B`.
- Revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- File: `model.safetensors`, exactly **988097824 bytes**.
- SHA-256: `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.

The `raw` URL above returns a **small pointer, not model weights**. Saving that
pointer as `model.safetensors` cannot satisfy this gate. The existing
`scripts/download_backbone.py` uses the pinned `resolve` URL for actual acquisition.
Inspecting the pointer does not mean the 988 MB payload has been downloaded or
executed. This anchor is a dated primary-source observation, not a verified
upstream cryptographic signature or an assurance about the model's quality.

The new check compares both the weight manifest entry and the measured weight
bytes against that **fixed independent anchor**. A locally edited manifest cannot
choose another accepted weight digest. There is no command-line digest, size,
fixture, URL, or alternate-model override.

## Run in the existing authorized local workspace

Keep the CUDA environment on the user's workstation. In the intended current
Layev checkout, supply the already downloaded regular-file snapshot:

```powershell
python -I -S scripts/verify_backbone_source.py --source '<existing-pinned-snapshot>'
if ($LASTEXITCODE -ne 0) { throw 'Backbone source gate did not verify' }
```

`-I -S` demonstrates that no environment/site packages are needed. The wrapper
executes this checkout's source module directly, not a historical installed wheel.
The command prints one JSON receipt and returns **0** for this narrow verified
scope, **1** for invalid input, or **2** for absent/unreadable prerequisites. A
no-input call returns 2 with `source_directory_required`, zero bytes read, and zero
native cases. Unrecognized command-line options are argparse usage errors (2),
not a JSON source-validation result. Save receipts outside the input directory,
using the workstation's normal no-overwrite evidence workflow.

The gate makes no network calls, downloads nothing, writes no input files and
creates no model/run/checkpoint/output directory. OS-maintained access timestamps
may change on reads. The Python wrapper disables bytecode-cache writes for the
module it executes. Receipt fields contain fixed file names, hashes, sizes and
controlled reason codes, not caller paths, manifest contents, credentials or raw
OS exceptions.

## What a verified receipt means

The five filenames and four top-level fields must match the acquisition manifest
written by the existing downloader: `repository`, `revision`, `license`, `sha256`.
All hashes must be lowercase hexadecimal SHA-256 strings; missing/extra files in
the manifest, duplicate JSON keys, and malformed declarations are rejected.
Additional unrelated files on disk are neither enumerated nor read.

The measured **weight bytes** must match the independently pinned digest and
size. `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `LICENSE` must
match **their local manifest hashes only**. Their upstream origin remains
**unverified by this gate**, even on exit 0. Auxiliary files are not parsed for
model compatibility, tokenizer functionality or license permissions. Matching
these local hashes cannot replace the separate source/tokenizer/license review.
`all_local_file_hashes_verified` describes integrity, not authenticated origin.

Every receipt explicitly leaves Safetensors parsing untested, upstream signature
verification incomplete, and tokenizer execution, pretrained initialization,
checkpoint reload, serving, CUDA and trained context not run. Representative
quality is unmeasured and Jev parity unknown. JSON receipts are observations, not
cryptographic attestations of execution. Record the reviewed code revision and
command with the receipt; hash syntax alone is not evidence of an actual run.

## Bounds and filesystem limitations

The gate first admits all required paths/sizes, then hashes small inputs and the
weight file last. Reads are at most 1 MiB; only the at-most-64-KiB source manifest
is retained for JSON parsing. The auxiliary maxima are 64 KiB for config/license,
256 KiB for tokenizer configuration, and 32 MiB for tokenizer JSON. An extra byte
may be read to detect growth. These are local admission limits, not a claim about
native load-time RAM, GPU memory or latency. No wall-clock deadline or cancellation
of blocked filesystem I/O is provided.

Inputs and their ancestors must be regular files/directories, without symlinks or
Windows reparse points (including junctions); FIFOs/devices are rejected before
opening. A Hugging Face cache represented by symlinks must not be passed directly.
Use the existing regular-file acquisition layout; the gate does not copy, resolve,
repair, chmod, or delete input data. It rejects parent-traversal path spelling.

File identity/size/time metadata are checked around open/read and again before
completion. Windows uses device, file ID, mode, size and modification time; its
creation-time value can differ between pathname and descriptor reads immediately
after a write. POSIX additionally checks change time. Directory identities are
rechecked. This catches the tested ordinary
replacement/edit races, **not all adversarial filesystem manipulation**. It is not
an atomic multi-file snapshot and does not lock files against later writes.
Windows behavior has source tests for reparse detection. Verification and later model loading
require trusted, stable inputs and exclusive ownership of the source/output
locations. A receipt does not prove that a later loader consumed the same bytes.
PR #16's immutable-buffer loader repair is complementary and is not replaced here.

## Source tests and acceptance boundary

```text
python -m pytest tests/test_backbone_provenance.py -q
python -m pytest -q
python -m compileall -q src tests scripts
```

Positive unit cases explicitly monkeypatch the module's anchor to small synthetic
bytes. That is **test mechanics only**, not a real-weight gate pass. A separate
subprocess runs the unmodified production pin and rejects the same self-consistent
synthetic snapshot. Tests cover missing/malformed/corrupt/oversized inputs,
manifest-selected paths, links, file changes, read bounds, controlled error output,
CLI codes, and package-free execution. No model library or native execution is
substituted. Repository and hosted checks are required for source changes; they
do not establish real-weight or native execution. This file does not assert that
all acceptance criteria are satisfied.
