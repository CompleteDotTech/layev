# Literal preprocessing v4: source candidate and native acceptance gate

Issue: https://github.com/CompleteDotTech/layev/issues/2
Base reviewed: `b4cb9df97be61729ebd68b8e0f5ee9bb2d193c73`.

## Why v4 instead of another v3 patch

Two recovered unpublished implementations both emitted `literal-bpe-v3` and
identical preprocessing metadata for the same tokenizer file, but one counted
output using the original source backend and the other using the literal backend.
A source-contract reproduction produced delimiter output lengths 1 versus 14 and
decomposed-Unicode output lengths 2 versus 3 under identical identities. These
are **double measurements, not pinned Qwen measurements**. A third prior candidate
used `literal-special-v3` and a different in-memory promotion strategy.

This reconciliation adopts an explicit new `literal-bpe-v4` identity. It does not
stack the previous patches and refuses both unpublished v3 tags. Do not relabel an
old checkpoint, training exposure, calibration receipt or run as v4. Fresh
initialization, training, calibration and acceptance evidence are still required.
The original candidate files and their evidence must be retained separately.

## Chosen input and output contracts

Input: read the original tokenizer JSON once, fingerprint those exact bytes, and
construct a separate deterministic BPE backend from those same bytes. Retain the
BPE vocabulary/merges, pre-tokenizer and ByteLevel decoder. Disable all added-token
extraction, normalization, post-processing, padding and truncation. Forbid every
original added-token ID in caller content, including `special: false` tokens.
Every accepted input must decode exactly. Reject unpaired surrogates, unsupported
backend structures, stochastic dropout and non-reversible inputs. No fallback.

Output accounting: `source-json-output-v1` uses the original source backend,
without padding, truncation or automatically inserted special tokens. It may
normalize text and recognize added-token spellings. It is deliberately not the
literal input policy. The original vocabulary size and control IDs are retained.

The JSON state wrapper remains `kev-laya-prefix-v2`. In addition to the original
source SHA-256 and literal version, the semantic identity binds the canonical
derived-backend SHA-256 and output policy in a recomputable v4 fingerprint inside
the literal_encoding string. The exact three-field telemetry v2 shape and original
source tokenizer identity remain unchanged; explicit components remain in checkpoint
metadata and native-gate receipts. Paths do not enter semantic identity.
Checkpoint loading requires the complete v4 identity and matching tokenizer
metadata. Missing, altered, conflicting or v3-tagged metadata is rejected.
Checkpoint export rejects a source tokenizer changed since initialization before
writing it into a new checkpoint directory.

Explicit `special-spelling-v1` and `literal-special-v2` modes keep historical
behavior and limitations. Pinned-Qwen v2 refusal is not silently upgraded. Legacy
byte-fixture identity is unchanged; omission compatibility is not extended to v4.

## Existing authorized workstation gate

Use the existing isolated Layev model environment on the user's PC. Do not move
CUDA, modify Overwatch's environment, terminate WSL, overwrite runs, or change run
identities. Inspect the actual primary checkout, worktrees, parent AGENTS.md and
pending patches before applying this replacement candidate on a branch.

Run the repository-required checks in a complete checkout:

```sh
python -m pytest -q -ra
python -m compileall -q src tests scripts
```

For the offline tokenizer gate, use the actual upstream snapshot of
`Qwen/Qwen2.5-0.5B` at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, its original source receipt, a separately
reviewed upstream tokenizer SHA-256, and the real installed `tokenizers==0.22.1`.
Do not bless an arbitrary local fixture by supplying its own hash as authority.

```sh
python scripts/verify_pinned_tokenizer.py \
  --source-dir "$KEV_LAYA_QWEN_DIR" \
  --expected-tokenizer-sha256 "$REVIEWED_PINNED_TOKENIZER_SHA256"
```

PowerShell uses `$env:KEV_LAYA_QWEN_DIR` and
`$env:REVIEWED_PINNED_TOKENIZER_SHA256`. Capture stdout to a new evidence file, not
an existing run record. The script itself performs no network calls or source
writes. Exit 2 is a missing/mismatched prerequisite, exit 1 is a failure, and exit
0 means only the tokenizer gate passed. A skipped/blocked gate is not acceptance.

The gate checks literal spellings, all original added tokens, Unicode/escaping,
nested JSON, metadata-only question IDs, exact token budgets, derived identity,
output-policy agreement with the original backend, metadata reload, rejection of
both v3 tags, and unchanged source bytes. It does NOT establish real-weight
initialization, trained checkpoint reload, serving, CUDA, trained 32k/64k context,
quality, official SDK acceptance, live transport or Jev parity.

## Evidence in this reconciliation session

On Python 3.13.5, the unchanged verified main encoding preimage fails 30 of the
38 literal contract-double tests and passes 8. The reconciled focused source,
identity, checkpoint-boundary and gate-preflight tests pass 132/132 with no skips.
An additional exact extraction of the tokenizer-only stage3 test functions passes
12/12; this is explicitly NOT the entire stage3 suite. Compileall passes.

The required full-suite attempt in the partial source export exits 2 during
collection because `kev_laya.telemetry` is absent from that export. That module
exists in the live main tree; this is an export limitation, not a claim that main
is broken. Ruff is unavailable. The native gate exits 2 (`source_directory_missing`)
with zero literal cases and all native/model/CUDA/quality fields untested.

No native acceptance, source commit, PR, push, merge, workstation synchronization,
or populated Overwatch UI verification is established by this candidate.

## Windows primary-checkout readback (September 27, 2026)

The patch was applied to the clean primary Layev checkout after PR #12. On
Python 3.13.12, the complete repository suite passed **413 tests with 4 skips**;
the real pinned-Qwen tokenizer test ran rather than skipping. Compileall and the
CI-pinned Ruff critical lint selection passed. The offline gate passed 36 literal
cases with `tokenizers==0.22.1`, including the selected source-output policy,
metadata reload and v3 refusal. The local 7,031,645-byte `tokenizer.json` had
SHA-256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`,
which matched a fresh download from Qwen's pinned upstream revision. The gate
reported the v4 semantic fingerprint
`literal-bpe-v4:002e76a7328e372b4ca78a015cc203a522994eb150be050f834a136572d4965e`.

This environment has a CPU-only PyTorch build and no local pinned model weights.
The CUDA, actual pretrained initialization/checkpoint/serving, trained-context,
official SDK, quality and Jev-parity gates remain open. The four skips were the
official SDK, CUDA FP32, CUDA BF16 and a Windows symlink-privilege case.
