# Jev comparison: offline paired scoring preparation

As checked on 2026-09-28, [TypeSafe's models page](https://docs.typesafe.ai/models)
lists `jev-1.13.0` as the current versioned Jev model. Its
[HTTP reference](https://docs.typesafe.ai/api) describes Choice, Score and
Noul answers with the same basic question fields. Pin the response's versioned
model ID in any real comparison; a moving alias is not an experiment identity.

`scripts/score_jev_comparison.py` is an **offline scorer**. It does not call
Layev or TypeSafe. It reads the frozen suite's untouched test partition and
two private JSONL transcript files in that exact record order, with one line
per test record. Each line has only `request` and `response`. The request is a
System One object with `state`, `questions` and the arm's `model`; the response
has `model` and `answers`. The scorer checks the model IDs, request identity
excluding the model name, answer IDs/types, option keys and finite probability
distributions. It applies the same accuracy, NLL, Brier, ECE, ordinal error and
risk/coverage rules to both arms, with summaries by question type, domain,
language, option order and option count. Request/response hashes and transcript
file hashes are retained in its output. It refuses to overwrite an output file.

Example, using private approved files and a frozen versioned Jev ID:

```text
python scripts/score_jev_comparison.py --suite SUITE_DIR --layev-transcripts LAYEV.jsonl --jev-transcripts JEV.jsonl --layev-model kev-laya-preview --jev-model jev-1.13.0 --out PRIVATE_REPORT.json
```

This tool does not establish data-review authenticity, remote-transfer
permission, response provenance, retries, latency, billed cost, native context
or representative quality. It marks latency, billed cost, authorization and
Jev parity as unverified. A real run still needs an approved and frozen request
set, an authenticated data-use record explicitly permitting transfer to
TypeSafe, account access, bounded request/cost controls, actual service
receipts, and the completed #4/#5 evidence. Keep failed requests and all
responses; do not feed only successful rows to this scorer and call that a
complete comparison.
