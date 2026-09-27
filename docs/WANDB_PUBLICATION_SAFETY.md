# W&B publication retry safety

This is a source-level safety repair related to
[Layev issue 10](https://github.com/CompleteDotTech/layev/issues/10).
It does not close the authorized live-publication gate, the matching
[Overwatch transport gate](https://github.com/CompleteDotTech/Overwatch/issues/5),
or any model-quality acceptance criterion.

## Defect and repair

Previously the publisher stored the result of `Api.run()` in `self.run` before
checking its framework or completing configuration setup. A different-framework
rejection or a configuration-update exception left this cached object behind.
The next call skipped the entire setup block, including the framework guard,
and could write a snapshot to a previously rejected run. The training loop catches
publication failures and calls the publisher again, so this was reachable through
ordinary retry scheduling, not just a standalone direct call.

The returned run now stays local until framework validation and configuration
update succeed. Failure leaves no initialized cached handle. Each later caller-
initiated attempt must resolve and validate a target again. On the successful
cached path, the publisher checks the available run configuration before each
summary write. Unrelated config entries, the explicit destination identity,
telemetry schemas, and the existing ten-second API timeout are preserved.

The publisher does not call `init()`, `finish()`, `create()`, `delete()` or any
run-state update. It adds no internal retry loop, transport, credentials, rate
limit override or automatic publication. Disabled mode makes no SDK calls.

## Evidence and limits

Run from the model environment:

```text
python -m pytest tests/test_wandb_publication_safety.py -q
python -m pytest -q
python -m compileall -q src tests scripts
```

The focused tests use explicit SDK doubles. Three tests also execute actual
three-step local CPU optimization and checkpoint writes with synthetic data and
injected framework/setup/summary failures. They verify warning accounting, run
identity retention, checkpoint durability and omission of exception content from
warnings. These are not real W&B exchanges, native Qwen or CUDA evidence, a full
Overwatch pass, representative quality, or Jev parity.

A cached config check is **not** a remote refresh or an atomic cross-writer
precondition. No cloud-side compare-and-swap or global thread/process ownership
is claimed. A failed network operation can have an uncertain remote outcome;
there is no rollback or exactly-once claim. An SDK may have its own transport
retries. Retrying setup after a failure may perform additional reads, and those
reads must be included in a real request-budget receipt. This change does not
introduce rate enforcement. Concurrent publisher use is not newly supported.

The upstream [Public API Run reference](https://docs.wandb.ai/models/ref/python/public-api/run)
describes configuration and mutable summary updates. Consult the exact installed
SDK version during live acceptance; documentation inspection is not a live SDK
compatibility result.

Before closing issue 10, use existing approved disposable targets and credentials,
record exact run identity, SDK version, payload/hash, request and rate accounting,
and observable config/summary readback. Exercise disabled mode and authorized
remote failures while retaining local optimization/checkpoint durability. Keep
S3 publication/readback and the collector-side evidence separate. Do not publish
credentials, user requests, answers, weights, or private datasets as test evidence.
