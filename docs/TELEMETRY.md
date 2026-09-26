# Telemetry v2, historical v1 and collector cache v2

The authoritative cross-field validator is `src/kev_laya/telemetry_contract.py`. The byte-identical standard-library-only copy in `integrations/overwatch/added/src/overwatch/kev_laya_contract.py` keeps Torch/training dependencies out of Overwatch. `schemas/telemetry.schema.json` accepts validated historical v1 and current v2 shapes; `telemetry-v1.schema.json` preserves the old contract. Run `python scripts/export_schemas.py` after a contract change.

## Producer contract

Snapshots remain bounded to 256 KiB, 128 history points, 128 recent serving latencies and 32 artifacts. Legacy top-level identity/progress/provenance/artifact/phase/serving fields remain recognizable. V2 adds validated `extensions`: serialization identity, source hashes/Git status, bounded attempt receipts/durability, parallel execution totals and sampled resources. Compact calibration fit details are optional for historical v2 compatibility. No request text, question text, response content or credentials are permitted. Null means unknown, not zero. V2 permits unknown data/config hashes rather than manufacturing placeholder hashes.

The trainer owns atomic exports and its attempt ledger; only Overwatch's collector owns the raw cache. Stable experiment/run IDs are distinct from attempt ID/index/sequence, scheduler namespace/job ID, W&B entity/project/run ID and checkpoint lineage. Receipts bind 64 consecutive attempts. The collector exposes unresolved observations separately from last verified attempts and preserves them across restarts. See `STAGE3_CORRECTNESS.md` for trust boundaries, missed-poll proof and failure behavior.

Counters distinguish optimizer steps, accumulation microbatches/examples, useful forward tokens, prefix passes, suffix tensor-batch passes, questions, padding and total forward compute. Compute excludes backward recomputation. RSS/CUDA memory are measured bytes; sampling timestamps/intervals and unavailable fields are explicit. Provider status, trainer phase, resume state, freshness and recovery counts are not interchangeable. Known provider recovery totals do not imply a known breakdown.

## Registration and transport budgets

```json
{"schema_version":1,"sources":[
  {"id":"local-run","transport":"local","path":"/absolute/path/to/telemetry.json"},
  {"id":"s3-run","transport":"s3","bucket":"authorized-bucket","key":"runs/run/telemetry.json"},
  {"id":"wandb-project","transport":"wandb","entity":"authorized-entity","project":"project","limit":32}
]}
```

The registry is bounded to 512 sources/128 KiB and validated entirely before payload I/O. No recursive scans or per-resource CLI/log subprocesses are used. `OVERWATCH_KEV_LAYA_REGISTRY` overrides the service-user default `~/.config/overwatch/kev-laya-sources.json`. Publishing is opt-in and separate from registration. W&B summary/config publication targets a pre-existing run and never calls `finish()`.

Per-pass defaults: 16 MiB payload, 512 request slots, eight explicit W&B pages, 4,096 stream reads and a 15-second **cooperative** deadline with bounded network timeouts. Reads stop at the remaining byte allowance. Invalid bytes and unknown-length overflow probes count. Registry bytes are accounted separately. W&B envelope bytes count; opaque transport overhead does not. In-flight SDK/credential work is not hard-cancelled. Live cloud behavior is unverified; instrumented doubles verify starts, reads, pages, retries configuration and stop behavior.

A source skipped by budget, deadline or outage remains registered and retains its last cached record plus last successful refresh time. A missing registry is not deregistration. Structured `sources`, `complete`, `registry_complete`, `warnings` and `accounting` explain a partial pass.

## Collector-owned cache and migration

`global/kev_laya/runs-v2.json` is versioned `kev_laya/raw/2`, with verified records, separate bounded unverified observations, collection-source freshness, accounting and warnings. The shared Overwatch root still defaults to `~/.cache/overwatch/raw-v1`: Sky/Flow/billing files are unchanged. `OVERWATCH_RAW_CACHE_ROOT` is an explicit override for isolated tests or an operator-selected cache.

If v2 is absent, the cache reader accepts `runs-v1.json` as migration input. The collector writes v2 without overwriting v1. A corrupt v2 is not silently replaced by an older v1: the reader returns an explicit cache warning. Rendering reads only cache files and never opens producer exports or provider APIs. The existing React resources page adds a model-experiment section; local model records do not masquerade as SkyPilot resources.

Exact primary-checkout preflight, local-only collector mode, supported environment prerequisites and the real application/UI gate are documented in `integrations/overwatch/README.md`. Adapter/provider-only checks are not the full application gate.
