# Operating boundaries

The default server is local and request-retention-disabled. API keys are explicit environment values,
constant-time compared, never included in telemetry. No unauthenticated non-loopback bind is allowed
by the CLI. Rate/token quotas are per key **per process**, reset on restart, and are not a distributed
billing/quota service. Overload refuses work before inference instead of maintaining an unbounded
queue. Body size, question count, serialized token budgets and concurrent request state are bounded.
Timeouts, TLS termination, reverse-proxy byte limits, credential rotation and deployment hardening
remain operator responsibilities; this package is not a production security certification.

Each request has a private prefix cache. Request/response contents are neither automatically logged
nor retained for training. Serving hooks see aggregate numbers only. Access logging is disabled by
the CLI. Retention-enabled diagnostics are deliberately unsupported: enabling them requires a separate
explicit design/consent path, not a hidden switch that begins saving user data. Model training data is
an explicit separately licensed input. Tests verify that ordinary inference creates no data files.

Trainer snapshot failures do not abort training or corrupt checkpoints; warnings identify only the
error class. Full checkpoints and evaluations are separate artifacts with checksums. Snapshot history
and serving latency windows are bounded. Checkpoint deletion is bounded within a training invocation;
cross-attempt retention, remote artifact lifecycle and secure deletion need a deployment policy.

The optional publisher is an explicit action: `publish_wandb` writes a validated compact snapshot to
its declared W&B identity; `publish_s3` writes only that snapshot to an explicit encrypted S3 object.
Neither is called by default. Their SDK contracts were tested with doubles; no real publishing was
performed. Full artifact uploads and persistent Sky checkpoint storage are not automatically
configured by those functions. CloudWatch defaults for legacy Flow are unchanged; the new model does
not scrape logs or disguise its output as Flow progress.

The SkyPilot YAML is a candidate resource/setup/run definition, not an executed release. A timeout
bounds the training subprocess, not all setup, provisioning, storage or cloud retry charges. No
self-imposed monetary cap can be inferred from it. Review the cost budget and provider lifecycle
before authorizing launch. The public source/download scripts perform no paid model evaluation.
There is no WSL shutdown, service termination, secret export or remote Git mutation in this package.

Source licenses and software locks are separate from model-weight and dataset rights. Pinned public
model licensing was inspected; native acquisition retains its own license/hash manifest. The only
included training data and trained weights are the generated marker fixture. Audit all real data
sources, consent, domains and deployment risks before training or automating decisions.
