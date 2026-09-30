# Paired144 development repair: timeout readback (September 30, 2026)

The fixed-budget synthetic training attempt timed out after **5400.990688599995 seconds**, with worker exit **-15**. The supervisor recorded the same runtime source commit at start and end (`192c54ddcc48b16bec20f7e37f31915bbd97de13`).

The durable checkpoint is **96 of 144 planned current-run updates** (426 total historical training steps). The retained checkpoint48 and checkpoint96 bytes match their manifests and form the recorded parent chain. Updates performed after the last durable checkpoint are **unknown**.

No checkpoint144, completed worker terminal, calibrated output, or repair readout was observed in this run directory. This attempt therefore supplies **no completed repair, calibration, readout, or acceptance result**. Resumable checkpoint metadata does not authorize resuming or extending the expired 5400-second budget.

The [earlier observed96 diagnostic](development-factorial96-20260930.md) remains unchanged: Choice79/96, Score90/96, Noul48/96 with all predictions true. It evaluated the earlier calibrated v8 checkpoint; it is not a readout of this paired training attempt. Its statement that paired144 was pending describes that historical readback. The original v8 marker gate remains failed at 3/18 against the unchanged 80% threshold; the historical v6 6/18 result remains separate.

A new run initialized from weights resets current-run exposure; resumed training preserves exposure. Even a future completed paired run would require the separately prescribed exact-limit continuation before independent recalibration and a new gate. No representative quality claim is supported, and this receipt changes no acceptance flags. Overwatch remains archived.

The [portable receipt](paired144-timeout-20260930.json) records supervisor and manifest hashes, verified checkpoint hashes, and the scoped file inventory. Verification used CPU file reads only, without model loading or GPU execution.
