# Partial calibration remains partial

A generated, group-distinct calibration partition used for local context
feasibility has one Noul example and no Choice or Score examples. Before this
change, `calibrate` correctly recorded `unfitted-no-samples` for those two
types but incorrectly marked the overall artifact `fitted-held-out`. The
service reports that overall status, and the native validator used it as a
calibration prerequisite. The partition SHA-256 is
`640477f3f0bfc951eb1f770e52bc7c11a5a152db6a1d73265d470857e7bf6404`.

The calibration stage now reports `partial-held-out` whenever any required
type has no fitted samples. It retains the explicit per-type temperatures,
counts and statuses. Native acceptance also checks all three per-type fit
statuses rather than trusting the aggregate string alone. A local stage run
on the generated Noul-only partition reported Choice and Score as unfitted,
Noul as fitted, and the overall artifact as partial. Focused tests cover both
a fully fitted result and refusal of partial calibration.

This changes the honesty of artifact and service status. It does not supply
representative calibration data, establish reliable predictive quality, or
make the generated Noul-only artifact acceptable for native validation.
