# Preserved feature research and current Layev mapping

`FEATURE_MATRIX.original.json` is a byte-identical copy of the September 25,
2026 companion research matrix, SHA-256
`bc002dbc82b69eb456c1f3d7e11c83d0da5cfe83c699219faa4b88aa54ded405`.
The companion folder is not a Git repository, so there is no source commit to
attribute to the original matrix. Its generator, verifier and clone receipt
files had these SHA-256 identities when inspected:

| Companion file | SHA-256 |
|---|---|
| `scripts/build_feature_matrix.py` | `8739e6673cda38e9523d2f80113d5a90a276d48aa171849016834d996ca9fa84` |
| `scripts/verify_research.py` | `055c756e5892cd4b9f413fec0ac6355ad68dc4d966c47b7a30c27f82640387d2` |
| `clones.json` | `5818fbaeca6711a5e964c914d5712f2d22db2ec61cadb48e8dde24b5f1320e5c` |

The generator defines the 45 IDs and primary TypeSafe citations. The verifier
checks the nine pinned comparison clones, source receipts, 45 unique IDs and
450 project-feature cells in that companion workspace. The clone receipt paths
are local to that workspace and are not portable checkout paths. The original
matrix includes commit-pinned HTTPS source URLs beside those local paths; use
those URLs from a clean Layev clone. Neither the generator nor the verifier
proves native Layev behavior, licensed representative quality, or Jev parity.

`layev-feature-mapping.json` preserves every original ID, title, definition,
group and citation in order, then records independent Layev source, test,
native, quality and Jev evidence. Its `Layev_inspected_revision` identifies the
source inventory snapshot. The 14 distinct feature citations and the additional
model-jaggedness reference were reachable on September 27, 2026; the original
September 25 definitions were retained rather than rewritten. Run
`python scripts/verify_feature_matrix.py` from the repository root to check
the original bytes, exact ID mapping, project-cell count and local source/test
paths. Hosted CI runs that verifier on each PR and main push.
