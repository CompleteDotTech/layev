# Matched model-seed quality variation

`python scripts/report_quality_seeds.py --run 11=SEED11.json --run 12=SEED12.json --run 13=SEED13.json --out PRIVATE_REPORT.json`

The offline reporter requires at least three distinct nonnegative model seeds.
Each input must be an existing evaluation report from `kev_laya.evaluation.evaluate`
with the same split, record/question IDs, groups, labels, question types and
declared slice metadata. It refuses missing or duplicate rows, mismatched
targets, inconsistent splits, nonfinite metrics and replacement of an existing
output. Input file hashes are included in the output.

For each metric, the output gives individual seed values, their mean, sample
standard deviation and observed range, including per-slice results. It is
descriptive variation across the supplied model seeds. It does not replace
source-group bootstrap, establish independence of data groups, prove dataset
approval, or turn fixture evaluations into representative quality evidence.
Keep the per-seed reports and any failed runs; report missing seeds rather than
selecting only favorable runs.
