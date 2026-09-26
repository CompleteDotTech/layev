# Learning and calibration contract

## PGPS-v1: Projected Gaussian Proper-Score Policy Gradient

For a valid K-option logit vector z, let P = I - 11ᵀ/K and μ = Pz. Draw G independent
ε_g ~ Normal(0, σ²I), project them by P, and take **detached** actions a_g = stop_gradient(μ) + Pε_g.
This is a Gaussian distribution on the K−1 dimensional zero-sum logit subspace. Invalid padded option
coordinates remain zero and are excluded from normalization, density, reward and gradients. A single
valid option has a zero-dimensional policy and zero policy gradient.

For q = softmax(a), target distribution y and ordered Score levels, the reward is

    R(q,y) = Σ y_k log q_k + w_spherical (y·q / ||q||₂)
             − 1[type=Score] w_ordinal Σ_j (CDF(q)_j − CDF(y)_j)² / max(K−1,1).

Log-softmax supplies the log score without clipping it to an arbitrary reward floor. The log score
and spherical score are proper scores, and negative ranked probability score is appropriate for
ordered categorical outcomes. Numerical floating-point implementations still need finite checks.
Matching a noisy-policy reward does **not** itself prove that the deterministic served distribution
is calibrated. That question is evaluated separately on held-out data.

The baseline for sample g is the mean reward of the **other** G−1 samples, conditional on the same
input. Both reward and baseline are detached. No group-standard-deviation normalization is applied.
The score-function surrogate is

    L_PGPS = − mean_g [(R_g − mean_{h≠g} R_h)
                       × (−||a_g − μ||² / (2σ²))].

The Gaussian normalizer is independent of μ and omitted. Centering μ correctly projects gradients
into the same subspace. This is a likelihood-ratio estimator, not a pathwise gradient through the
sample. A Monte Carlo unit test compares its expected gradient with an independently differentiable
pathwise reward reference; additional tests check finite gradients in the backbone, pointer head,
LoRA path, masked coordinates and activation-checkpointed execution.

Total training loss is configurable:

    L = c_CE × soft_target_cross_entropy + c_PGPS × L_PGPS + c_ordinal × RPS.

Hard labels are converted to one-hot targets. Soft targets must be finite, nonnegative, correctly
sized and sum to one. Coefficients, noise, group size, seed, precision and optimizer configuration are
written to every run's config and checkpoint provenance. Defaults do not impersonate an upstream
proprietary recipe. No auxiliary action/confidence head is exposed.

## Data and calibration

A suite contains exactly train, development, calibration and test partitions. Its immutable manifest
records file hashes, counts, source/license statements and grouping policy. Loading verifies hashes,
unique record IDs, group separation and normalized-state exact duplicate separation. **Semantic
near-duplicate detection is not automatic**: real data needs audited source-family grouping before
freezing. The included fixture uses source-case groups; this is not a substitute for real-data leakage
auditing. Serving never calls dataset construction or appends inference records to these partitions.

Calibration caches detached logits from the calibration partition and minimizes NLL per question
type with a positive temperature bounded to [0.2,5]. The optimizer keeps the best value including
T=1; absent types remain unfitted. No model weight is updated. The fitting CLI cannot select test or
training partitions. A calibrated artifact is deliberately inference-only; continuing weight training
resets the old temperature fit and requires a new calibration stage.

Evaluation reports NLL, multiclass Brier, 10-bin ECE, reliability bins, expected hard/soft-target
agreement, ordinal expectation MAE and risk-versus-coverage, with breakdowns by question type, domain,
language, option count and context length. Coverage boundaries respect tied confidences. The
probability quantity used for calibration plots is max(p), **not** public entropy concentration.
Neither post-hoc fitting nor one ECE value guarantees calibration on another population.

## Controlled miniature ablation

The retained protocol freezes 160 SFT optimizer steps, two 40-step extensions from the identical
parent (SFT-only versus PGPS), and a temperature-fit readout. Both extensions use the same learning
rate, accumulation and seed. Evaluation definitions and the 0.70 fixture threshold precede outputs.
All four arms reached 1.0 marker accuracy, so this task gives no evidence of an accuracy advantage
from reward training. Its small proper-score differences are descriptive fixture results, not a
statistical superiority claim. No checkpoint or hyperparameter was selected using the test results.

The fixture's four language tags change marker words; they do not establish genuine multilingual
understanding. Native long-context training, broad licensed held-out quality, paired uncertainty
estimates and the pinned Kev/Laya baselines are unrun. Jev-relative quality is unverified without an
explicitly authorized and budgeted comparison. Do not weaken a benchmark to convert these gaps into
success. The original 45-column research matrix remains unavailable.

## Observed option-order failure and engineering correction

After the first frozen fixture read, the real installed-server probe returned a confidently wrong
Choice when the criterion insertion order was reversed. A complete diagnostic on its 34 held-out
Choice cases measured 20 flips (58.82%); reversed-order accuracy was 41.18%, despite 100% canonical
order accuracy. This failure is retained in the evidence rather than hidden behind the successful
aggregate test score. It also illustrates why entropy concentration is not probability of correctness.

The correction adds configurable label-preserving Choice permutation augmentation to the same
training path. Soft targets are remapped with their labels; Score levels and Noul order are never
shuffled. The permutation RNG is checkpointed and interrupted continuation remains exact. The
follow-up v2 engineering protocol keeps the original 0.70 fixture criterion and separately fixes an
order-flip threshold of 0.10 before its readout. Its weights had zero flips on all 34 reordered cases.
This is a regression result on a previously observed simple fixture, not untouched benchmark
evidence, a general option-order invariance proof, or a claim of Jev equivalence. Both before and
after reports are delivered. New recommended training configs enable the augmentation; the exact
unaugmented config is retained as `configs/smoke-legacy-unaugmented.json`.


## Retained out-of-fixture failure

The final installed-wheel check returned **blue** (probability 0.8653368189) for the explicit
state `color=red; level=1; case=99999`. The direct in-process result and HTTP result agree.
This is a model/data generalization failure, not a serving discrepancy. The synthetic generator
correlates color, ordinal level, language keyword and source-case number through its case index;
the split does not remove those shortcuts. It therefore cannot support real-quality claims even
when all 102 recorded test questions are correct. The follow-up permutation intervention fixed
the measured option-order regression on those 34 Choice cases only. No thresholds were changed
to mark the out-of-fixture failure as a quality pass. The diagnostic remains failed in
`evidence/installed-http-verification/report.json`. Real curated data and counterfactual
challenge sets, frozen independently before training, are an outstanding acceptance gate.
