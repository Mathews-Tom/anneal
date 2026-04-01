# ADR-002: Wilcoxon Signed-Rank Test for Stochastic Comparison

## Status
Accepted

## Context

Stochastic evaluation produces a vector of per-sample scores rather than a single deterministic value. For example, a prompt optimization target with `sample_count = 10` produces 10 binary pass/fail scores per run. To decide whether a challenger mutation is better than the current baseline, these score vectors must be compared statistically.

The baseline and challenger are both evaluated against the same set of test prompts in the same experiment, producing paired observations: one score pair per test prompt.

Two common choices for paired comparison are:

- **Paired t-test**: Assumes the differences between pairs are normally distributed. Binary scores (0 or 1) produce differences of {-1, 0, +1}, which are not normally distributed. With small sample counts (6–20), the normality assumption does not hold, making p-values from the t-test unreliable.
- **Wilcoxon signed-rank test**: A non-parametric test that ranks the absolute differences and tests whether the median difference is zero. Makes no distributional assumption. Valid for binary, ordinal, and continuous paired data. With small samples, it is the appropriate test.

A minimum of 6 paired samples is required for the Wilcoxon test to produce a meaningful result (below 6, there are too few distinct rank assignments). When fewer than 6 paired samples are available, the system falls back to an effect-size threshold using Cohen's d on the paired differences.

Holm-Bonferroni correction is applied optionally when `holm_bonferroni=True` is set in the runner, dividing the alpha threshold by the number of remaining comparisons in a rolling 50-experiment window. This controls the family-wise error rate across sequential experiments.

## Decision

`GreedySearch._stochastic_compare` (in `anneal/engine/search.py`) uses `scipy.stats.wilcoxon` with a one-sided alternative (`"greater"` for `HIGHER_IS_BETTER`, `"less"` for `LOWER_IS_BETTER`).

The comparison proceeds as:
1. Compute element-wise differences: `challenger_raw[i] - baseline_raw[i]`.
2. Early-reject if the mean difference is in the wrong direction.
3. If `n < 6`, apply a Cohen's d effect-size threshold of 0.5 (medium effect).
4. Otherwise, run Wilcoxon on the differences with the one-sided alternative.
5. Accept if `p_value < alpha` (default `alpha = 0.05` for `confidence = 0.95`).

The simulated annealing strategy (`SimulatedAnnealingSearch`) uses mean comparison rather than a statistical test, because it deliberately accepts regressions during the warm phase and the acceptance decision is probabilistic by design.

## Consequences

- Stochastic comparison is statistically valid for binary and small-sample eval results, which are the common case in prompt optimization.
- The non-parametric approach makes no assumption about score distribution, making it robust to changes in eval design.
- Requiring at least 6 paired samples means that very cheap evals (1–5 samples) fall back to effect-size comparison, which is less rigorous but still directional.
- `scipy` is a required core dependency. Removing it would break stochastic comparison entirely.
- The Holm-Bonferroni correction reduces the false-positive acceptance rate in long runs at the cost of reduced sensitivity for later experiments in the window.
