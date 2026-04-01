"""Statistical tests for benchmark comparison.

Implements the analysis pipeline:
  1. Pairwise Wilcoxon signed-rank test between configuration pairs.
  2. Holm-Bonferroni correction over n_targets comparisons.
  3. Cohen's d effect size with 95% bootstrap confidence interval.
  4. Summary statistics (mean, median, std, IQR) per config per metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from benchmarks.analysis.loader import RunResult


_BOOTSTRAP_ITERATIONS = 10_000
_BOOTSTRAP_CI_LEVEL = 0.95
_WILCOXON_ZERO_METHOD: Literal["wilcox", "pratt", "zsplit"] = "wilcox"


@dataclass
class ComparisonResult:
    """Statistical comparison between two configurations on one metric.

    Attributes:
        target_id: Benchmark target the comparison belongs to.
        config_a: Name of the first (reference) configuration.
        config_b: Name of the second (treatment) configuration.
        metric: Name of the metric compared (e.g. "final_score").
        mean_a: Mean of the metric for config_a.
        mean_b: Mean of the metric for config_b.
        p_value: Raw two-sided Wilcoxon p-value.
        p_value_corrected: Holm-Bonferroni corrected p-value.
        cohens_d: Cohen's d effect size (positive means config_b > config_a).
        cohens_d_ci: 95% bootstrap CI for Cohen's d as (lower, upper).
        significant: True if p_value_corrected < alpha.
    """

    target_id: str
    config_a: str
    config_b: str
    metric: str
    mean_a: float
    mean_b: float
    p_value: float
    p_value_corrected: float
    cohens_d: float
    cohens_d_ci: tuple[float, float]
    significant: bool


@dataclass
class SummaryStats:
    """Descriptive statistics for a metric within one configuration.

    Attributes:
        config_name: Configuration label.
        metric: Metric name.
        n: Number of observations.
        mean: Arithmetic mean.
        median: Median.
        std: Standard deviation.
        iqr: Interquartile range (Q3 - Q1).
        minimum: Minimum value.
        maximum: Maximum value.
    """

    config_name: str
    metric: str
    n: int
    mean: float
    median: float
    std: float
    iqr: float
    minimum: float
    maximum: float


def _extract_metric(runs: list[RunResult], metric: str) -> np.ndarray:
    """Extract a named metric as a 1-D numpy array from a list of RunResult.

    Supported metric names: "final_score", "convergence_experiment",
    "acceptance_rate", "total_cost_usd".

    Raises:
        ValueError: If the metric name is not recognized.
    """
    valid = {"final_score", "convergence_experiment", "acceptance_rate", "total_cost_usd"}
    if metric not in valid:
        raise ValueError(f"Unknown metric '{metric}'. Valid options: {valid}")
    return np.array([getattr(r, metric) for r in runs], dtype=float)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    Uses pooled standard deviation.  Returns positive value when b > a.

    Args:
        a: Sample from group A.
        b: Sample from group B.

    Returns:
        Cohen's d (float). Returns 0.0 when the pooled SD is zero.
    """
    mean_diff = float(np.mean(b) - np.mean(a))
    pooled_var = (np.var(a, ddof=1) * (len(a) - 1) + np.var(b, ddof=1) * (len(b) - 1)) / (
        len(a) + len(b) - 2
    )
    pooled_sd = float(np.sqrt(pooled_var))
    if pooled_sd == 0.0:
        return 0.0
    return mean_diff / pooled_sd


def _bootstrap_cohens_d_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_iterations: int = _BOOTSTRAP_ITERATIONS,
    ci_level: float = _BOOTSTRAP_CI_LEVEL,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Cohen's d.

    Args:
        a: Sample from group A.
        b: Sample from group B.
        n_iterations: Number of bootstrap resamples.
        ci_level: Confidence level (default 0.95).
        rng: Optional numpy Generator for reproducibility.

    Returns:
        Tuple (lower, upper) CI bounds.
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    bootstrap_ds: list[float] = []
    for _ in range(n_iterations):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        bootstrap_ds.append(_cohens_d(sample_a, sample_b))

    alpha = 1.0 - ci_level
    lower = float(np.percentile(bootstrap_ds, 100.0 * alpha / 2.0))
    upper = float(np.percentile(bootstrap_ds, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def _holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values.

    Args:
        p_values: Raw p-values for m comparisons.
        alpha: Family-wise error rate threshold.

    Returns:
        Corrected p-values in the same order as the input.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values by magnitude; keep original indices for reinversion.
    order = sorted(range(m), key=lambda i: p_values[i])
    corrected = [0.0] * m

    for rank, idx in enumerate(order):
        adjusted = p_values[idx] * (m - rank)
        # Monotonicity constraint: corrected p cannot decrease along the sequence.
        if rank > 0:
            prev_corrected_idx = order[rank - 1]
            adjusted = max(adjusted, corrected[prev_corrected_idx])
        corrected[idx] = min(adjusted, 1.0)

    return corrected


def compare_configs(
    results: list[RunResult],
    config_a: str,
    config_b: str,
    metric: str = "final_score",
    alpha: float = 0.05,
    n_corrections: int = 1,
) -> ComparisonResult:
    """Wilcoxon signed-rank test comparing two configurations on one metric.

    Filters results to runs from config_a and config_b, aligns them by seed,
    and performs the paired Wilcoxon test.  Holm-Bonferroni correction is
    applied assuming n_corrections total comparisons.

    Args:
        results: All run results (any target, any config).
        config_a: Reference configuration name.
        config_b: Treatment configuration name.
        metric: Metric to compare.
        alpha: Significance threshold after correction.
        n_corrections: Number of simultaneous comparisons for Holm-Bonferroni.
            Pass the total number of comparisons being made in the family.

    Returns:
        ComparisonResult with raw and corrected statistics.

    Raises:
        ValueError: If there are no matching runs for either configuration.
    """
    runs_a = {r.seed: r for r in results if r.config_name == config_a}
    runs_b = {r.seed: r for r in results if r.config_name == config_b}

    shared_seeds = sorted(set(runs_a) & set(runs_b))
    if not shared_seeds:
        raise ValueError(
            f"No shared seeds between configs '{config_a}' and '{config_b}'. "
            f"Seeds in A: {sorted(runs_a)}, seeds in B: {sorted(runs_b)}"
        )

    paired_a = np.array([getattr(runs_a[s], metric) for s in shared_seeds], dtype=float)
    paired_b = np.array([getattr(runs_b[s], metric) for s in shared_seeds], dtype=float)

    differences = paired_b - paired_a

    if np.all(differences == 0):
        p_raw = 1.0
    else:
        stat_result = stats.wilcoxon(
            differences,
            zero_method=_WILCOXON_ZERO_METHOD,
            alternative="two-sided",
        )
        p_raw = float(stat_result.pvalue)

    # Holm-Bonferroni with a single comparison simplifies to Bonferroni.
    raw_family = [p_raw] + [1.0] * (n_corrections - 1)
    corrected_family = _holm_bonferroni(raw_family, alpha=alpha)
    p_corrected = corrected_family[0]

    d = _cohens_d(paired_a, paired_b)
    d_ci = _bootstrap_cohens_d_ci(paired_a, paired_b)

    target_ids = {r.target_id for r in results if r.config_name in (config_a, config_b)}
    target_id = next(iter(target_ids)) if len(target_ids) == 1 else "mixed"

    return ComparisonResult(
        target_id=target_id,
        config_a=config_a,
        config_b=config_b,
        metric=metric,
        mean_a=float(np.mean(paired_a)),
        mean_b=float(np.mean(paired_b)),
        p_value=p_raw,
        p_value_corrected=p_corrected,
        cohens_d=d,
        cohens_d_ci=d_ci,
        significant=p_corrected < alpha,
    )


def run_all_comparisons(
    results: list[RunResult],
    pairs: list[tuple[str, str]],
    n_targets: int = 5,
    metric: str = "final_score",
    alpha: float = 0.05,
) -> list[ComparisonResult]:
    """Run pairwise comparisons with Holm-Bonferroni correction over all targets.

    For each (config_a, config_b) pair, this function:
      1. Groups results by target_id.
      2. Runs a Wilcoxon test per target.
      3. Applies Holm-Bonferroni correction across the n_targets p-values for
         each pair.

    Args:
        results: All run results across all targets and configurations.
        pairs: List of (config_a, config_b) tuples to compare.
        n_targets: Number of targets (controls Holm-Bonferroni family size).
        metric: Metric to compare.
        alpha: Family-wise significance threshold.

    Returns:
        List of ComparisonResult objects, one per (pair, target).
    """
    from benchmarks.analysis.loader import group_by_target

    by_target = group_by_target(results)
    all_results: list[ComparisonResult] = []

    for config_a, config_b in pairs:
        raw_p_values: list[float] = []
        per_target_data: list[tuple[str, list[RunResult]]] = []

        for target_id in sorted(by_target):
            target_runs = by_target[target_id]
            runs_a = [r for r in target_runs if r.config_name == config_a]
            runs_b = [r for r in target_runs if r.config_name == config_b]

            if not runs_a or not runs_b:
                continue

            per_target_data.append((target_id, target_runs))

            # Compute raw p without correction for now.
            try:
                interim = compare_configs(
                    target_runs,
                    config_a,
                    config_b,
                    metric=metric,
                    n_corrections=1,
                )
                raw_p_values.append(interim.p_value)
            except ValueError:
                raw_p_values.append(1.0)

        corrected = _holm_bonferroni(raw_p_values, alpha=alpha)

        for i, (target_id, target_runs) in enumerate(per_target_data):
            try:
                cr = compare_configs(
                    target_runs,
                    config_a,
                    config_b,
                    metric=metric,
                    n_corrections=1,
                )
                # Replace the corrected p-value with the family-corrected one.
                corrected_result = ComparisonResult(
                    target_id=target_id,
                    config_a=cr.config_a,
                    config_b=cr.config_b,
                    metric=cr.metric,
                    mean_a=cr.mean_a,
                    mean_b=cr.mean_b,
                    p_value=cr.p_value,
                    p_value_corrected=corrected[i],
                    cohens_d=cr.cohens_d,
                    cohens_d_ci=cr.cohens_d_ci,
                    significant=corrected[i] < alpha,
                )
                all_results.append(corrected_result)
            except ValueError:
                continue

    return all_results


def summarize_config(
    results: list[RunResult],
    config_name: str,
    metric: str = "final_score",
) -> SummaryStats:
    """Compute descriptive statistics for one configuration's metric values.

    Args:
        results: Run results (may span multiple targets).
        config_name: Configuration to summarize.
        metric: Metric name.

    Returns:
        SummaryStats dataclass.

    Raises:
        ValueError: If no results exist for the given config.
    """
    runs = [r for r in results if r.config_name == config_name]
    if not runs:
        raise ValueError(f"No results found for config '{config_name}'")

    values = _extract_metric(runs, metric)
    q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))

    return SummaryStats(
        config_name=config_name,
        metric=metric,
        n=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values, ddof=1)),
        iqr=q3 - q1,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
    )
