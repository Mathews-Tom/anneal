"""Summary generation for benchmark statistical analysis.

Generates:
  1. Markdown table of statistical comparisons (for README or paper appendix).
  2. JSON file with all results for programmatic consumption.
  3. Enhancement attribution analysis linking metrics to treatment components.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.analysis.loader import RunResult, group_by_config, group_by_target
from benchmarks.analysis.statistics import (
    ComparisonResult,
    SummaryStats,
    summarize_config,
)


# Enhancements present in the "treatment" configuration and the metric each
# most directly influences. Used for attribution analysis.
_ENHANCEMENT_METRIC_MAP: dict[str, str] = {
    "Strategy Manifest": "final_score",
    "Dual-Agent Mutation": "total_cost_usd",
    "Two-Phase Mutation": "acceptance_rate",
    "Lineage Context": "convergence_experiment",
    "Episodic Memory": "final_score",
}


def _fmt_p(p: float) -> str:
    """Format a p-value for table display."""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _fmt_d(d: float, ci: tuple[float, float]) -> str:
    """Format Cohen's d with CI for table display."""
    return f"{d:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def generate_markdown_table(comparisons: list[ComparisonResult]) -> str:
    """Render a Markdown table of pairwise statistical comparison results.

    One row per (target, pair). Columns: target, config pair, means, p-value
    (raw and corrected), Cohen's d with CI, and significance flag.

    Args:
        comparisons: List of ComparisonResult objects.

    Returns:
        Markdown-formatted string.
    """
    header = (
        "| Target | Comparison | Mean A | Mean B | p (raw) | p (corrected) "
        "| Cohen's d [95% CI] | Significant |\n"
        "|--------|------------|--------|--------|---------|---------------"
        "|--------------------|-------------|\n"
    )
    rows: list[str] = []

    for cr in comparisons:
        pair = f"{cr.config_a} vs {cr.config_b}"
        sig = "Yes" if cr.significant else "No"
        rows.append(
            f"| {cr.target_id} | {pair} | {cr.mean_a:.4f} | {cr.mean_b:.4f} "
            f"| {_fmt_p(cr.p_value)} | {_fmt_p(cr.p_value_corrected)} "
            f"| {_fmt_d(cr.cohens_d, cr.cohens_d_ci)} | {sig} |"
        )

    return header + "\n".join(rows) + "\n"


def generate_summary_stats_table(stats: list[SummaryStats]) -> str:
    """Render a Markdown table of descriptive statistics per configuration.

    Args:
        stats: List of SummaryStats objects.

    Returns:
        Markdown-formatted string.
    """
    header = (
        "| Config | Metric | N | Mean | Median | Std | IQR | Min | Max |\n"
        "|--------|--------|---|------|--------|-----|-----|-----|-----|\n"
    )
    rows: list[str] = []

    for s in stats:
        rows.append(
            f"| {s.config_name} | {s.metric} | {s.n} "
            f"| {s.mean:.4f} | {s.median:.4f} | {s.std:.4f} "
            f"| {s.iqr:.4f} | {s.minimum:.4f} | {s.maximum:.4f} |"
        )

    return header + "\n".join(rows) + "\n"


def generate_attribution_analysis(
    results: list[RunResult],
    control_config: str = "control",
    treatment_config: str = "treatment",
) -> dict[str, dict[str, float]]:
    """Estimate which enhancements contributed most to treatment improvements.

    For each enhancement, computes the relative difference between treatment
    and control on the enhancement's primary metric, aggregated across all
    targets.

    Args:
        results: All run results.
        control_config: Name of the control (baseline) configuration.
        treatment_config: Name of the treatment configuration.

    Returns:
        Dict mapping enhancement name to a dict with keys:
            - "metric": The primary metric.
            - "control_mean": Mean of metric for control.
            - "treatment_mean": Mean of metric for treatment.
            - "relative_change_pct": 100 * (treatment - control) / |control|.
    """
    by_target = group_by_target(results)
    attribution: dict[str, dict[str, float | str]] = {}

    for enhancement, metric in _ENHANCEMENT_METRIC_MAP.items():
        control_values: list[float] = []
        treatment_values: list[float] = []

        for target_runs in by_target.values():
            ctrl_runs = [r for r in target_runs if r.config_name == control_config]
            trt_runs = [r for r in target_runs if r.config_name == treatment_config]

            control_values.extend(getattr(r, metric) for r in ctrl_runs)
            treatment_values.extend(getattr(r, metric) for r in trt_runs)

        if not control_values or not treatment_values:
            continue

        ctrl_mean = sum(control_values) / len(control_values)
        trt_mean = sum(treatment_values) / len(treatment_values)
        denom = abs(ctrl_mean) if ctrl_mean != 0.0 else 1.0
        rel_change = 100.0 * (trt_mean - ctrl_mean) / denom

        attribution[enhancement] = {
            "metric": metric,
            "control_mean": round(ctrl_mean, 6),
            "treatment_mean": round(trt_mean, 6),
            "relative_change_pct": round(rel_change, 2),
        }

    return attribution  # type: ignore[return-value]


def write_json_results(
    comparisons: list[ComparisonResult],
    summary_stats: list[SummaryStats],
    attribution: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Serialise all analysis results to a JSON file.

    Args:
        comparisons: Statistical comparison results.
        summary_stats: Descriptive statistics per config.
        attribution: Enhancement attribution analysis.
        output_path: Destination JSON file path.
    """
    payload: dict[str, object] = {
        "comparisons": [asdict(cr) for cr in comparisons],
        "summary_stats": [asdict(s) for s in summary_stats],
        "attribution": attribution,
    }

    # ComparisonResult contains tuple fields that JSON cannot serialise
    # natively; convert to lists.
    def _to_serialisable(obj: object) -> object:
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serialisable(v) for v in obj]
        return obj

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_serialisable(payload), indent=2),
        encoding="utf-8",
    )


def write_markdown_summary(
    comparisons: list[ComparisonResult],
    summary_stats: list[SummaryStats],
    attribution: dict[str, dict[str, float]],
    output_path: Path,
    config_pairs: list[tuple[str, str]] | None = None,
) -> None:
    """Write a full Markdown summary document to disk.

    Sections: statistical comparisons, descriptive stats, attribution analysis.

    Args:
        comparisons: Statistical comparison results.
        summary_stats: Descriptive statistics per config.
        attribution: Enhancement attribution analysis.
        output_path: Destination Markdown file path.
        config_pairs: Optional list of config pairs for section headings.
    """
    sections: list[str] = ["# Benchmark Analysis Summary\n"]

    sections.append("## Statistical Comparisons\n")
    sections.append(generate_markdown_table(comparisons))

    sections.append("\n## Descriptive Statistics\n")
    sections.append(generate_summary_stats_table(summary_stats))

    sections.append("\n## Enhancement Attribution\n")
    attr_header = (
        "| Enhancement | Metric | Control Mean | Treatment Mean "
        "| Relative Change (%) |\n"
        "|-------------|--------|-------------|----------------|"
        "--------------------|\n"
    )
    attr_rows: list[str] = []
    for enhancement, data in attribution.items():
        attr_rows.append(
            f"| {enhancement} | {data['metric']} "
            f"| {data['control_mean']:.4f} | {data['treatment_mean']:.4f} "
            f"| {data['relative_change_pct']:+.2f}% |"
        )
    sections.append(attr_header + "\n".join(attr_rows) + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")


def compute_all_summary_stats(
    results: list[RunResult],
    metrics: list[str] | None = None,
) -> list[SummaryStats]:
    """Compute descriptive statistics for every config-metric combination.

    Args:
        results: All run results.
        metrics: Metrics to summarise. Defaults to all four primary metrics.

    Returns:
        List of SummaryStats, one per (config, metric) pair.
    """
    if metrics is None:
        metrics = ["final_score", "convergence_experiment", "acceptance_rate", "total_cost_usd"]

    by_config = group_by_config(results)
    all_stats: list[SummaryStats] = []

    for config_name, config_runs in sorted(by_config.items()):
        for metric in metrics:
            try:
                s = summarize_config(config_runs, config_name, metric=metric)
                all_stats.append(s)
            except ValueError:
                continue

    return all_stats
