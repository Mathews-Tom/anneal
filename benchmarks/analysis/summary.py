"""Summary generation for benchmark statistical analysis.

Generates:
  1. Markdown table of statistical comparisons (for README or paper appendix).
  2. JSON file with all results for programmatic consumption.
  3. Enhancement attribution analysis linking metrics to treatment components.

All aggregation is done per-target. Pooling descriptive statistics across
targets is unsound because targets use heterogeneous metric spaces (e.g. B3
reports wall-clock seconds under ``direction=minimize`` while B1/B2/B4/B5
report composite 0-5 scores under ``direction=maximize``).
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


# Targets whose primary metric is minimized rather than maximized. Used to
# flip the sign of relative-change calculations in the attribution table so
# that "improvement" always reads as a positive percentage.
_MINIMIZATION_TARGETS: frozenset[str] = frozenset({"B3"})

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
    """Render a Markdown table of descriptive statistics per (target, config).

    Rows are sorted by target_id, then config_name, then metric so the output
    is deterministic and diff-friendly.

    Args:
        stats: List of SummaryStats objects.

    Returns:
        Markdown-formatted string.
    """
    header = (
        "| Target | Config | Metric | N | Mean | Median | Std | IQR | Min | Max |\n"
        "|--------|--------|--------|---|------|--------|-----|-----|-----|-----|\n"
    )
    ordered = sorted(stats, key=lambda s: (s.target_id, s.config_name, s.metric))
    rows: list[str] = []

    for s in ordered:
        rows.append(
            f"| {s.target_id} | {s.config_name} | {s.metric} | {s.n} "
            f"| {s.mean:.4f} | {s.median:.4f} | {s.std:.4f} "
            f"| {s.iqr:.4f} | {s.minimum:.4f} | {s.maximum:.4f} |"
        )

    return header + "\n".join(rows) + "\n"


def generate_attribution_analysis(
    results: list[RunResult],
    control_config: str = "control",
    treatment_config: str = "treatment",
) -> dict[str, dict[str, dict[str, float | str]]]:
    """Estimate which enhancements contributed to treatment improvements.

    For each (enhancement, target) pair, computes the relative difference
    between treatment and control on the enhancement's primary metric.
    Minimization targets (``_MINIMIZATION_TARGETS``) have the sign flipped so
    that a positive relative change always reads as "treatment improved over
    control". Metrics from different targets are never pooled — every number
    is scoped to one target.

    Args:
        results: All run results.
        control_config: Name of the control (baseline) configuration.
        treatment_config: Name of the treatment configuration.

    Returns:
        Nested dict ``{enhancement: {target_id: {metric, direction,
        control_mean, treatment_mean, relative_change_pct}}}``.
    """
    by_target = group_by_target(results)
    attribution: dict[str, dict[str, dict[str, float | str]]] = {}

    for enhancement, metric in _ENHANCEMENT_METRIC_MAP.items():
        per_target: dict[str, dict[str, float | str]] = {}

        for target_id, target_runs in sorted(by_target.items()):
            ctrl_runs = [r for r in target_runs if r.config_name == control_config]
            trt_runs = [r for r in target_runs if r.config_name == treatment_config]
            if not ctrl_runs or not trt_runs:
                continue

            ctrl_values = [float(getattr(r, metric)) for r in ctrl_runs]
            trt_values = [float(getattr(r, metric)) for r in trt_runs]

            ctrl_mean = sum(ctrl_values) / len(ctrl_values)
            trt_mean = sum(trt_values) / len(trt_values)
            denom = abs(ctrl_mean) if ctrl_mean != 0.0 else 1.0
            raw_change = 100.0 * (trt_mean - ctrl_mean) / denom

            direction = "minimize" if target_id in _MINIMIZATION_TARGETS else "maximize"
            # Minimization: a drop in the metric is an improvement, so flip
            # the sign so positive always means "treatment better than control".
            rel_change = -raw_change if direction == "minimize" else raw_change

            per_target[target_id] = {
                "metric": metric,
                "direction": direction,
                "control_mean": round(ctrl_mean, 6),
                "treatment_mean": round(trt_mean, 6),
                "relative_change_pct": round(rel_change, 2),
            }

        if per_target:
            attribution[enhancement] = per_target

    return attribution


def write_json_results(
    comparisons: list[ComparisonResult],
    summary_stats: list[SummaryStats],
    attribution: dict[str, dict[str, dict[str, float | str]]],
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
    attribution: dict[str, dict[str, dict[str, float | str]]],
    output_path: Path,
) -> None:
    """Write a full Markdown summary document to disk.

    Sections: statistical comparisons, descriptive stats, attribution analysis.

    Args:
        comparisons: Statistical comparison results.
        summary_stats: Descriptive statistics per (target, config, metric).
        attribution: Nested attribution dict ``{enhancement: {target_id: ...}}``.
        output_path: Destination Markdown file path.
    """
    sections: list[str] = ["# Benchmark Analysis Summary\n"]

    sections.append("## Statistical Comparisons\n")
    sections.append(generate_markdown_table(comparisons))

    sections.append("\n## Descriptive Statistics\n")
    sections.append(generate_summary_stats_table(summary_stats))

    sections.append("\n## Enhancement Attribution\n")
    attr_header = (
        "| Enhancement | Target | Metric | Direction "
        "| Control Mean | Treatment Mean | Relative Change (%) |\n"
        "|-------------|--------|--------|-----------"
        "|--------------|----------------|---------------------|\n"
    )
    attr_rows: list[str] = []
    for enhancement, per_target in attribution.items():
        for target_id, data in sorted(per_target.items()):
            ctrl_mean = float(data["control_mean"])  # type: ignore[arg-type]
            trt_mean = float(data["treatment_mean"])  # type: ignore[arg-type]
            rel_change = float(data["relative_change_pct"])  # type: ignore[arg-type]
            attr_rows.append(
                f"| {enhancement} | {target_id} | {data['metric']} "
                f"| {data['direction']} | {ctrl_mean:.4f} | {trt_mean:.4f} "
                f"| {rel_change:+.2f}% |"
            )
    sections.append(attr_header + "\n".join(attr_rows) + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")


def compute_all_summary_stats(
    results: list[RunResult],
    metrics: list[str] | None = None,
) -> list[SummaryStats]:
    """Compute descriptive statistics for every (target, config, metric) cell.

    Args:
        results: All run results.
        metrics: Metrics to summarise. Defaults to all four primary metrics.

    Returns:
        List of SummaryStats, one per (target_id, config_name, metric) triple.
    """
    if metrics is None:
        metrics = ["final_score", "convergence_experiment", "acceptance_rate", "total_cost_usd"]

    by_target = group_by_target(results)
    all_stats: list[SummaryStats] = []

    for target_id, target_runs in sorted(by_target.items()):
        by_config = group_by_config(target_runs)
        for config_name, config_runs in sorted(by_config.items()):
            for metric in metrics:
                try:
                    s = summarize_config(
                        config_runs,
                        config_name=config_name,
                        target_id=target_id,
                        metric=metric,
                    )
                    all_stats.append(s)
                except ValueError:
                    continue

    return all_stats
