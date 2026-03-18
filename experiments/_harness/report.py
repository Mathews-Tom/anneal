"""Standardized report generation for experiment results."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from experiments._harness.types import ResultRecord


def write_csv(records: list[ResultRecord], path: Path) -> None:
    """Write experiment records to CSV with dynamic per-criterion columns."""
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all per_criterion keys across records
    criterion_keys: list[str] = []
    seen: set[str] = set()
    for r in records:
        for k in r.per_criterion:
            if k not in seen:
                criterion_keys.append(k)
                seen.add(k)

    base_fields = [
        "experiment_id", "condition", "hypothesis", "score",
        "ci_lower", "ci_upper", "baseline_score", "kept",
        "cost_usd", "duration_seconds", "seed", "tags", "failure_mode",
    ]
    header = base_fields + [f"criterion_{k}" for k in criterion_keys]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in records:
            row = [
                r.experiment_idx,
                r.condition,
                r.hypothesis,
                r.score,
                r.ci_lower,
                r.ci_upper,
                r.baseline_score,
                r.kept,
                r.cost_usd,
                r.duration_seconds,
                r.seed,
                ";".join(r.tags),
                r.failure_mode,
            ]
            for k in criterion_keys:
                row.append(r.per_criterion.get(k, ""))
            writer.writerow(row)


def write_summary(
    results: dict[str, list[ResultRecord]],
    path: Path,
    title: str = "EXPERIMENT SUMMARY",
) -> None:
    """Write a summary.txt matching the established format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"{title}")
    lines.append(sep)

    best_condition = ""
    best_score = float("-inf")

    for condition, records in sorted(results.items()):
        scores = [r.score for r in records]
        total_cost = sum(r.cost_usd for r in records)
        total_time = sum(r.duration_seconds for r in records)
        kept_count = sum(1 for r in records if r.kept)
        cond_best = max(scores) if scores else 0.0

        lines.append("")
        lines.append(f"  {condition}:")
        lines.append(f"    Best score:        {cond_best:.4f}")
        lines.append(f"    Mean score:        {sum(scores) / len(scores):.4f}" if scores else "    Mean score:        N/A")
        lines.append(f"    Total cost:        ${total_cost:.4f}")
        lines.append(f"    Experiments:       {len(records)}")
        lines.append(f"    Kept mutations:    {kept_count}")
        lines.append(f"    Total time:        {total_time / 60:.1f} min ({total_time / max(len(records), 1):.1f}s avg/experiment)")

        if cond_best > best_score:
            best_score = cond_best
            best_condition = condition

    total_cost_all = sum(r.cost_usd for recs in results.values() for r in recs)
    total_time_all = sum(r.duration_seconds for recs in results.values() for r in recs)

    lines.append("")
    lines.append("-" * 60)
    lines.append(f"  Winner: {best_condition} (score: {best_score:.4f})")
    lines.append(f"  Total time: {total_time_all / 60:.1f} min")
    lines.append(f"  Total cost: ${total_cost_all:.4f}")
    lines.append(sep)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_summary_to_string(results: dict[str, list[ResultRecord]], title: str = "EXPERIMENT SUMMARY") -> str:
    """Return summary as a string (for terminal output)."""
    buf = io.StringIO()
    path = Path("/dev/null")  # placeholder
    # Use write_summary logic inline
    lines: list[str] = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(title)
    lines.append(sep)

    for condition, records in sorted(results.items()):
        scores = [r.score for r in records]
        total_cost = sum(r.cost_usd for r in records)
        kept_count = sum(1 for r in records if r.kept)
        cond_best = max(scores) if scores else 0.0

        lines.append(f"\n  {condition}:")
        lines.append(f"    Best score:     {cond_best:.4f}")
        lines.append(f"    Experiments:    {len(records)}")
        lines.append(f"    Kept:           {kept_count}")
        lines.append(f"    Cost:           ${total_cost:.4f}")

    lines.append(sep)
    return "\n".join(lines)
