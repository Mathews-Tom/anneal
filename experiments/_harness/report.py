"""Standardized report generation for experiment results."""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone
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


def write_jsonl(records: list[ResultRecord], path: Path, gate_name: str = "") -> None:
    """Write experiment records as engine-compatible JSONL.

    Produces the same format as anneal's KnowledgeStore (ExperimentRecord),
    making the output readable by `anneal dashboard`.
    """
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for r in records:
            entry = {
                "id": str(uuid.uuid4()),
                "target_id": gate_name or r.condition,
                "git_sha": "",
                "pre_experiment_sha": "",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "hypothesis": r.hypothesis,
                "hypothesis_source": "agent",
                "mutation_diff_summary": "",
                "score": r.score,
                "score_ci_lower": r.ci_lower,
                "score_ci_upper": r.ci_upper,
                "raw_scores": r.raw_scores if r.raw_scores else None,
                "baseline_score": r.baseline_score,
                "outcome": "KEPT" if r.kept else "DISCARDED",
                "failure_mode": r.failure_mode or None,
                "duration_seconds": r.duration_seconds,
                "tags": r.tags,
                "learnings": "",
                "cost_usd": r.cost_usd,
                "bootstrap_seed": r.seed,
            }
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
