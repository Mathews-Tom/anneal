"""Experiment run comparison — side-by-side analysis of two runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(stderr=True)


def load_run_records(run_path: Path) -> list[dict[str, Any]]:
    """Load experiment records from a run directory.

    Accepts either a directory containing experiments.jsonl or experiments.csv,
    or a direct path to a JSONL file.
    """
    if run_path.is_file() and run_path.suffix == ".jsonl":
        jsonl_path = run_path
    elif run_path.is_dir():
        jsonl_path = run_path / "experiments.jsonl"
        if not jsonl_path.exists():
            jsonl_path = run_path / "experiments.csv"
            if jsonl_path.exists():
                return _load_csv_records(jsonl_path)
            raise FileNotFoundError(f"No experiments.jsonl or experiments.csv in {run_path}")
    else:
        raise FileNotFoundError(f"Not a valid run path: {run_path}")

    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


def _load_csv_records(csv_path: Path) -> list[dict[str, Any]]:
    """Load records from a CSV file (gate experiment format)."""
    import csv
    records: list[dict[str, Any]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            record: dict[str, Any] = dict(row)
            for numeric_field in ["score", "ci_lower", "ci_upper", "baseline_score", "cost_usd", "duration_seconds"]:
                if numeric_field in record and record[numeric_field]:
                    try:
                        record[numeric_field] = float(record[numeric_field])
                    except ValueError:
                        pass
            records.append(record)
    return records


def compare_runs(
    run_a_path: Path,
    run_b_path: Path,
    label_a: str = "Run A",
    label_b: str = "Run B",
) -> None:
    """Compare two experiment runs and print a rich comparison table."""
    records_a = load_run_records(run_a_path)
    records_b = load_run_records(run_b_path)

    stats_a = _compute_stats(records_a)
    stats_b = _compute_stats(records_b)

    # Summary comparison table
    table = Table(title="Experiment Run Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column(label_a, justify="right")
    table.add_column(label_b, justify="right")
    table.add_column("Delta", justify="right")

    rows = [
        ("Experiments", stats_a["count"], stats_b["count"]),
        ("Best score", stats_a["best_score"], stats_b["best_score"]),
        ("Mean score", stats_a["mean_score"], stats_b["mean_score"]),
        ("Score std dev", stats_a["std_score"], stats_b["std_score"]),
        ("Kept", stats_a["kept"], stats_b["kept"]),
        ("Kept rate", stats_a["kept_rate"], stats_b["kept_rate"]),
        ("Discarded", stats_a["discarded"], stats_b["discarded"]),
        ("Crashed", stats_a["crashed"], stats_b["crashed"]),
        ("Total cost", stats_a["total_cost"], stats_b["total_cost"]),
        ("Avg duration (s)", stats_a["avg_duration"], stats_b["avg_duration"]),
        ("Total duration (min)", stats_a["total_duration_min"], stats_b["total_duration_min"]),
    ]

    for label, val_a, val_b in rows:
        str_a = _format_value(val_a)
        str_b = _format_value(val_b)
        delta = _format_delta(val_a, val_b)
        table.add_row(label, str_a, str_b, delta)

    console.print()
    console.print(table)

    # Score trajectory comparison (text-based)
    if stats_a["scores"] and stats_b["scores"]:
        console.print()
        _print_trajectory_comparison(stats_a["scores"], stats_b["scores"], label_a, label_b)


def _compute_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from experiment records."""
    scores = []
    costs = []
    durations = []
    kept = 0
    discarded = 0
    crashed = 0

    for r in records:
        score = r.get("score")
        if score is not None:
            try:
                scores.append(float(score))
            except (ValueError, TypeError):
                pass

        cost = r.get("cost_usd", 0.0)
        try:
            costs.append(float(cost))
        except (ValueError, TypeError):
            pass

        dur = r.get("duration_seconds", 0.0)
        try:
            durations.append(float(dur))
        except (ValueError, TypeError):
            pass

        outcome = str(r.get("outcome", r.get("kept", ""))).upper()
        if outcome in ("KEPT", "TRUE"):
            kept += 1
        elif outcome in ("DISCARDED", "FALSE"):
            discarded += 1
        elif outcome in ("CRASHED",):
            crashed += 1

    count = len(records)
    total_cost = sum(costs)
    total_duration = sum(durations)

    return {
        "count": count,
        "scores": scores,
        "best_score": max(scores) if scores else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "kept": kept,
        "kept_rate": kept / max(count, 1),
        "discarded": discarded,
        "crashed": crashed,
        "total_cost": total_cost,
        "avg_duration": total_duration / max(count, 1),
        "total_duration_min": total_duration / 60,
    }


def _format_value(val: object) -> str:
    if isinstance(val, float):
        if val == int(val) and val < 10000:
            return str(int(val))
        return f"{val:.4f}"
    return str(val)


def _format_delta(val_a: object, val_b: object) -> str:
    if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
        return ""
    delta = val_b - val_a
    if delta == 0:
        return "[dim]--[/dim]"
    sign = "+" if delta > 0 else ""
    color = "green" if delta > 0 else "red"
    if isinstance(delta, float):
        return f"[{color}]{sign}{delta:.4f}[/{color}]"
    return f"[{color}]{sign}{delta}[/{color}]"


def _print_trajectory_comparison(
    scores_a: list[float],
    scores_b: list[float],
    label_a: str,
    label_b: str,
) -> None:
    """Print a text-based running-best comparison."""
    def running_best(scores: list[float]) -> list[float]:
        result: list[float] = []
        best = float("-inf")
        for s in scores:
            best = max(best, s)
            result.append(best)
        return result

    best_a = running_best(scores_a)
    best_b = running_best(scores_b)

    # Show at key milestones (10%, 25%, 50%, 75%, 100%)
    lines: list[str] = []
    for pct in [0.1, 0.25, 0.5, 0.75, 1.0]:
        idx_a = min(int(pct * len(best_a)) - 1, len(best_a) - 1)
        idx_b = min(int(pct * len(best_b)) - 1, len(best_b) - 1)
        if idx_a >= 0 and idx_b >= 0:
            lines.append(
                f"  At {pct:>4.0%}:  {label_a}={best_a[idx_a]:.4f}  "
                f"{label_b}={best_b[idx_b]:.4f}"
            )

    if lines:
        console.print(Panel(
            "\n".join(lines),
            title="Running Best Score at Milestones",
            style="dim",
        ))
