"""Load raw benchmark results from JSONL files.

Each JSONL file in benchmarks/raw_results/ follows the naming convention:
    <target_id>-<config_name>-seed<seed>.jsonl

Each line is a JSON object representing one ExperimentRecord produced by the
anneal runner. The loader groups records by (target_id, config_name, seed)
and computes per-run aggregate statistics.

Expected filename pattern: B1-control-seed3.jsonl
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# Regex for the benchmark result filename convention.
_FILENAME_RE = re.compile(
    r"^(?P<target>[A-Za-z0-9_-]+)-(?P<config>[A-Za-z0-9_-]+)-seed(?P<seed>\d+)\.jsonl$"
)

# Fraction of best score considered "converged".
_CONVERGENCE_THRESHOLD = 0.90


@dataclass
class RunResult:
    """Aggregated statistics for a single optimization run (one seed).

    Attributes:
        target_id: Benchmark target identifier (e.g. "B1").
        config_name: Configuration label (e.g. "control", "treatment").
        seed: Random seed index used for this run.
        final_score: Score at the last experiment.
        scores: Score trajectory across all experiments in order.
        costs: Cumulative cost (USD) trajectory across all experiments.
        convergence_experiment: First experiment index (1-based) where the
            score reaches at least 90% of the run's best score.  Set to the
            total experiment count when that threshold is never reached within
            the run.
        acceptance_rate: Fraction of experiments with outcome "KEPT".
        total_cost_usd: Total API spend across all experiments in the run.
        experiment_count: Number of experiments recorded in the run.
    """

    target_id: str
    config_name: str
    seed: int
    final_score: float
    scores: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    convergence_experiment: int = 0
    acceptance_rate: float = 0.0
    total_cost_usd: float = 0.0
    experiment_count: int = 0


def _parse_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL file and return a list of parsed records.

    Skips blank lines and raises ValueError on malformed JSON.
    """
    records: list[dict[str, object]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON on line {lineno} of {path}: {exc}"
            ) from exc
        records.append(obj)
    return records


def _build_run_result(
    target_id: str,
    config_name: str,
    seed: int,
    records: list[dict[str, object]],
) -> RunResult:
    """Compute per-run aggregates from a list of raw ExperimentRecord dicts."""
    if not records:
        return RunResult(
            target_id=target_id,
            config_name=config_name,
            seed=seed,
            final_score=0.0,
            experiment_count=0,
        )

    scores: list[float] = []
    costs: list[float] = []
    kept_count: int = 0
    cumulative_cost: float = 0.0

    for rec in records:
        score = float(rec.get("score", 0.0))
        cost = float(rec.get("cost_usd", 0.0))
        outcome = str(rec.get("outcome", ""))

        scores.append(score)
        cumulative_cost += cost
        costs.append(cumulative_cost)

        if outcome == "KEPT":
            kept_count += 1

    n = len(records)
    final_score = scores[-1]
    best_score = max(scores)
    threshold = _CONVERGENCE_THRESHOLD * best_score

    convergence_experiment = n  # default: never reached within the run
    for idx, s in enumerate(scores):
        if s >= threshold:
            convergence_experiment = idx + 1  # 1-based
            break

    acceptance_rate = kept_count / n if n > 0 else 0.0

    return RunResult(
        target_id=target_id,
        config_name=config_name,
        seed=seed,
        final_score=final_score,
        scores=scores,
        costs=costs,
        convergence_experiment=convergence_experiment,
        acceptance_rate=acceptance_rate,
        total_cost_usd=cumulative_cost,
        experiment_count=n,
    )


def load_results(results_dir: Path) -> list[RunResult]:
    """Load all benchmark run results from a directory of JSONL files.

    Files that do not match the expected naming pattern are silently skipped.

    Args:
        results_dir: Directory containing ``<target>-<config>-seed<N>.jsonl``
            files.

    Returns:
        List of RunResult objects, one per matched file.

    Raises:
        FileNotFoundError: If results_dir does not exist.
        ValueError: If a matched JSONL file contains malformed JSON.
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    run_results: list[RunResult] = []

    for path in sorted(results_dir.glob("*.jsonl")):
        match = _FILENAME_RE.match(path.name)
        if match is None:
            continue

        target_id = match.group("target")
        config_name = match.group("config")
        seed = int(match.group("seed"))

        records = _parse_jsonl(path)
        run_result = _build_run_result(target_id, config_name, seed, records)
        run_results.append(run_result)

    return run_results


def group_by_target(results: list[RunResult]) -> dict[str, list[RunResult]]:
    """Group run results by target_id.

    Args:
        results: Flat list of RunResult objects.

    Returns:
        Dict mapping each target_id to the runs belonging to that target.
    """
    grouped: dict[str, list[RunResult]] = {}
    for run in results:
        grouped.setdefault(run.target_id, []).append(run)
    return grouped


def group_by_config(results: list[RunResult]) -> dict[str, list[RunResult]]:
    """Group run results by config_name.

    Args:
        results: Flat list of RunResult objects.

    Returns:
        Dict mapping each config_name to all runs using that configuration.
    """
    grouped: dict[str, list[RunResult]] = {}
    for run in results:
        grouped.setdefault(run.config_name, []).append(run)
    return grouped
