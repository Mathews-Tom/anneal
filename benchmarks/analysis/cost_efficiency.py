"""Cost-efficiency plot generation for benchmark results.

Generates matplotlib plots showing score vs. cumulative cost (USD), one line
per configuration, averaged across seeds.  Demonstrates which configuration
reaches target scores at lower API spend.

matplotlib is optional and must be installed via the [dashboard] extra:
    uv pip install 'anneal-cli[dashboard]'
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.analysis.loader import RunResult, group_by_config


_INTERPOLATION_POINTS = 200


def _interpolate_cost_curve(
    runs: list[RunResult],
    cost_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate each run's score-vs-cost trajectory onto a common cost grid.

    For each run, score is held constant between experiments (step function).
    The final score is extrapolated to the right edge of the cost grid.

    Args:
        runs: List of RunResult objects for one configuration.
        cost_grid: 1-D array of cost values at which to evaluate score.

    Returns:
        2-D array of shape (len(runs), len(cost_grid)).
    """
    matrix = np.zeros((len(runs), len(cost_grid)), dtype=float)

    for i, run in enumerate(runs):
        if not run.costs or not run.scores:
            continue
        costs = np.array(run.costs, dtype=float)
        scores = np.array(run.scores, dtype=float)
        # Use step interpolation: left-fill for values below first cost point.
        matrix[i] = np.interp(cost_grid, costs, scores, left=scores[0], right=scores[-1])

    return matrix


# Color palette — colorblind-friendly, consistent with convergence.py.
_CONFIG_COLORS = {
    "raw": "#E69F00",
    "greedy": "#56B4E9",
    "control": "#009E73",
    "treatment": "#CC79A7",
}
_FALLBACK_COLORS = ["#0072B2", "#D55E00", "#F0E442", "#000000"]


def _config_color(config: str, used: list[str]) -> str:
    if config in _CONFIG_COLORS:
        return _CONFIG_COLORS[config]
    idx = len(used) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]


def plot_cost_efficiency(
    results: list[RunResult],
    target_id: str,
    output_path: Path,
    configs: list[str] | None = None,
) -> None:
    """Generate and save a cost-efficiency plot for one benchmark target.

    X-axis: cumulative API cost in USD.
    Y-axis: score (averaged across seeds at each cost point).

    Args:
        results: All run results (filtered internally to target_id).
        target_id: Benchmark target to plot (e.g. "B1").
        output_path: Destination path for the PNG file.
        configs: Optional list of config names to include.  Defaults to all
            configs present in the data for the given target.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If no results exist for target_id.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for cost efficiency plots. "
            "Install it with: uv pip install 'anneal-cli[dashboard]'"
        ) from exc

    target_runs = [r for r in results if r.target_id == target_id]
    if not target_runs:
        raise ValueError(f"No results found for target '{target_id}'")

    by_config = group_by_config(target_runs)
    if configs is not None:
        by_config = {k: v for k, v in by_config.items() if k in configs}

    if not by_config:
        raise ValueError(
            f"No matching configurations found for target '{target_id}'. "
            f"Available: {list(group_by_config(target_runs))}"
        )

    # Determine global cost range across all configurations.
    max_cost = 0.0
    for runs in by_config.values():
        for run in runs:
            if run.costs:
                max_cost = max(max_cost, run.costs[-1])

    if max_cost <= 0.0:
        raise ValueError(
            f"All runs for target '{target_id}' have zero cost. "
            "Cannot generate cost efficiency plot."
        )

    cost_grid = np.linspace(0.0, max_cost, _INTERPOLATION_POINTS)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    used_fallback: list[str] = []

    for config_name, runs in sorted(by_config.items()):
        color = _config_color(config_name, used_fallback)
        if config_name not in _CONFIG_COLORS:
            used_fallback.append(config_name)

        matrix = _interpolate_cost_curve(runs, cost_grid)
        mean_scores = matrix.mean(axis=0)

        ax.plot(cost_grid, mean_scores, label=config_name, color=color, linewidth=1.8)

    ax.set_xlabel("Cumulative Cost (USD)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Cost Efficiency — Target {target_id}", fontsize=13)
    ax.set_xlim(0.0, max_cost)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
