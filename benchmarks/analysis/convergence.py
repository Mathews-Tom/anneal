"""Convergence curve generation for benchmark results.

Generates matplotlib convergence plots: score vs. experiment number, one line
per configuration, averaged across seeds with bootstrap CI bands.

matplotlib is optional and must be installed via the [dashboard] extra:
    uv pip install 'anneal-cli[dashboard]'
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.analysis.loader import RunResult, group_by_config


_BOOTSTRAP_ITERATIONS = 2_000
_CI_LEVEL = 0.95


def _align_trajectories(runs: list[RunResult], n_experiments: int) -> np.ndarray:
    """Stack score trajectories into a 2-D array of shape (n_runs, n_experiments).

    Short trajectories are padded on the right with their final score.
    Long trajectories are truncated to n_experiments.

    Args:
        runs: List of RunResult objects.
        n_experiments: Target trajectory length.

    Returns:
        Array of shape (len(runs), n_experiments).
    """
    matrix = np.zeros((len(runs), n_experiments), dtype=float)
    for i, run in enumerate(runs):
        scores = run.scores[:n_experiments]
        n_actual = len(scores)
        if n_actual == 0:
            continue
        matrix[i, :n_actual] = scores
        if n_actual < n_experiments:
            matrix[i, n_actual:] = scores[-1]
    return matrix


def _bootstrap_mean_ci(
    matrix: np.ndarray,
    n_iterations: int = _BOOTSTRAP_ITERATIONS,
    ci_level: float = _CI_LEVEL,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pointwise bootstrap mean and CI bands.

    Args:
        matrix: Array of shape (n_runs, n_experiments).
        n_iterations: Number of bootstrap resamples.
        ci_level: Confidence level.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        Tuple of (mean, lower, upper), each an array of shape (n_experiments,).
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    n_runs = matrix.shape[0]
    bootstrap_means = np.zeros((n_iterations, matrix.shape[1]), dtype=float)

    for i in range(n_iterations):
        indices = rng.integers(0, n_runs, size=n_runs)
        bootstrap_means[i] = matrix[indices].mean(axis=0)

    alpha = 1.0 - ci_level
    lower = np.percentile(bootstrap_means, 100.0 * alpha / 2.0, axis=0)
    upper = np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0), axis=0)
    mean = matrix.mean(axis=0)
    return mean, lower, upper


# Color palette suitable for publication (colorblind-friendly).
_CONFIG_COLORS = {
    "raw": "#E69F00",
    "greedy": "#56B4E9",
    "control": "#009E73",
    "treatment": "#CC79A7",
}
_FALLBACK_COLORS = ["#0072B2", "#D55E00", "#F0E442", "#000000"]


def _config_color(config: str, used: list[str]) -> str:
    """Return a consistent color for a configuration name."""
    if config in _CONFIG_COLORS:
        return _CONFIG_COLORS[config]
    idx = len(used) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]


def plot_convergence(
    results: list[RunResult],
    target_id: str,
    output_path: Path,
    configs: list[str] | None = None,
    n_experiments: int = 50,
) -> None:
    """Generate and save a convergence curve plot for one benchmark target.

    X-axis: experiment number (1 to n_experiments).
    Y-axis: score (averaged across seeds).
    Each configuration is rendered as a solid line with a shaded 95%
    bootstrap CI band.

    Args:
        results: All run results (filtered internally to target_id).
        target_id: Benchmark target to plot (e.g. "B1").
        output_path: Destination path for the PNG file.
        configs: Optional list of config names to include.  Defaults to all
            configs present in the data for the given target.
        n_experiments: Number of experiment steps on the x-axis.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If no results exist for target_id.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for convergence plots. "
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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(1, n_experiments + 1)
    used_fallback: list[str] = []

    for config_name, runs in sorted(by_config.items()):
        color = _config_color(config_name, used_fallback)
        if config_name not in _CONFIG_COLORS:
            used_fallback.append(config_name)

        matrix = _align_trajectories(runs, n_experiments)
        mean, lower, upper = _bootstrap_mean_ci(matrix)

        ax.plot(x, mean, label=config_name, color=color, linewidth=1.8)
        ax.fill_between(x, lower, upper, alpha=0.20, color=color)

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Convergence — Target {target_id}", fontsize=13)
    ax.set_xlim(1, n_experiments)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
