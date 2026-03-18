"""Standardized plotting for experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments._harness.types import ResultRecord

COLORS = ["#58a6ff", "#3fb950", "#d29922", "#f778ba", "#bc8cff", "#79c0ff"]


def plot_trajectory(
    results: dict[str, list[ResultRecord]],
    path: Path,
    title: str = "Score Trajectory",
) -> None:
    """Plot score vs experiment index, one series per condition."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for i, (condition, records) in enumerate(sorted(results.items())):
        color = COLORS[i % len(COLORS)]
        scores = [r.score for r in records]
        indices = list(range(1, len(scores) + 1))

        ax.plot(indices, scores, color=color, linewidth=1.5, label=condition, alpha=0.9)

        # CI band if available
        ci_lowers = [r.ci_lower for r in records if r.ci_lower is not None]
        ci_uppers = [r.ci_upper for r in records if r.ci_upper is not None]
        if len(ci_lowers) == len(scores):
            ax.fill_between(indices, ci_lowers, ci_uppers, color=color, alpha=0.15)

    ax.set_xlabel("Experiment", color="#8b949e")
    ax.set_ylabel("Score", color="#8b949e")
    ax.set_title(title, color="#c9d1d9", fontsize=12)
    ax.tick_params(colors="#484f58")
    ax.spines["bottom"].set_color("#21262d")
    ax.spines["left"].set_color("#21262d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#161b22", edgecolor="#21262d", labelcolor="#c9d1d9")
    ax.grid(axis="y", color="#21262d", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_variance(
    scores_per_run: dict[str, list[float]],
    path: Path,
    title: str = "Score Variance Across Runs",
) -> None:
    """Box plot showing score distribution per criterion or run."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    labels = sorted(scores_per_run.keys())
    data = [scores_per_run[k] for k in labels]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "#58a6ff", "alpha": 0.3, "edgecolor": "#58a6ff"},
        whiskerprops={"color": "#484f58"},
        capprops={"color": "#484f58"},
        medianprops={"color": "#3fb950", "linewidth": 2},
        flierprops={"markerfacecolor": "#f85149", "markeredgecolor": "#f85149", "markersize": 4},
    )

    ax.set_ylabel("Score", color="#8b949e")
    ax.set_title(title, color="#c9d1d9", fontsize=12)
    ax.tick_params(colors="#484f58")
    ax.spines["bottom"].set_color("#21262d")
    ax.spines["left"].set_color("#21262d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#21262d", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_domain_comparison(
    domain_results: dict[str, dict[str, float]],
    path: Path,
    title: str = "Cross-Domain Comparison",
) -> None:
    """Grouped bar chart comparing metrics across domains."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    domains = sorted(domain_results.keys())
    if not domains:
        plt.close(fig)
        return

    # Collect all metric keys
    metric_keys: list[str] = []
    seen: set[str] = set()
    for d in domains:
        for k in domain_results[d]:
            if k not in seen:
                metric_keys.append(k)
                seen.add(k)

    x = np.arange(len(domains))
    width = 0.8 / max(len(metric_keys), 1)

    for i, metric in enumerate(metric_keys):
        color = COLORS[i % len(COLORS)]
        values = [domain_results[d].get(metric, 0.0) for d in domains]
        offset = (i - len(metric_keys) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, color="#8b949e")
    ax.set_ylabel("Value", color="#8b949e")
    ax.set_title(title, color="#c9d1d9", fontsize=12)
    ax.tick_params(colors="#484f58")
    ax.spines["bottom"].set_color("#21262d")
    ax.spines["left"].set_color("#21262d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#161b22", edgecolor="#21262d", labelcolor="#c9d1d9")
    ax.grid(axis="y", color="#21262d", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
