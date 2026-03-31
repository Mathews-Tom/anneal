from __future__ import annotations

import random
from collections import Counter

from anneal.engine.strategy_selector import AgentSelector


def test_agent_selector_returns_valid_names() -> None:
    random.seed(42)
    selector = AgentSelector()
    result = selector.select(0.5)
    assert result in {"primary", "exploration"}


def test_agent_selector_early_favors_exploration() -> None:
    random.seed(42)
    selector = AgentSelector()
    counts = Counter(selector.select(0.1) for _ in range(200))
    assert counts["exploration"] > 100


def test_agent_selector_late_favors_primary() -> None:
    random.seed(42)
    selector = AgentSelector()
    counts = Counter(selector.select(0.9) for _ in range(200))
    assert counts["primary"] > 120


def test_agent_selector_mid_range_balanced() -> None:
    random.seed(42)
    selector = AgentSelector()
    counts = Counter(selector.select(0.5) for _ in range(200))
    assert counts["primary"] > 50
    assert counts["exploration"] > 50


def test_agent_selector_update_reward_shifts_distribution() -> None:
    random.seed(42)

    # Fresh selector baseline
    baseline = AgentSelector()
    baseline_counts = Counter(baseline.select(0.5) for _ in range(200))

    # Trained selector with exploration rewards
    random.seed(42)
    trained = AgentSelector()
    for _ in range(20):
        trained.update("exploration", improved=True)
    trained_counts = Counter(trained.select(0.5) for _ in range(200))

    assert trained_counts["exploration"] > baseline_counts["exploration"]


def test_agent_selector_summary_returns_means() -> None:
    selector = AgentSelector()
    summary = selector.summary()
    assert "primary" in summary
    assert "exploration" in summary
    assert 0.0 <= summary["primary"]["mean"] <= 1.0
    assert 0.0 <= summary["exploration"]["mean"] <= 1.0


def test_agent_selector_update_primary() -> None:
    selector = AgentSelector()
    initial_alpha = selector._primary.alpha
    selector.update("primary", True)
    assert selector._primary.alpha == initial_alpha + 1.0
