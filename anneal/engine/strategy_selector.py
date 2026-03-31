"""Contextual bandit for adaptive search strategy selection."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class StrategyArm:
    """Beta-Binomial arm for one search strategy."""

    name: str
    alpha: float = 1.0  # Successes + prior
    beta: float = 1.0   # Failures + prior

    def sample(self) -> float:
        """Draw from Beta posterior."""
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: bool) -> None:
        if reward:
            self.alpha += 1.0
        else:
            self.beta += 1.0


class StrategySelector:
    """Thompson Sampling selector over search strategies."""

    def __init__(self, strategy_names: list[str]) -> None:
        self._arms = {name: StrategyArm(name=name) for name in strategy_names}

    def select(self) -> str:
        """Sample from posteriors, return arm with highest sample."""
        return max(self._arms.values(), key=lambda a: a.sample()).name

    def update(self, strategy_name: str, improved: bool) -> None:
        """Update the arm with reward signal."""
        arm = self._arms.get(strategy_name)
        if arm:
            arm.update(improved)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return mean reward estimates per strategy."""
        return {
            name: {"mean": arm.alpha / (arm.alpha + arm.beta), "n": arm.alpha + arm.beta - 2}
            for name, arm in self._arms.items()
        }


class AgentSelector:
    """Thompson Sampling selector for primary vs exploration agent."""

    def __init__(self) -> None:
        self._primary = StrategyArm(name="primary")
        self._exploration = StrategyArm(name="exploration")

    def select(self, experiment_ratio: float) -> str:
        """Select agent based on Thompson Sampling + experiment progress.

        experiment_ratio: current_experiment / max_experiments (0.0 to 1.0)
        Returns "primary" or "exploration".
        """
        if experiment_ratio < 0.2:
            exploration_boost = 0.7
        elif experiment_ratio > 0.8:
            exploration_boost = 0.2
        else:
            exploration_boost = 0.5

        primary_sample = self._primary.sample()
        exploration_sample = self._exploration.sample() * exploration_boost / 0.5
        return "exploration" if exploration_sample > primary_sample else "primary"

    def update(self, agent_name: str, improved: bool) -> None:
        arm = self._primary if agent_name == "primary" else self._exploration
        arm.update(improved)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            "primary": {"mean": self._primary.alpha / (self._primary.alpha + self._primary.beta)},
            "exploration": {"mean": self._exploration.alpha / (self._exploration.alpha + self._exploration.beta)},
        }
