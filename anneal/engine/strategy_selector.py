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
