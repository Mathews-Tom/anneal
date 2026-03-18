"""Experiment infrastructure types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ConditionConfig:
    """One arm of an experiment (e.g., 'guided', 'random', 'memoryless')."""

    name: str
    description: str = ""
    knowledge_enabled: bool = False
    search_strategy: str = "greedy"
    agent_model: str = "sonnet"
    evaluator_model: str = "gpt-4.1"
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen configuration for a validation experiment.

    Loaded from config.toml at run start. A copy is written to results/
    for reproducibility.
    """

    name: str
    checkpoint: str
    description: str
    conditions: list[ConditionConfig]
    max_experiments_per_condition: int
    seed: int = 42
    output_dir: Path = Path("results")
    artifact_paths: list[str] = field(default_factory=list)
    eval_criteria_path: str = ""
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class ResultRecord:
    """One row of experiment output. Written to experiments.csv."""

    condition: str
    experiment_idx: int
    hypothesis: str
    score: float
    ci_lower: float | None
    ci_upper: float | None
    baseline_score: float
    kept: bool
    cost_usd: float
    duration_seconds: float
    seed: int
    raw_scores: list[float] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    per_criterion: dict[str, float] = field(default_factory=dict)
    failure_mode: str = ""
