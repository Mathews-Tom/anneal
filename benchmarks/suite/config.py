"""Benchmark configuration dataclasses.

Defines the data model for benchmark targets, per-target configurations,
and individual run descriptors used by the suite runner.

Usage: uv run python benchmarks/suite/run_suite.py --dry-run
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkTarget:
    """Describes a single optimization target registered with anneal.

    Attributes:
        id: Short identifier, e.g. "B1".
        name: Snake-case name, e.g. "classification_prompt".
        domain: High-level domain — "prompt", "code", or "documentation".
        eval_mode: "stochastic" or "deterministic".
        criteria_count: Number of binary criteria (stochastic targets only).
        experiment_budget: Maximum experiments per run.
        artifact_path: Repo-relative path to the artifact under optimization.
        scope_path: Repo-relative path to the scope YAML file.
        criteria_path: Repo-relative path to the criteria TOML (stochastic only).
        run_cmd: Shell command to execute the artifact (deterministic only).
        parse_cmd: Shell command to extract the metric value (deterministic only).
        direction: "maximize" or "minimize".
    """

    id: str
    name: str
    domain: str
    eval_mode: str
    criteria_count: int
    experiment_budget: int
    artifact_path: str
    scope_path: str
    criteria_path: str | None = None
    run_cmd: str | None = None
    parse_cmd: str | None = None
    direction: str = "maximize"


@dataclass
class BenchmarkConfig:
    """One of the four experimental configurations applied to each target.

    Attributes:
        name: Short identifier — "raw", "greedy", "control", or "treatment".
        description: Human-readable description of what is enabled.
        search_strategy: anneal --search flag value, or "none" for raw baseline.
        enhancements: Mapping of enhancement name to enabled flag.
    """

    name: str
    description: str
    search_strategy: str
    enhancements: dict[str, bool] = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    """Fully-specified descriptor for a single benchmark execution.

    Attributes:
        target: The optimization target being benchmarked.
        config: The configuration variant being applied.
        seed: Random seed for reproducibility.
        output_dir: Directory where result JSONL files are written.
    """

    target: BenchmarkTarget
    config: BenchmarkConfig
    seed: int
    output_dir: Path

    @property
    def run_id(self) -> str:
        """Canonical run identifier: e.g. "B1-treatment-seed3"."""
        return f"{self.target.id}-{self.config.name}-seed{self.seed}"

    @property
    def target_name(self) -> str:
        """Fully-qualified anneal target name: e.g. "B1-treatment"."""
        return f"{self.target.id}-{self.config.name}"

    @property
    def result_path(self) -> Path:
        """JSONL file path for this run's results."""
        return self.output_dir / f"{self.run_id}.jsonl"
