"""Frozen interface contracts for the Anneal engine.

These types are the communication boundaries between engine components.
They must be locked before parallel development begins. Changes require
coordinated updates across all consumers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, NamedTuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DomainTier(Enum):
    """Target domain classification."""

    SANDBOX = "sandbox"
    DEPLOYMENT = "deployment"


class EvalMode(Enum):
    """Evaluation strategy for a target."""

    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


class Direction(Enum):
    """Optimization direction."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class Outcome(Enum):
    """Experiment outcome classification."""

    KEPT = "KEPT"
    DISCARDED = "DISCARDED"
    BLOCKED = "BLOCKED"
    KILLED = "KILLED"
    CRASHED = "CRASHED"


class RunnerState(Enum):
    """Runner state machine states."""

    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    KILLED = "KILLED"
    PAUSED = "PAUSED"
    HALTED = "HALTED"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Configuration for the mutation agent."""

    mode: Literal["claude_code", "api"]
    model: str
    evaluator_model: str
    max_budget_usd: float = 0.10
    max_context_tokens: int = 80_000
    temperature: float = 0.7
    sandbox: bool = False


@dataclass
class DeterministicEval:
    """Configuration for deterministic evaluation."""

    run_command: str
    parse_command: str
    timeout_seconds: int


@dataclass
class BinaryCriterion:
    """A single binary evaluation criterion for stochastic eval."""

    name: str
    question: str


@dataclass
class StochasticEval:
    """Configuration for stochastic evaluation."""

    sample_count: int
    criteria: list[BinaryCriterion]
    test_prompts: list[str]
    generation_prompt_template: str
    output_format: str
    confidence_level: float = 0.95
    generation_agent_config: AgentConfig | None = None
    held_out_prompts: list[str] = field(default_factory=list)
    min_criterion_scores: dict[str, float] = field(default_factory=dict)
    judgment_votes: int = 3


@dataclass
class MetricConstraint:
    """A constraint that must be satisfied for an experiment to be KEPT."""

    metric_name: str
    threshold: float
    direction: Direction


@dataclass
class ConstraintCommand:
    """A secondary deterministic eval command used as a constraint."""

    name: str
    run_command: str
    parse_command: str
    timeout_seconds: int
    threshold: float
    direction: Direction


@dataclass
class EvalConfig:
    """Evaluation configuration for a target."""

    metric_name: str
    direction: Direction
    min_improvement_threshold: float = 0.0
    deterministic: DeterministicEval | None = None
    stochastic: StochasticEval | None = None
    held_out_interval: int = 10
    constraints: list[MetricConstraint] = field(default_factory=list)
    constraint_commands: list[ConstraintCommand] = field(default_factory=list)


@dataclass
class BudgetCap:
    """Budget enforcement configuration."""

    max_usd_per_day: float
    cumulative_usd_spent: float = 0.0


@dataclass
class NotificationConfig:
    """Notification hooks configuration."""

    webhook_url: str | None = None
    fallback_webhook_url: str | None = None
    status_file: str = ".anneal-status"
    notify_on: list[str] = field(default_factory=lambda: ["PAUSED", "HALTED"])
    milestone_interval: int = 10
    webhook_retry_count: int = 3
    webhook_retry_delay_seconds: float = 5.0


@dataclass
class ScopeConfig:
    """Parsed scope.yaml content."""

    editable: list[str]
    immutable: list[str]
    watch: list[str] = field(default_factory=list)
    allowed_deletions: list[str] = field(default_factory=list)
    constraints: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OptimizationTarget — the full target registration record
# ---------------------------------------------------------------------------


@dataclass
class OptimizationTarget:
    """Complete configuration for one optimization target."""

    id: str
    domain_tier: DomainTier
    artifact_paths: list[str]
    scope_path: str
    scope_hash: str
    eval_mode: EvalMode
    eval_config: EvalConfig
    agent_config: AgentConfig
    time_budget_seconds: int
    loop_interval_seconds: int
    knowledge_path: str
    worktree_path: str
    git_branch: str
    baseline_score: float
    baseline_raw_scores: list[float] = field(default_factory=list)
    max_consecutive_failures: int = 5
    budget_cap: BudgetCap | None = None
    meta_depth: int = 0
    inject_knowledge_context: bool = False
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    approval_callback: Callable[[str], bool] | None = None
    population_config: PopulationConfig | None = None


# ---------------------------------------------------------------------------
# Inter-component result types (frozen contracts)
# ---------------------------------------------------------------------------


class EvalResult(NamedTuple):
    """Returned by eval engine, consumed by runner and search strategy."""

    score: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    raw_scores: list[float] | None = None
    cost_usd: float = 0.0


@dataclass
class AgentInvocationResult:
    """Returned by agent invoker, consumed by runner and cost tracker."""

    success: bool
    cost_usd: float
    input_tokens: int
    output_tokens: int
    hypothesis: str | None
    hypothesis_source: Literal["agent", "synthesized"]
    tags: list[str]
    raw_output: str


@dataclass(frozen=True)
class ScopeViolationResult:
    """Returned by scope enforcer, consumed by runner."""

    has_violations: bool
    violated_paths: list[str]
    valid_paths: list[str]
    all_blocked: bool


@dataclass
class ExperimentRecord:
    """One experiment's complete record. Written by runner, read by
    knowledge store, cost tracker, and consolidation."""

    id: str
    target_id: str
    git_sha: str
    pre_experiment_sha: str
    timestamp: datetime
    hypothesis: str
    hypothesis_source: Literal["agent", "synthesized"]
    mutation_diff_summary: str
    score: float
    score_ci_lower: float | None
    score_ci_upper: float | None
    raw_scores: list[float] | None
    baseline_score: float
    outcome: Outcome
    failure_mode: str | None
    duration_seconds: float
    tags: list[str]
    learnings: str
    cost_usd: float
    bootstrap_seed: int
    held_out_score: float | None = None


@dataclass
class CostEstimate:
    """Pre-experiment cost estimate for budget enforcement."""

    context_input_tokens: float
    generation_output_tokens: float
    eval_input_tokens: float
    total_usd: float


@dataclass
class ConsolidationRecord:
    """Structured summary produced every 50 experiments."""

    experiment_range: tuple[int, int]
    timestamp: datetime
    total_experiments: int
    kept_count: int
    discarded_count: int
    crashed_count: int
    score_start: float
    score_end: float
    top_improvements: list[dict[str, str | float]]
    failed_approaches: list[dict[str, str | float]]
    tags_frequency: dict[str, int]
    criterion_variances: dict[str, float] = field(default_factory=dict)
    score_variance: float = 0.0


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------


@dataclass
class DriftEntry:
    """A single criterion exhibiting evaluator drift."""

    criterion_name: str
    variance: float
    mean_score: float
    window_size: int


# ---------------------------------------------------------------------------
# Population-based search
# ---------------------------------------------------------------------------


@dataclass
class PopulationConfig:
    """Configuration for population-based search strategy."""

    population_size: int = 4
    tournament_size: int = 2


# ---------------------------------------------------------------------------
# Git environment types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorktreeInfo:
    """Information about a git worktree."""

    path: Path
    branch: str
    head_sha: str
