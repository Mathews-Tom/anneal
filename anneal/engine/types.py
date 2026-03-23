"""Frozen interface contracts for the Anneal engine.

These types are the communication boundaries between engine components.
They must be locked before parallel development begins. Changes require
coordinated updates across all consumers.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field


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
# Configuration models
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Configuration for the mutation agent."""

    mode: Literal["claude_code", "api"]
    model: str
    evaluator_model: str
    max_budget_usd: float = Field(default=0.10, gt=0)
    max_context_tokens: int = Field(default=80_000, gt=0)
    temperature: float = Field(default=0.7, ge=0, le=2.0)
    sandbox: bool = False


class DeterministicEval(BaseModel):
    """Configuration for deterministic evaluation."""

    run_command: str
    parse_command: str
    timeout_seconds: int = Field(gt=0)
    max_retries: int = Field(default=1, ge=1)
    retry_delay_seconds: float = Field(default=5.0, ge=0)
    flake_detection: bool = False


class BinaryCriterion(BaseModel):
    """A single binary evaluation criterion for stochastic eval."""

    name: str
    question: str


class StochasticEval(BaseModel):
    """Configuration for stochastic evaluation."""

    sample_count: int = Field(gt=0)
    criteria: list[BinaryCriterion]
    test_prompts: list[str]
    generation_prompt_template: str
    output_format: str
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    generation_agent_config: AgentConfig | None = None
    held_out_prompts: list[str] = Field(default_factory=list)
    min_criterion_scores: dict[str, float] = Field(default_factory=dict)
    judgment_votes: int = Field(default=3, gt=0)
    comparison_mode: str = "majority_vote"


class MetricConstraint(BaseModel):
    """A constraint that must be satisfied for an experiment to be KEPT."""

    metric_name: str
    threshold: float
    direction: Direction


class ConstraintCommand(BaseModel):
    """A secondary deterministic eval command used as a constraint."""

    name: str
    run_command: str
    parse_command: str
    timeout_seconds: int = Field(gt=0)
    threshold: float
    direction: Direction


class FidelityStage(BaseModel):
    """A stage in a multi-fidelity evaluation pipeline."""

    name: str
    run_command: str
    parse_command: str
    timeout_seconds: int = Field(default=30, gt=0)
    min_pass_score: float = 0.0


class EvalConfig(BaseModel):
    """Evaluation configuration for a target."""

    metric_name: str
    direction: Direction
    min_improvement_threshold: float = Field(default=0.0, ge=0)
    deterministic: DeterministicEval | None = None
    stochastic: StochasticEval | None = None
    held_out_interval: int = Field(default=10, gt=0)
    constraints: list[MetricConstraint] = Field(default_factory=list)
    constraint_commands: list[ConstraintCommand] = Field(default_factory=list)
    fidelity_stages: list[FidelityStage] = Field(default_factory=list)


class BudgetCap(BaseModel):
    """Budget enforcement configuration."""

    max_usd_per_day: float = Field(gt=0)
    cumulative_usd_spent: float = Field(default=0.0, ge=0)


class NotificationConfig(BaseModel):
    """Notification hooks configuration."""

    webhook_url: str | None = None
    fallback_webhook_url: str | None = None
    status_file: str = ".anneal-status"
    notify_on: list[str] = Field(default_factory=lambda: ["PAUSED", "HALTED"])
    milestone_interval: int = Field(default=10, gt=0)
    webhook_retry_count: int = Field(default=3, ge=0)
    webhook_retry_delay_seconds: float = Field(default=5.0, ge=0)


class ScopeConfig(BaseModel):
    """Parsed scope.yaml content."""

    editable: list[str]
    immutable: list[str]
    watch: list[str] = Field(default_factory=list)
    allowed_deletions: list[str] = Field(default_factory=list)
    constraints: list[dict[str, str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PopulationConfig (must be defined before OptimizationTarget)
# ---------------------------------------------------------------------------


class PopulationConfig(BaseModel):
    """Configuration for population-based search strategy."""

    population_size: int = Field(default=4, gt=0)
    tournament_size: int = Field(default=2, gt=0)


# ---------------------------------------------------------------------------
# Eval environment (cloud/remote eval lifecycle)
# ---------------------------------------------------------------------------


class EvalEnvironment(BaseModel):
    """Environment requirements for eval commands (cloud/remote targets)."""

    requires_network: bool = False
    env_vars: list[str] = Field(default_factory=list)
    setup_command: str | None = None
    teardown_command: str | None = None


# ---------------------------------------------------------------------------
# OptimizationTarget — the full target registration record
# ---------------------------------------------------------------------------


class OptimizationTarget(BaseModel):
    """Complete configuration for one optimization target."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    domain_tier: DomainTier
    artifact_paths: list[str] = Field(min_length=1)
    scope_path: str
    scope_hash: str
    eval_mode: EvalMode
    eval_config: EvalConfig
    agent_config: AgentConfig
    time_budget_seconds: int = Field(gt=0)
    loop_interval_seconds: int = Field(ge=0)
    knowledge_path: str
    worktree_path: str
    git_branch: str
    baseline_score: float
    baseline_raw_scores: list[float] = Field(default_factory=list)
    max_consecutive_failures: int = Field(default=5, gt=0)
    budget_cap: BudgetCap | None = None
    meta_depth: int = Field(default=0, ge=0)
    inject_knowledge_context: bool = False
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    approval_callback: Callable[[str], bool] | None = Field(default=None, exclude=True)
    population_config: PopulationConfig | None = None
    eval_environment: EvalEnvironment | None = None


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
    criterion_names: list[str] | None = None
    per_criterion_scores: dict[str, float] | None = None


class AgentInvocationResult(BaseModel):
    """Returned by agent invoker, consumed by runner and cost tracker."""

    success: bool
    cost_usd: float
    input_tokens: int
    output_tokens: int
    hypothesis: str | None
    hypothesis_source: Literal["agent", "synthesized"]
    tags: list[str]
    raw_output: str


class ScopeViolationResult(BaseModel):
    """Returned by scope enforcer, consumed by runner."""

    model_config = ConfigDict(frozen=True)

    has_violations: bool
    violated_paths: list[str]
    valid_paths: list[str]
    all_blocked: bool


class ExperimentRecord(BaseModel):
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
    agent_model: str = ""
    held_out_score: float | None = None
    criterion_names: list[str] | None = None
    per_criterion_scores: dict[str, float] | None = None


class CostEstimate(BaseModel):
    """Pre-experiment cost estimate for budget enforcement."""

    context_input_tokens: float
    generation_output_tokens: float
    eval_input_tokens: float
    total_usd: float


class ConsolidationRecord(BaseModel):
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
    criterion_variances: dict[str, float] = Field(default_factory=dict)
    score_variance: float = 0.0


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------


class DriftEntry(BaseModel):
    """A single criterion exhibiting evaluator drift."""

    criterion_name: str
    variance: float
    mean_score: float
    window_size: int


# ---------------------------------------------------------------------------
# Git environment types
# ---------------------------------------------------------------------------


class WorktreeInfo(BaseModel):
    """Information about a git worktree."""

    model_config = ConfigDict(frozen=True)

    path: Path
    branch: str
    head_sha: str
