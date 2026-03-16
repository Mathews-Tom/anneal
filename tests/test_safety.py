"""Tests for anneal.engine.safety — cost estimation and budget checks."""

from __future__ import annotations

from pathlib import Path

from anneal.engine.safety import (
    check_budget,
    check_disk_space,
    estimate_experiment_cost,
    pre_experiment_check,
)
from anneal.engine.types import (
    AgentConfig,
    BinaryCriterion,
    BudgetCap,
    CostEstimate,
    Direction,
    EvalConfig,
    EvalMode,
    OptimizationTarget,
    StochasticEval,
)


def _make_target(
    *,
    eval_mode: EvalMode = EvalMode.DETERMINISTIC,
    budget_cap: BudgetCap | None = None,
    max_budget_usd: float = 0.10,
    stochastic: StochasticEval | None = None,
) -> OptimizationTarget:
    agent = AgentConfig(
        mode="api",
        model="gpt-4.1",
        evaluator_model="gpt-4.1-mini",
        max_budget_usd=max_budget_usd,
    )
    eval_config = EvalConfig(
        metric_name="accuracy",
        direction=Direction.HIGHER_IS_BETTER,
        stochastic=stochastic,
    )
    return OptimizationTarget(
        id="target-test",
        domain_tier="sandbox",
        artifact_paths=["src/main.py"],
        scope_path="scope.yaml",
        scope_hash="abc123",
        eval_mode=eval_mode,
        eval_config=eval_config,
        agent_config=agent,
        time_budget_seconds=3600,
        loop_interval_seconds=30,
        knowledge_path="knowledge/",
        worktree_path="worktree/",
        git_branch="anneal/target-test",
        baseline_score=0.75,
        budget_cap=budget_cap,
    )


def _make_stochastic_target(
    *,
    budget_cap: BudgetCap | None = None,
) -> OptimizationTarget:
    stochastic = StochasticEval(
        sample_count=10,
        criteria=[
            BinaryCriterion(name="relevant", question="Is it relevant?"),
            BinaryCriterion(name="coherent", question="Is it coherent?"),
        ],
        test_prompts=["test prompt"],
        generation_prompt_template="Generate: {input}",
        output_format="text",
    )
    return _make_target(
        eval_mode=EvalMode.STOCHASTIC,
        stochastic=stochastic,
        budget_cap=budget_cap,
    )


def test_estimate_cost_deterministic_positive(tmp_path: Path) -> None:
    target = _make_target()
    estimate = estimate_experiment_cost(target, context_tokens=1000)
    assert isinstance(estimate, CostEstimate)
    assert estimate.total_usd > 0


def test_estimate_cost_stochastic_includes_eval(tmp_path: Path) -> None:
    target = _make_stochastic_target()
    estimate = estimate_experiment_cost(target, context_tokens=0)
    assert estimate.total_usd > 0
    assert estimate.eval_input_tokens > 0
    assert estimate.generation_output_tokens > 0


def test_check_budget_under_cap() -> None:
    cap = BudgetCap(max_usd_per_day=10.0, cumulative_usd_spent=1.0)
    target = _make_target(budget_cap=cap)
    estimate = CostEstimate(
        context_input_tokens=0,
        generation_output_tokens=0,
        eval_input_tokens=0,
        total_usd=0.50,
    )
    assert check_budget(target, estimate) is True


def test_check_budget_over_cap() -> None:
    cap = BudgetCap(max_usd_per_day=1.0, cumulative_usd_spent=0.90)
    target = _make_target(budget_cap=cap)
    estimate = CostEstimate(
        context_input_tokens=0,
        generation_output_tokens=0,
        eval_input_tokens=0,
        total_usd=0.20,
    )
    assert check_budget(target, estimate) is False


def test_check_budget_no_cap() -> None:
    target = _make_target(budget_cap=None)
    estimate = CostEstimate(
        context_input_tokens=0,
        generation_output_tokens=0,
        eval_input_tokens=0,
        total_usd=999.0,
    )
    assert check_budget(target, estimate) is True


def test_check_disk_space_tmp(tmp_path: Path) -> None:
    assert check_disk_space(tmp_path) is True


def test_pre_experiment_check_safe(tmp_path: Path) -> None:
    target = _make_target(budget_cap=BudgetCap(max_usd_per_day=100.0))
    safe, reason = pre_experiment_check(target, tmp_path, context_tokens=0)
    assert safe is True
    assert reason == ""


def test_pre_experiment_check_budget_exceeded(tmp_path: Path) -> None:
    cap = BudgetCap(max_usd_per_day=0.01, cumulative_usd_spent=0.009)
    target = _make_target(budget_cap=cap)
    safe, reason = pre_experiment_check(target, tmp_path, context_tokens=0)
    assert safe is False
    assert "Budget cap exceeded" in reason
