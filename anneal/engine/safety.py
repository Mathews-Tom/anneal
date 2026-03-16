"""Pre-experiment safety checks: cost estimation, budget enforcement, disk space."""

from __future__ import annotations

import shutil
from pathlib import Path

from anneal.engine.types import CostEstimate, EvalMode, OptimizationTarget

# Conservative pricing (Sonnet-class, USD per token)
_INPUT_PRICE_PER_TOKEN = 3.0 / 1_000_000  # $3/MTok
_OUTPUT_PRICE_PER_TOKEN = 15.0 / 1_000_000  # $15/MTok

# Stochastic eval token estimates (conservative)
_GEN_INPUT_TOKENS = 2000
_GEN_OUTPUT_TOKENS = 1000
_SCORE_INPUT_TOKENS = 500
_SCORE_OUTPUT_TOKENS = 10


def estimate_experiment_cost(
    target: OptimizationTarget,
    context_tokens: int = 0,
) -> CostEstimate:
    """Conservative cost estimate for one experiment cycle."""
    context_cost_usd = context_tokens * _INPUT_PRICE_PER_TOKEN

    # Mutation cost: use agent's max_budget_usd as ceiling
    mutation_cost_usd = target.agent_config.max_budget_usd

    eval_input_tokens = 0.0
    eval_cost_usd = 0.0

    if target.eval_mode == EvalMode.STOCHASTIC and target.eval_config.stochastic is not None:
        stochastic = target.eval_config.stochastic
        n = stochastic.sample_count
        k = len(stochastic.criteria)

        gen_cost = (
            _GEN_INPUT_TOKENS * _INPUT_PRICE_PER_TOKEN
            + _GEN_OUTPUT_TOKENS * _OUTPUT_PRICE_PER_TOKEN
        )
        score_cost = (
            _SCORE_INPUT_TOKENS * _INPUT_PRICE_PER_TOKEN
            + _SCORE_OUTPUT_TOKENS * _OUTPUT_PRICE_PER_TOKEN
        )

        eval_cost_usd = n * (gen_cost + k * score_cost)
        eval_input_tokens = n * (_GEN_INPUT_TOKENS + k * _SCORE_INPUT_TOKENS)

    total_usd = context_cost_usd + mutation_cost_usd + eval_cost_usd

    return CostEstimate(
        context_input_tokens=float(context_tokens),
        generation_output_tokens=float(
            _GEN_OUTPUT_TOKENS * target.eval_config.stochastic.sample_count
            if target.eval_mode == EvalMode.STOCHASTIC
            and target.eval_config.stochastic is not None
            else 0
        ),
        eval_input_tokens=eval_input_tokens,
        total_usd=total_usd,
    )


def check_budget(
    target: OptimizationTarget,
    estimated_cost: CostEstimate,
) -> bool:
    """Return True if budget allows another experiment. False -> PAUSED."""
    if target.budget_cap is None:
        return True
    return (
        target.budget_cap.cumulative_usd_spent + estimated_cost.total_usd
        <= target.budget_cap.max_usd_per_day
    )


def check_disk_space(
    path: Path,
    min_free_bytes: int = 500 * 1024 * 1024,
) -> bool:
    """Return True if sufficient disk space. False -> PAUSED."""
    return shutil.disk_usage(path).free >= min_free_bytes


def pre_experiment_check(
    target: OptimizationTarget,
    worktree_path: Path,
    context_tokens: int = 0,
) -> tuple[bool, str]:
    """Run all pre-experiment safety checks.

    Returns (safe_to_proceed, reason_if_not).
    """
    if not check_disk_space(worktree_path):
        return False, f"Insufficient disk space at {worktree_path} (< 500MB free)"

    estimate = estimate_experiment_cost(target, context_tokens)
    if not check_budget(target, estimate):
        cap = target.budget_cap
        assert cap is not None  # check_budget returns True when cap is None
        return False, (
            f"Budget cap exceeded: "
            f"cumulative ${cap.cumulative_usd_spent:.4f} + "
            f"estimated ${estimate.total_usd:.4f} > "
            f"cap ${cap.max_usd_per_day:.4f}"
        )

    return True, ""
