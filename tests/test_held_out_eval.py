"""Tests for held-out evaluation set (F1): StochasticEvaluator.evaluate_held_out and EvalEngine.evaluate_held_out."""

from __future__ import annotations

from pathlib import Path

import pytest

from anneal.engine.eval import EvalEngine, EvalError, StochasticEvaluator
from anneal.engine.types import (
    BinaryCriterion,
    Direction,
    EvalConfig,
    StochasticEval,
)


def _make_stochastic_config(
    *,
    held_out_prompts: list[str] | None = None,
    test_prompts: list[str] | None = None,
) -> StochasticEval:
    return StochasticEval(
        sample_count=5,
        criteria=[BinaryCriterion(name="relevant", question="Is it relevant?")],
        test_prompts=test_prompts or ["prompt 1"],
        generation_prompt_template="Generate: {test_prompt} {artifact_content}",
        output_format="text",
        held_out_prompts=held_out_prompts or [],
    )


def _make_eval_config(
    *,
    held_out_prompts: list[str] | None = None,
    stochastic: StochasticEval | None = None,
) -> EvalConfig:
    if stochastic is None:
        stochastic = _make_stochastic_config(held_out_prompts=held_out_prompts)
    return EvalConfig(
        metric_name="accuracy",
        direction=Direction.HIGHER_IS_BETTER,
        stochastic=stochastic,
    )


class TestStochasticEvaluatorHeldOut:
    """Tests for StochasticEvaluator.evaluate_held_out structural logic."""

    @pytest.mark.asyncio
    async def test_raises_when_no_held_out_prompts(self, tmp_path: Path) -> None:
        evaluator = StochasticEvaluator()
        config = _make_stochastic_config(held_out_prompts=[])
        with pytest.raises(EvalError, match="No held_out_prompts configured"):
            await evaluator.evaluate_held_out(tmp_path, config, "artifact text")

    @pytest.mark.asyncio
    async def test_raises_when_held_out_prompts_none(self, tmp_path: Path) -> None:
        config = _make_stochastic_config()
        # Default is empty list, which is falsy
        assert not config.held_out_prompts
        evaluator = StochasticEvaluator()
        with pytest.raises(EvalError, match="No held_out_prompts configured"):
            await evaluator.evaluate_held_out(tmp_path, config, "artifact text")


class TestEvalEngineHeldOut:
    """Tests for EvalEngine.evaluate_held_out dispatch logic."""

    @pytest.mark.asyncio
    async def test_raises_when_no_stochastic_config(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        eval_config = EvalConfig(
            metric_name="accuracy",
            direction=Direction.HIGHER_IS_BETTER,
        )
        with pytest.raises(EvalError, match="Held-out evaluation requires stochastic config"):
            await engine.evaluate_held_out(tmp_path, eval_config, "artifact text")

    @pytest.mark.asyncio
    async def test_raises_when_no_held_out_prompts_in_stochastic(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        eval_config = _make_eval_config(held_out_prompts=[])
        with pytest.raises(EvalError, match="No held_out_prompts configured"):
            await engine.evaluate_held_out(tmp_path, eval_config, "artifact text")

    @pytest.mark.asyncio
    async def test_raises_when_stochastic_none(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        eval_config = EvalConfig(
            metric_name="accuracy",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=None,
        )
        with pytest.raises(EvalError, match="Held-out evaluation requires stochastic config"):
            await engine.evaluate_held_out(tmp_path, eval_config, "artifact text")


class TestDivergenceThresholds:
    """Tests for held-out divergence threshold constants and logic."""

    def test_warning_threshold_value(self) -> None:
        from anneal.engine.runner import DIVERGENCE_WARNING
        assert DIVERGENCE_WARNING == 0.10

    def test_critical_threshold_value(self) -> None:
        from anneal.engine.runner import DIVERGENCE_CRITICAL
        assert DIVERGENCE_CRITICAL == 0.25

    def test_warning_at_10_percent_above_warning_below_critical(self) -> None:
        from anneal.engine.runner import DIVERGENCE_CRITICAL, DIVERGENCE_WARNING
        # 15% divergence: above warning, below critical
        divergence = abs(0.85 - 1.0) / 1.0  # = 0.15
        assert divergence > DIVERGENCE_WARNING
        assert divergence < DIVERGENCE_CRITICAL

    def test_critical_at_25_percent_above_critical(self) -> None:
        from anneal.engine.runner import DIVERGENCE_CRITICAL
        # 30% divergence: above critical
        divergence = abs(0.70 - 1.0) / 1.0  # = 0.30
        assert divergence > DIVERGENCE_CRITICAL
