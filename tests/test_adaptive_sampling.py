"""Tests for StochasticEvaluator adaptive sample sizing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from anneal.engine.eval import StochasticEvaluator
from anneal.engine.types import AgentConfig, BinaryCriterion, StochasticEval


def _make_stochastic_config(**overrides: object) -> StochasticEval:
    defaults: dict[str, object] = dict(
        sample_count=10,
        criteria=[BinaryCriterion(name="quality", question="Is it good?")],
        test_prompts=["test prompt"] * 10,
        generation_prompt_template="{test_prompt} {artifact_content}",
        output_format="text",
        generation_agent_config=AgentConfig(
            mode="api",
            model="test-model",
            evaluator_model="test-model",
        ),
    )
    defaults.update(overrides)
    return StochasticEval(**defaults)  # type: ignore[arg-type]


def _make_sample_side_effect(scores: list[float]) -> AsyncMock:
    """Build an AsyncMock whose successive calls return (score, 0.01, {"quality": score})."""
    call_index = 0

    async def _side_effect(*_args: object, **_kwargs: object) -> tuple[float, float, dict[str, float]]:
        nonlocal call_index
        score = scores[call_index % len(scores)]
        call_index += 1
        return score, 0.01, {"quality": score}

    mock = AsyncMock(side_effect=_side_effect)
    return mock


class TestAdaptiveSamplingDisabled:
    @pytest.mark.asyncio
    async def test_adaptive_disabled_uses_fixed_count(self, tmp_path: Path) -> None:
        # Arrange
        config = _make_stochastic_config(adaptive_sampling=False, sample_count=10)
        evaluator = StochasticEvaluator()
        scores = [0.5] * 20
        mock_sample = _make_sample_side_effect(scores)

        # Act
        with patch.object(evaluator, "_evaluate_single_sample", mock_sample):
            await evaluator.evaluate(tmp_path, config, "artifact text")

        # Assert
        assert mock_sample.call_count == 10


class TestAdaptiveEarlyStop:
    @pytest.mark.asyncio
    async def test_adaptive_early_stop_on_large_effect(self, tmp_path: Path) -> None:
        # Arrange — high mean, moderate variance → Cohen's d > 1.0
        # mean ≈ 8.17, std ≈ 0.76, d ≈ 10.7
        config = _make_stochastic_config(
            adaptive_sampling=True,
            sample_count=10,
            min_sample_count=3,
            early_stop_effect_size=1.0,
            extend_effect_size=0.3,
        )
        evaluator = StochasticEvaluator()
        high_scores = [8.0, 9.0, 7.5, 8.5, 8.0, 9.0, 7.5, 8.5, 8.0, 9.0]
        mock_sample = _make_sample_side_effect(high_scores)

        # Act
        with patch.object(evaluator, "_evaluate_single_sample", mock_sample):
            result = await evaluator.evaluate(tmp_path, config, "artifact text")

        # Assert — stops at initial batch (max(min_sample_count=3, sample_count//2=5) = 5)
        # and does NOT extend because effect size is large
        assert mock_sample.call_count <= 5
        assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_adaptive_respects_min_sample_count(self, tmp_path: Path) -> None:
        # Arrange — adaptive mode with explicit min_sample_count=3
        config = _make_stochastic_config(
            adaptive_sampling=True,
            sample_count=10,
            min_sample_count=3,
            early_stop_effect_size=1.0,
            extend_effect_size=0.3,
        )
        evaluator = StochasticEvaluator()
        # Scores still trigger early stop — but we verify at least 3 samples were collected
        mock_sample = _make_sample_side_effect([8.0, 9.0, 7.5, 8.5, 8.0])

        # Act
        with patch.object(evaluator, "_evaluate_single_sample", mock_sample):
            await evaluator.evaluate(tmp_path, config, "artifact text")

        # Assert — at least min_sample_count evaluations always run
        assert mock_sample.call_count >= 3


class TestAdaptiveExtension:
    @pytest.mark.asyncio
    async def test_adaptive_extends_on_small_effect(self, tmp_path: Path) -> None:
        # Arrange — scores near zero with tiny variance → Cohen's d < 0.3
        # mean ≈ 0.043, std ≈ 0.076, d ≈ 0.57 — but with [0.01, 0.02, 0.01]:
        # mean=0.0133, std=0.0058, d=2.3 — need truly near-zero variance
        # Use scores where d < 0.3: mean=0.05, std=0.20 → d=0.25
        config = _make_stochastic_config(
            adaptive_sampling=True,
            sample_count=10,
            min_sample_count=3,
            early_stop_effect_size=1.0,
            extend_effect_size=0.3,
        )
        evaluator = StochasticEvaluator()
        # Build scores where initial batch has small effect size: all near 0 with spread
        # initial_count = max(3, 10//2) = 5
        # Use [0.3, -0.2, 0.1, -0.3, 0.2]: mean=0.02, std=0.248, d=0.08
        small_effect_scores = [0.3, -0.2, 0.1, -0.3, 0.2] + [0.1] * 10
        mock_sample = _make_sample_side_effect(small_effect_scores)

        # Act
        with patch.object(evaluator, "_evaluate_single_sample", mock_sample):
            await evaluator.evaluate(tmp_path, config, "artifact text")

        # Assert — extended beyond initial 5 samples
        initial_count = max(config.min_sample_count, config.sample_count // 2)
        assert mock_sample.call_count > initial_count

    @pytest.mark.asyncio
    async def test_adaptive_respects_max_extension(self, tmp_path: Path) -> None:
        # Arrange — always small effect to force maximum extension
        config = _make_stochastic_config(
            adaptive_sampling=True,
            sample_count=10,
            min_sample_count=3,
            early_stop_effect_size=1.0,
            extend_effect_size=0.3,
        )
        evaluator = StochasticEvaluator()
        # Scores with perpetually tiny effect size
        small_effect_scores = [0.3, -0.2, 0.1, -0.3, 0.2] + [0.1] * 20
        mock_sample = _make_sample_side_effect(small_effect_scores)

        # Act
        with patch.object(evaluator, "_evaluate_single_sample", mock_sample):
            await evaluator.evaluate(tmp_path, config, "artifact text")

        # Assert — never exceeds sample_count * 1.5
        max_allowed = config.sample_count + config.sample_count // 2
        assert mock_sample.call_count <= max_allowed


class TestAdaptiveCostReduction:
    @pytest.mark.asyncio
    async def test_adaptive_cost_reduction(self, tmp_path: Path) -> None:
        # Arrange — large-effect scenario should cost ≤60% of fixed-sample scenario
        high_effect_scores = [8.0, 9.0, 7.5, 8.5, 8.0, 9.0, 7.5, 8.5, 8.0, 9.0]

        adaptive_config = _make_stochastic_config(
            adaptive_sampling=True,
            sample_count=10,
            min_sample_count=3,
            early_stop_effect_size=1.0,
            extend_effect_size=0.3,
        )
        fixed_config = _make_stochastic_config(
            adaptive_sampling=False,
            sample_count=10,
        )
        evaluator = StochasticEvaluator()

        # Act — adaptive run
        adaptive_mock = _make_sample_side_effect(high_effect_scores)
        with patch.object(evaluator, "_evaluate_single_sample", adaptive_mock):
            adaptive_result = await evaluator.evaluate(tmp_path, adaptive_config, "artifact text")
        adaptive_calls = adaptive_mock.call_count

        # Act — fixed run
        fixed_mock = _make_sample_side_effect(high_effect_scores)
        with patch.object(evaluator, "_evaluate_single_sample", fixed_mock):
            fixed_result = await evaluator.evaluate(tmp_path, fixed_config, "artifact text")
        fixed_calls = fixed_mock.call_count

        # Assert — adaptive uses ≤60% of samples (each call costs 0.01 USD)
        assert fixed_calls == 10
        assert adaptive_calls <= fixed_calls * 0.60
        assert adaptive_result.score > 0.0
        assert fixed_result.score > 0.0
