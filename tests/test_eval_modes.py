"""Tests for mode-agnostic eval dispatch (claude_code + API for generation and judgment)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from anneal.engine.eval import StochasticEvaluator
from anneal.engine.types import (
    AgentConfig,
    AgentInvocationResult,
    BinaryCriterion,
)


@pytest.fixture
def evaluator() -> StochasticEvaluator:
    return StochasticEvaluator()


@pytest.fixture
def api_config() -> AgentConfig:
    return AgentConfig(
        mode="api",
        model="gpt-4.1",
        evaluator_model="gpt-4.1",
        max_budget_usd=0.02,
    )


@pytest.fixture
def claude_code_config() -> AgentConfig:
    return AgentConfig(
        mode="claude_code",
        model="sonnet",
        evaluator_model="sonnet",
        max_budget_usd=0.10,
    )


@pytest.fixture
def criterion() -> BinaryCriterion:
    return BinaryCriterion(name="test_criterion", question="Is this good? YES or NO.")


# ---------------------------------------------------------------------------
# _generate_sample dispatch
# ---------------------------------------------------------------------------


class TestGenerateSampleDispatch:
    @pytest.mark.asyncio
    async def test_api_mode_uses_openai_client(
        self, evaluator: StochasticEvaluator, api_config: AgentConfig, tmp_path: Path
    ) -> None:
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="generated text"))]
        mock_response.usage = AsyncMock(prompt_tokens=100, completion_tokens=50)

        with patch("anneal.engine.eval.make_client") as mock_client:
            client_instance = AsyncMock()
            client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = client_instance

            text, cost = await evaluator._generate_sample(
                api_config, "test prompt", "text", tmp_path,
            )

        assert text == "generated text"
        mock_client.assert_called_once_with("gpt-4.1")

    @pytest.mark.asyncio
    async def test_claude_code_mode_uses_agent_invoker(
        self, evaluator: StochasticEvaluator, claude_code_config: AgentConfig, tmp_path: Path
    ) -> None:
        mock_result = AgentInvocationResult(
            success=True, cost_usd=0.05, input_tokens=200, output_tokens=100,
            hypothesis=None, hypothesis_source="synthesized",
            tags=[], raw_output="generated via claude code",
        )

        mock_invoker = AsyncMock()
        mock_invoker.invoke = AsyncMock(return_value=mock_result)
        evaluator._invoker = mock_invoker

        text, cost = await evaluator._generate_sample(
            claude_code_config, "test prompt", "text", tmp_path,
        )

        assert text == "generated via claude code"
        assert cost == 0.05
        mock_invoker.invoke.assert_called_once()
        call_kwargs = mock_invoker.invoke.call_args
        assert call_kwargs[1].get("deployment_mode") is True or call_kwargs[0][4] is True


# ---------------------------------------------------------------------------
# _score_criterion_once dispatch
# ---------------------------------------------------------------------------


class TestScoreCriterionDispatch:
    @pytest.mark.asyncio
    async def test_api_mode_uses_openai_client(
        self, evaluator: StochasticEvaluator, api_config: AgentConfig,
        criterion: BinaryCriterion, tmp_path: Path,
    ) -> None:
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="YES"))]
        mock_response.usage = AsyncMock(prompt_tokens=50, completion_tokens=5)

        with patch("anneal.engine.eval.make_client") as mock_client:
            client_instance = AsyncMock()
            client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = client_instance

            binary, cost = await evaluator._score_criterion_once(
                api_config, "sample text", criterion, tmp_path,
            )

        assert binary == 1.0
        mock_client.assert_called_once_with("gpt-4.1")

    @pytest.mark.asyncio
    async def test_claude_code_mode_uses_agent_invoker(
        self, evaluator: StochasticEvaluator, claude_code_config: AgentConfig,
        criterion: BinaryCriterion, tmp_path: Path,
    ) -> None:
        mock_result = AgentInvocationResult(
            success=True, cost_usd=0.02, input_tokens=100, output_tokens=10,
            hypothesis=None, hypothesis_source="synthesized",
            tags=[], raw_output="YES",
        )

        mock_invoker = AsyncMock()
        mock_invoker.invoke = AsyncMock(return_value=mock_result)
        evaluator._invoker = mock_invoker

        binary, cost = await evaluator._score_criterion_once(
            claude_code_config, "sample text", criterion, tmp_path,
        )

        assert binary == 1.0
        assert cost == 0.02

    @pytest.mark.asyncio
    async def test_no_answer_defaults_to_zero(
        self, evaluator: StochasticEvaluator, claude_code_config: AgentConfig,
        criterion: BinaryCriterion, tmp_path: Path,
    ) -> None:
        mock_result = AgentInvocationResult(
            success=True, cost_usd=0.02, input_tokens=100, output_tokens=10,
            hypothesis=None, hypothesis_source="synthesized",
            tags=[], raw_output="NO",
        )

        mock_invoker = AsyncMock()
        mock_invoker.invoke = AsyncMock(return_value=mock_result)
        evaluator._invoker = mock_invoker

        binary, cost = await evaluator._score_criterion_once(
            claude_code_config, "sample text", criterion, tmp_path,
        )

        assert binary == 0.0


# ---------------------------------------------------------------------------
# Judgment config fallback
# ---------------------------------------------------------------------------


class TestJudgmentConfigFallback:
    def test_judgment_agent_config_defaults_none(self) -> None:
        from anneal.engine.types import StochasticEval

        stoch = StochasticEval(
            sample_count=5,
            criteria=[BinaryCriterion(name="x", question="q")],
            test_prompts=["p"],
            generation_prompt_template="{test_prompt}",
            output_format="text",
            generation_agent_config=AgentConfig(
                mode="api", model="gemini-2.5-flash", evaluator_model="gpt-4.1",
            ),
        )
        assert stoch.judgment_agent_config is None

    def test_judgment_agent_config_explicit(self) -> None:
        from anneal.engine.types import StochasticEval

        judge = AgentConfig(
            mode="claude_code", model="sonnet", evaluator_model="sonnet",
        )
        stoch = StochasticEval(
            sample_count=5,
            criteria=[BinaryCriterion(name="x", question="q")],
            test_prompts=["p"],
            generation_prompt_template="{test_prompt}",
            output_format="text",
            generation_agent_config=AgentConfig(
                mode="api", model="gemini-2.5-flash", evaluator_model="gpt-4.1",
            ),
            judgment_agent_config=judge,
        )
        assert stoch.judgment_agent_config is not None
        assert stoch.judgment_agent_config.mode == "claude_code"
        assert stoch.judgment_agent_config.model == "sonnet"
