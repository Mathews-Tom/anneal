"""Tests for policy agent (continuous meta-optimizer, E1)."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from anneal.engine.policy_agent import PolicyAgent
from anneal.engine.types import (
    ExperimentRecord,
    FailureClassification,
    Outcome,
    PolicyConfig,
)


def _make_config(**kwargs: object) -> PolicyConfig:
    defaults: dict[str, object] = {
        "enabled": True,
        "model": "gpt-4.1-mini",
        "max_budget_usd": 0.02,
        "lookback_window": 10,
        "rewrite_interval": 3,
    }
    defaults.update(kwargs)
    return PolicyConfig(**defaults)


def _make_record(
    outcome: Outcome = Outcome.DISCARDED,
    score: float = 0.5,
    baseline_score: float = 0.5,
    hypothesis: str = "test hypothesis",
    failure_classification: FailureClassification | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        id="test", target_id="t", git_sha="abc", pre_experiment_sha="abc",
        timestamp=datetime.now(tz=timezone.utc), hypothesis=hypothesis,
        hypothesis_source="agent", mutation_diff_summary="", score=score,
        score_ci_lower=None, score_ci_upper=None, raw_scores=None,
        baseline_score=baseline_score, outcome=outcome, failure_mode=None,
        failure_classification=failure_classification,
        duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
        bootstrap_seed=0,
    )


class TestShouldRewrite:
    def test_rewrite_at_interval(self) -> None:
        agent = PolicyAgent(_make_config(rewrite_interval=3))
        assert agent.should_rewrite(0) is False
        assert agent.should_rewrite(1) is False
        assert agent.should_rewrite(2) is False
        assert agent.should_rewrite(3) is True
        assert agent.should_rewrite(6) is True

    def test_rewrite_every_experiment(self) -> None:
        agent = PolicyAgent(_make_config(rewrite_interval=1))
        assert agent.should_rewrite(1) is True
        assert agent.should_rewrite(2) is True


class TestComputeReward:
    def test_first_rewrite_returns_zero(self) -> None:
        agent = PolicyAgent(_make_config())
        assert agent.compute_reward(0.5) == 0.0

    def test_positive_reward(self) -> None:
        agent = PolicyAgent(_make_config())
        agent._last_rewrite_score = 0.5
        assert agent.compute_reward(0.7) == pytest.approx(0.2)

    def test_negative_reward(self) -> None:
        agent = PolicyAgent(_make_config())
        agent._last_rewrite_score = 0.5
        assert agent.compute_reward(0.3) == pytest.approx(-0.2)


class TestRewriteInstructions:
    @pytest.mark.asyncio
    async def test_successful_rewrite(self) -> None:
        agent = PolicyAgent(_make_config())
        records = [
            _make_record(outcome=Outcome.DISCARDED, score=0.3),
            _make_record(outcome=Outcome.KEPT, score=0.6),
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Focus on output format validation."
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 50

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("anneal.engine.policy_agent.make_client", return_value=mock_client):
            instructions, cost = await agent.rewrite_instructions(
                recent_records=records,
                current_instructions="old instructions",
                target_description="Optimize score",
            )

        assert instructions == "Focus on output format validation."
        assert agent.rewrite_count == 1
        assert agent.current_instructions == instructions

    @pytest.mark.asyncio
    async def test_api_error_keeps_current(self) -> None:
        agent = PolicyAgent(_make_config())
        agent._current_instructions = "keep these"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        with patch("anneal.engine.policy_agent.make_client", return_value=mock_client):
            instructions, cost = await agent.rewrite_instructions(
                recent_records=[],
                current_instructions="keep these",
                target_description="test",
            )

        assert instructions == "keep these"
        assert cost == 0.0
        assert agent.rewrite_count == 0

    @pytest.mark.asyncio
    async def test_empty_response_keeps_current(self) -> None:
        agent = PolicyAgent(_make_config())

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("anneal.engine.policy_agent.make_client", return_value=mock_client):
            instructions, cost = await agent.rewrite_instructions(
                recent_records=[],
                current_instructions="existing instructions",
                target_description="test",
            )

        assert instructions == "existing instructions"
        assert agent.rewrite_count == 0

    @pytest.mark.asyncio
    async def test_includes_failure_distribution(self) -> None:
        agent = PolicyAgent(_make_config())
        records = [_make_record()]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "new instructions"
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("anneal.engine.policy_agent.make_client", return_value=mock_client):
            await agent.rewrite_instructions(
                recent_records=records,
                current_instructions="",
                target_description="test",
                failure_distribution={"output_format": 5, "logic_error": 3},
            )

        # Verify the prompt included failure distribution
        call_args = mock_client.chat.completions.create.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert "output_format" in prompt_content
        assert "logic_error" in prompt_content

    @pytest.mark.asyncio
    async def test_tracks_rewrite_score(self) -> None:
        agent = PolicyAgent(_make_config())
        records = [_make_record(score=0.7)]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "new"
        mock_response.usage = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("anneal.engine.policy_agent.make_client", return_value=mock_client):
            await agent.rewrite_instructions(
                recent_records=records,
                current_instructions="",
                target_description="test",
            )

        assert agent._last_rewrite_score == 0.7
        assert agent.compute_reward(0.9) == pytest.approx(0.2)


class TestPolicyConfigDefault:
    def test_disabled_by_default(self) -> None:
        config = PolicyConfig()
        assert config.enabled is False
        assert config.model == ""
        assert config.rewrite_interval == 3

    def test_none_policy_config_on_target(self) -> None:
        """OptimizationTarget.policy_config defaults to None."""
        from anneal.engine.types import (
            AgentConfig, DeterministicEval, Direction, DomainTier,
            EvalConfig, EvalMode, OptimizationTarget,
        )
        target = OptimizationTarget(
            id="test", domain_tier=DomainTier.SANDBOX,
            artifact_paths=["a.md"], scope_path="scope.yaml",
            scope_hash="abc", eval_mode=EvalMode.DETERMINISTIC,
            eval_config=EvalConfig(
                metric_name="s", direction=Direction.HIGHER_IS_BETTER,
                deterministic=DeterministicEval(
                    run_command="echo 1", parse_command="cat", timeout_seconds=30,
                ),
            ),
            agent_config=AgentConfig(mode="api", model="t", evaluator_model="t"),
            time_budget_seconds=60, loop_interval_seconds=60,
            knowledge_path="k", worktree_path="w", git_branch="b",
            baseline_score=0.0,
        )
        assert target.policy_config is None
