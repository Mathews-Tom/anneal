"""Unit tests for the research operator."""

from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from anneal.engine.context import ContextBudget
from anneal.engine.research import ResearchOperator, ResearchResult, ResearchSuggestion
from anneal.engine.types import AgentConfig, ResearchConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SUGGESTIONS_PAYLOAD = [
    {
        "technique": "Gradient Clipping",
        "description": "Clips gradient norms to prevent exploding gradients.",
        "source": "https://arxiv.org/abs/1211.5063",
        "relevance": "Stabilizes training when loss spikes.",
    },
    {
        "technique": "Cosine Annealing",
        "description": "Cyclically reduces learning rate following a cosine curve.",
        "source": "https://arxiv.org/abs/1608.03983",
        "relevance": "Helps escape local minima during optimization.",
    },
]


def _agent_config() -> AgentConfig:
    return AgentConfig(
        mode="api",
        model="gpt-4.1",
        evaluator_model="gpt-4.1-mini",
        max_budget_usd=1.0,
        max_context_tokens=80000,
    )


def _mock_response(payload: list[dict[str, str]]) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps({"suggestions": payload})
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response


def _patch_client(response: MagicMock, cost: float = 0.001):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)
    return (
        patch("anneal.engine.research.make_client", return_value=mock_client),
        patch("anneal.engine.research.compute_cost", return_value=cost),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_operator_returns_suggestions() -> None:
    """research() returns ResearchResult with correct suggestion count."""
    # Arrange
    config = ResearchConfig(enabled=True, max_budget_usd=1.0)
    operator = ResearchOperator(config)
    response = _mock_response(SUGGESTIONS_PAYLOAD)
    patch_client, patch_cost = _patch_client(response)

    # Act
    with patch_client, patch_cost:
        result = await operator.research(
            target_description="Optimize neural network training",
            current_artifact_summary="A basic training loop.",
            failed_criteria=["convergence speed"],
            recent_hypotheses=["increase batch size"],
            agent_config=_agent_config(),
        )

    # Assert
    assert result is not None
    assert isinstance(result, ResearchResult)
    assert len(result.suggestions) == 2
    assert all(isinstance(s, ResearchSuggestion) for s in result.suggestions)
    assert result.suggestions[0].technique == "Gradient Clipping"
    assert result.suggestions[1].technique == "Cosine Annealing"


@pytest.mark.asyncio
async def test_research_operator_budget_exhausted_returns_none() -> None:
    """research() returns None when total cost meets max_budget_usd."""
    # Arrange
    config = ResearchConfig(enabled=True, max_budget_usd=0.01)
    operator = ResearchOperator(config)
    operator._total_cost = 0.01

    # Act
    result = await operator.research(
        target_description="Optimize training",
        current_artifact_summary="Loop.",
        failed_criteria=["speed"],
        recent_hypotheses=[],
        agent_config=_agent_config(),
    )

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_research_operator_disabled_after_failures_returns_none() -> None:
    """Three consecutive failure outcomes disable the operator."""
    # Arrange
    config = ResearchConfig(enabled=True, disable_after_failures=3)
    operator = ResearchOperator(config)

    # Act
    operator.record_outcome(False)
    operator.record_outcome(False)
    operator.record_outcome(False)

    # Assert
    assert operator._disabled is True

    result = await operator.research(
        target_description="Optimize training",
        current_artifact_summary="Loop.",
        failed_criteria=["speed"],
        recent_hypotheses=[],
        agent_config=_agent_config(),
    )
    assert result is None


def test_research_operator_re_enabled_on_success() -> None:
    """A successful outcome resets consecutive failures and keeps operator enabled."""
    # Arrange
    config = ResearchConfig(enabled=True, disable_after_failures=3)
    operator = ResearchOperator(config)

    # Act
    operator.record_outcome(False)
    operator.record_outcome(False)
    operator.record_outcome(True)

    # Assert
    assert operator._consecutive_failures == 0
    assert operator._disabled is False


def test_research_config_disabled_by_default() -> None:
    """Default ResearchConfig has enabled=False."""
    # Arrange / Act
    config = ResearchConfig()

    # Assert
    assert config.enabled is False


@pytest.mark.asyncio
async def test_research_cost_tracked_after_api_call() -> None:
    """_total_cost increases by the mocked cost after a research() call."""
    # Arrange
    config = ResearchConfig(enabled=True, max_budget_usd=1.0)
    operator = ResearchOperator(config)
    response = _mock_response(SUGGESTIONS_PAYLOAD)
    expected_cost = 0.0042
    patch_client, patch_cost = _patch_client(response, cost=expected_cost)

    # Act
    with patch_client, patch_cost:
        await operator.research(
            target_description="Optimize training",
            current_artifact_summary="Loop.",
            failed_criteria=["speed"],
            recent_hypotheses=[],
            agent_config=_agent_config(),
        )

    # Assert
    assert operator._total_cost == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_research_api_failure_graceful_returns_none() -> None:
    """research() returns None when the API client raises an exception."""
    # Arrange
    config = ResearchConfig(enabled=True, max_budget_usd=1.0)
    operator = ResearchOperator(config)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("API unavailable"),
    )

    # Act
    with patch("anneal.engine.research.make_client", return_value=mock_client):
        result = await operator.research(
            target_description="Optimize training",
            current_artifact_summary="Loop.",
            failed_criteria=["speed"],
            recent_hypotheses=[],
            agent_config=_agent_config(),
        )

    # Assert
    assert result is None


def test_research_build_query_contains_context() -> None:
    """_build_query output includes target, failed criteria, and recent hypotheses."""
    # Arrange
    config = ResearchConfig(enabled=True)
    operator = ResearchOperator(config)
    target = "Reduce inference latency"
    failed = ["p99 latency", "throughput"]
    hypotheses = ["quantize weights", "prune layers", "use distillation"]

    # Act
    query = operator._build_query(target, failed, hypotheses)

    # Assert
    assert target in query
    assert "p99 latency" in query
    assert "throughput" in query
    assert "prune layers" in query
    assert "use distillation" in query


def test_research_hints_appear_in_context_slot() -> None:
    """Research hints are assembled into context budget when provided."""
    # Arrange
    budget = ContextBudget(max_tokens=10_000)
    budget.add_slot("system", "System prompt here", priority=1, required=True)

    hints = ResearchResult(
        suggestions=[
            ResearchSuggestion(
                technique="Beam Search",
                description="Explores multiple paths simultaneously.",
                source="general knowledge",
                relevance="May improve diversity of solutions.",
            ),
        ],
        cost_usd=0.002,
        query_used="test query",
    )

    # Simulate the context.py logic for research_hints slot
    hints_text = "# External Research Suggestions\n\n"
    for s in hints.suggestions:
        hints_text += (
            f"## {s.technique}\n"
            f"{s.description}\n"
            f"Source: {s.source}\n"
            f"Relevance: {s.relevance}\n\n"
        )
    hints_text += (
        "These are suggestions from external research. "
        "Use them as inspiration if relevant. Ignore if not applicable."
    )
    budget.add_slot("research_hints", hints_text, priority=5, required=False)

    # Act
    assembled = budget.assemble()

    # Assert
    assert "External Research Suggestions" in assembled
    assert "Beam Search" in assembled
    assert "Explores multiple paths simultaneously" in assembled
