"""Tests for EvalCache integration in EvalEngine."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from anneal.engine.eval import EvalEngine
from anneal.engine.eval_cache import EvalCache
from anneal.engine.types import (
    AgentConfig,
    BinaryCriterion,
    DeterministicEval,
    Direction,
    EvalConfig,
    EvalResult,
    StochasticEval,
)


def _make_stochastic_config() -> EvalConfig:
    return EvalConfig(
        metric_name="quality",
        direction=Direction.HIGHER_IS_BETTER,
        stochastic=StochasticEval(
            sample_count=2,
            criteria=[BinaryCriterion(name="clarity", question="Is it clear?")],
            test_prompts=["test"],
            generation_prompt_template="{test_prompt} {artifact_content}",
            output_format="text",
            generation_agent_config=AgentConfig(
                mode="api",
                model="test-model",
                evaluator_model="test-model",
            ),
        ),
    )


def _make_deterministic_config() -> EvalConfig:
    return EvalConfig(
        metric_name="perf",
        direction=Direction.HIGHER_IS_BETTER,
        deterministic=DeterministicEval(
            run_command="echo 0.9",
            parse_command="cat",
            timeout_seconds=10,
        ),
    )


_STOCHASTIC_RESULT = EvalResult(
    score=0.85,
    ci_lower=0.7,
    ci_upper=0.95,
    raw_scores=[0.8, 0.9],
    cost_usd=0.01,
    criterion_names=["clarity"],
    per_criterion_scores={"clarity": 0.85},
)


class TestEvalEngineCacheHit:
    """Tests for cache hit path in EvalEngine.evaluate."""

    def test_eval_engine_cache_hit_returns_cached_result(self) -> None:
        cache = EvalCache(max_size=10)
        cache.put("artifact text", ["clarity"], 0.85, [0.8, 0.9])
        engine = EvalEngine(cache=cache)

        with patch.object(engine._stochastic, "evaluate", new_callable=AsyncMock) as mock_eval:
            result = asyncio.get_event_loop().run_until_complete(
                engine.evaluate(
                    Path("/tmp/worktree"),
                    _make_stochastic_config(),
                    artifact_content="artifact text",
                )
            )

            mock_eval.assert_not_called()

        assert result.score == 0.85
        assert result.raw_scores == [0.8, 0.9]
        assert result.criterion_names == ["clarity"]


class TestEvalEngineCacheMiss:
    """Tests for cache miss path in EvalEngine.evaluate."""

    def test_eval_engine_cache_miss_invokes_evaluator(self) -> None:
        cache = EvalCache(max_size=10)
        engine = EvalEngine(cache=cache)

        with patch.object(
            engine._stochastic, "evaluate", new_callable=AsyncMock, return_value=_STOCHASTIC_RESULT,
        ) as mock_eval:
            result = asyncio.get_event_loop().run_until_complete(
                engine.evaluate(
                    Path("/tmp/worktree"),
                    _make_stochastic_config(),
                    artifact_content="new artifact",
                )
            )

            mock_eval.assert_called_once()

        assert result.score == 0.85
        assert cache.size == 1

        cached = cache.get("new artifact", ["clarity"])
        assert cached is not None
        assert cached.score == 0.85


class TestEvalEngineCacheDisabled:
    """Tests for EvalEngine with cache=None."""

    def test_eval_engine_cache_disabled_when_none(self) -> None:
        engine = EvalEngine(cache=None)

        with patch.object(
            engine._stochastic, "evaluate", new_callable=AsyncMock, return_value=_STOCHASTIC_RESULT,
        ) as mock_eval:
            result = asyncio.get_event_loop().run_until_complete(
                engine.evaluate(
                    Path("/tmp/worktree"),
                    _make_stochastic_config(),
                    artifact_content="artifact text",
                )
            )

            mock_eval.assert_called_once()

        assert result.score == 0.85
        assert engine._cache is None


class TestEvalEngineDeterministicBypassesCache:
    """Tests that deterministic eval never touches cache."""

    def test_eval_engine_deterministic_bypasses_cache(self) -> None:
        cache = EvalCache(max_size=10)
        engine = EvalEngine(cache=cache)

        with patch.object(
            engine._deterministic,
            "evaluate",
            new_callable=AsyncMock,
            return_value=EvalResult(score=0.9),
        ):
            result = asyncio.get_event_loop().run_until_complete(
                engine.evaluate(
                    Path("/tmp/worktree"),
                    _make_deterministic_config(),
                )
            )

        assert result.score == 0.9
        assert cache.size == 0
        assert cache.hit_rate == 0.0
