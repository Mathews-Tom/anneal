"""Coverage-targeted tests for anneal.engine.eval — uncovered line ranges."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from anneal.engine.eval import (
    DeterministicEvaluator,
    EvalEngine,
    EvalError,
    StochasticEvaluator,
)
from anneal.engine.types import (
    AgentConfig,
    BinaryCriterion,
    ConstraintCommand,
    DeterministicEval,
    Direction,
    EvalConfig,
    EvalResult,
    StochasticEval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_det_eval(
    *,
    run_command: str = "echo done",
    parse_command: str = "echo 0.5",
    timeout_seconds: int = 10,
    max_retries: int = 1,
    retry_delay_seconds: float = 0.0,
    flake_detection: bool = False,
) -> DeterministicEval:
    return DeterministicEval(
        run_command=run_command,
        parse_command=parse_command,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        flake_detection=flake_detection,
    )


def _make_subprocess_mock(returncode: int = 0, stdout: bytes = b"0.8", stderr: bytes = b"") -> AsyncMock:
    """Return a mock subprocess Process with communicate() returning (stdout, stderr)."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = AsyncMock()
    proc.wait = AsyncMock()
    return proc


def _make_stochastic_config(
    *,
    test_prompts: list[str] | None = None,
    held_out_prompts: list[str] | None = None,
    judgment_votes: int = 1,
    min_criterion_scores: dict[str, float] | None = None,
    adaptive_sampling: bool = False,
) -> StochasticEval:
    gen_cfg = AgentConfig(
        mode="api",
        model="gpt-4.1",
        evaluator_model="gpt-4.1",
        max_budget_usd=0.10,
    )
    criteria = [
        BinaryCriterion(name="clarity", question="Is this clear?"),
        BinaryCriterion(name="accuracy", question="Is this accurate?"),
    ]
    return StochasticEval(
        sample_count=2,
        criteria=criteria,
        test_prompts=test_prompts or ["prompt A", "prompt B"],
        generation_prompt_template="{test_prompt}\n\n{artifact_content}",
        output_format="text",
        generation_agent_config=gen_cfg,
        held_out_prompts=held_out_prompts or [],
        min_criterion_scores=min_criterion_scores or {},
        judgment_votes=judgment_votes,
        adaptive_sampling=adaptive_sampling,
    )


# ---------------------------------------------------------------------------
# Lines 111-117 — DeterministicEvaluator.evaluate flake detection
# ---------------------------------------------------------------------------


class TestDeterministicEvaluatorFlakeDetection:
    @pytest.mark.asyncio
    async def test_evaluate_flake_detection_runs_three_times_returns_median(
        self, tmp_path: Path
    ) -> None:
        """flake_detection=True runs _evaluate_with_retry 3× and returns median score."""
        # Arrange
        config = _make_det_eval(flake_detection=True)
        evaluator = DeterministicEvaluator()

        # _evaluate_with_retry calls _evaluate_once which calls subprocess twice.
        # We patch _evaluate_once directly to return fixed scores in order.
        scores = [EvalResult(score=0.6), EvalResult(score=0.9), EvalResult(score=0.7)]
        call_count = 0

        async def _fake_once(worktree: Path, cfg: DeterministicEval) -> EvalResult:
            nonlocal call_count
            result = scores[call_count]
            call_count += 1
            return result

        evaluator._evaluate_once = _fake_once  # type: ignore[method-assign]

        # Act
        result = await evaluator.evaluate(tmp_path, config)

        # Assert — median of [0.6, 0.7, 0.9] is 0.7
        assert call_count == 3
        assert result.score == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_evaluate_flake_detection_median_of_lowest_two(
        self, tmp_path: Path
    ) -> None:
        """flake_detection median is the middle value of sorted 3 scores."""
        # Arrange
        config = _make_det_eval(flake_detection=True)
        evaluator = DeterministicEvaluator()

        scores_iter = iter([EvalResult(score=0.3), EvalResult(score=0.3), EvalResult(score=1.0)])

        async def _fake_once(worktree: Path, cfg: DeterministicEval) -> EvalResult:
            return next(scores_iter)

        evaluator._evaluate_once = _fake_once  # type: ignore[method-assign]

        # Act
        result = await evaluator.evaluate(tmp_path, config)

        # Assert — sorted([0.3, 0.3, 1.0])[1] == 0.3
        assert result.score == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Lines 129-137 — _evaluate_with_retry retry loop with delay
# ---------------------------------------------------------------------------


class TestDeterministicEvaluatorRetry:
    @pytest.mark.asyncio
    async def test_evaluate_with_retry_succeeds_on_second_attempt(
        self, tmp_path: Path
    ) -> None:
        """Retry loop calls _evaluate_once again after EvalError, then returns success."""
        # Arrange
        config = _make_det_eval(max_retries=2, retry_delay_seconds=0.0)
        evaluator = DeterministicEvaluator()

        attempt = 0

        async def _once_fail_then_succeed(worktree: Path, cfg: DeterministicEval) -> EvalResult:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise EvalError("transient failure")
            return EvalResult(score=0.75)

        evaluator._evaluate_once = _once_fail_then_succeed  # type: ignore[method-assign]

        # Act
        with patch("anneal.engine.eval.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await evaluator._evaluate_with_retry(tmp_path, config)

        # Assert
        assert result.score == pytest.approx(0.75)
        assert attempt == 2
        # sleep called once between retries (but delay=0 so value=0.0)
        mock_sleep.assert_called_once_with(0.0)

    @pytest.mark.asyncio
    async def test_evaluate_with_retry_exhausts_retries_raises_last_error(
        self, tmp_path: Path
    ) -> None:
        """After max_retries failures, _evaluate_with_retry re-raises the last EvalError."""
        # Arrange
        config = _make_det_eval(max_retries=3, retry_delay_seconds=0.0)
        evaluator = DeterministicEvaluator()

        async def _always_fail(worktree: Path, cfg: DeterministicEval) -> EvalResult:
            raise EvalError("persistent failure")

        evaluator._evaluate_once = _always_fail  # type: ignore[method-assign]

        # Act + Assert
        with patch("anneal.engine.eval.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(EvalError, match="persistent failure"):
                await evaluator._evaluate_with_retry(tmp_path, config)

    @pytest.mark.asyncio
    async def test_evaluate_with_retry_delay_called_between_retries(
        self, tmp_path: Path
    ) -> None:
        """retry_delay_seconds is passed to asyncio.sleep between attempts."""
        # Arrange
        config = _make_det_eval(max_retries=2, retry_delay_seconds=2.5)
        evaluator = DeterministicEvaluator()

        attempt = 0

        async def _once_then_succeed(worktree: Path, cfg: DeterministicEval) -> EvalResult:
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise EvalError("fail")
            return EvalResult(score=0.5)

        evaluator._evaluate_once = _once_then_succeed  # type: ignore[method-assign]

        # Act
        with patch("anneal.engine.eval.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await evaluator._evaluate_with_retry(tmp_path, config)

        # Assert
        mock_sleep.assert_called_once_with(2.5)


# ---------------------------------------------------------------------------
# Lines 157-161 — _evaluate_once run_command timeout
# ---------------------------------------------------------------------------


class TestDeterministicEvaluatorOnce:
    @pytest.mark.asyncio
    async def test_evaluate_once_run_command_timeout_raises_eval_error(
        self, tmp_path: Path
    ) -> None:
        """When run_command hits timeout, EvalError is raised with timeout message."""
        # Arrange
        config = _make_det_eval(timeout_seconds=1)
        evaluator = DeterministicEvaluator()
        run_proc = _make_subprocess_mock()
        run_proc.communicate.side_effect = asyncio.TimeoutError()

        # Act + Assert
        with patch("anneal.engine.eval.asyncio.create_subprocess_shell", return_value=run_proc) as mock_create:
            with patch("anneal.engine.eval.asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                with pytest.raises(EvalError, match="run_command timed out"):
                    await evaluator._evaluate_once(tmp_path, config)

        run_proc.kill.assert_called_once()
        run_proc.wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_once_run_command_nonzero_exit_raises_eval_error(
        self, tmp_path: Path
    ) -> None:
        """Non-zero run_command returncode raises EvalError with exit code info."""
        # Arrange
        config = _make_det_eval()
        evaluator = DeterministicEvaluator()

        run_proc = _make_subprocess_mock(returncode=1, stdout=b"", stderr=b"command failed")

        call_count = 0

        async def _fake_create(*args: object, **kwargs: object) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            return run_proc

        with patch("anneal.engine.eval.asyncio.create_subprocess_shell", side_effect=_fake_create):
            with patch("anneal.engine.eval.asyncio.wait_for", return_value=(b"", b"command failed")):
                with pytest.raises(EvalError, match="run_command exited with code 1"):
                    await evaluator._evaluate_once(tmp_path, config)

    @pytest.mark.asyncio
    async def test_evaluate_once_parse_command_timeout_raises_eval_error(
        self, tmp_path: Path
    ) -> None:
        """When parse_command times out, EvalError is raised with parse timeout message."""
        # Arrange
        config = _make_det_eval(timeout_seconds=5)
        evaluator = DeterministicEvaluator()

        run_proc = _make_subprocess_mock(returncode=0, stdout=b"raw output", stderr=b"")
        parse_proc = _make_subprocess_mock()
        parse_proc.communicate.side_effect = asyncio.TimeoutError()

        create_calls = 0

        async def _fake_create(*args: object, **kwargs: object) -> AsyncMock:
            nonlocal create_calls
            create_calls += 1
            if create_calls == 1:
                return run_proc
            return parse_proc

        wait_calls = 0

        async def _fake_wait_for(coro: object, timeout: object) -> object:
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                return (b"raw output", b"")
            raise asyncio.TimeoutError()

        with patch("anneal.engine.eval.asyncio.create_subprocess_shell", side_effect=_fake_create):
            with patch("anneal.engine.eval.asyncio.wait_for", side_effect=_fake_wait_for):
                with pytest.raises(EvalError, match="parse_command timed out"):
                    await evaluator._evaluate_once(tmp_path, config)

        parse_proc.kill.assert_called_once()
        parse_proc.wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_once_parse_command_nonzero_exit_raises_eval_error(
        self, tmp_path: Path
    ) -> None:
        """Non-zero parse_command returncode raises EvalError."""
        # Arrange
        config = _make_det_eval()
        evaluator = DeterministicEvaluator()

        run_proc = _make_subprocess_mock(returncode=0, stdout=b"raw output", stderr=b"")
        parse_proc = _make_subprocess_mock(returncode=2, stdout=b"", stderr=b"parse failed")

        create_calls = 0

        async def _fake_create(*args: object, **kwargs: object) -> AsyncMock:
            nonlocal create_calls
            create_calls += 1
            if create_calls == 1:
                return run_proc
            return parse_proc

        wait_calls = 0

        async def _fake_wait_for(coro: object, timeout: object) -> object:
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                return (b"raw output", b"")
            return (b"", b"parse failed")

        with patch("anneal.engine.eval.asyncio.create_subprocess_shell", side_effect=_fake_create):
            with patch("anneal.engine.eval.asyncio.wait_for", side_effect=_fake_wait_for):
                with pytest.raises(EvalError, match="parse_command exited with code 2"):
                    await evaluator._evaluate_once(tmp_path, config)

    @pytest.mark.asyncio
    async def test_evaluate_once_parse_output_not_float_raises_eval_error(
        self, tmp_path: Path
    ) -> None:
        """Non-numeric parse_command output raises EvalError about float parsing."""
        # Arrange
        config = _make_det_eval()
        evaluator = DeterministicEvaluator()

        run_proc = _make_subprocess_mock(returncode=0, stdout=b"raw", stderr=b"")
        parse_proc = _make_subprocess_mock(returncode=0, stdout=b"not_a_number", stderr=b"")

        create_calls = 0

        async def _fake_create(*args: object, **kwargs: object) -> AsyncMock:
            nonlocal create_calls
            create_calls += 1
            if create_calls == 1:
                return run_proc
            return parse_proc

        wait_calls = 0

        async def _fake_wait_for(coro: object, timeout: object) -> object:
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                return (b"raw", b"")
            return (b"not_a_number", b"")

        with patch("anneal.engine.eval.asyncio.create_subprocess_shell", side_effect=_fake_create):
            with patch("anneal.engine.eval.asyncio.wait_for", side_effect=_fake_wait_for):
                with pytest.raises(EvalError, match="Cannot parse score as float"):
                    await evaluator._evaluate_once(tmp_path, config)


# ---------------------------------------------------------------------------
# Lines 282, 296 — _evaluate_with_prompts / evaluate_held_out error paths
# ---------------------------------------------------------------------------


class TestStochasticEvaluatorErrorPaths:
    @pytest.mark.asyncio
    async def test_evaluate_held_out_no_held_out_prompts_raises(
        self, tmp_path: Path
    ) -> None:
        """evaluate_held_out raises EvalError when held_out_prompts is empty."""
        # Arrange
        evaluator = StochasticEvaluator()
        config = _make_stochastic_config(held_out_prompts=[])

        # Act + Assert
        with pytest.raises(EvalError, match="No held_out_prompts configured"):
            await evaluator.evaluate_held_out(tmp_path, config, "artifact content")

    @pytest.mark.asyncio
    async def test_evaluate_with_prompts_no_gen_cfg_raises(
        self, tmp_path: Path
    ) -> None:
        """_evaluate_with_prompts raises EvalError when generation_agent_config is None."""
        # Arrange
        evaluator = StochasticEvaluator()
        config = StochasticEval(
            sample_count=2,
            criteria=[BinaryCriterion(name="q", question="Q?")],
            test_prompts=["p1"],
            generation_prompt_template="{test_prompt}",
            output_format="text",
            generation_agent_config=None,  # missing
        )

        # Act + Assert
        with pytest.raises(EvalError, match="StochasticEval requires generation_agent_config"):
            await evaluator.evaluate(tmp_path, config, "artifact")


# ---------------------------------------------------------------------------
# Lines 331-405 — _evaluate_single_sample (via _evaluate_fixed)
# ---------------------------------------------------------------------------


class TestEvaluateSingleSample:
    """Test _evaluate_single_sample logic through _evaluate_fixed.

    Mock _generate_sample and _score_criterion to isolate vote merging logic.
    """

    def _make_gen_judge_cfg(self) -> tuple[AgentConfig, AgentConfig]:
        gen_cfg = AgentConfig(
            mode="api", model="gpt-4.1", evaluator_model="gpt-4.1", max_budget_usd=0.10,
        )
        judge_cfg = AgentConfig(
            mode="api", model="gpt-4.1", evaluator_model="gpt-4.1", max_budget_usd=0.10,
        )
        return gen_cfg, judge_cfg

    @pytest.mark.asyncio
    async def test_single_vote_uses_shuffle_path_returns_score(
        self, tmp_path: Path
    ) -> None:
        """judgment_votes=1 takes the single-vote shuffle path."""
        # Arrange
        config = _make_stochastic_config(judgment_votes=1, test_prompts=["p1", "p2"])
        gen_cfg, judge_cfg = self._make_gen_judge_cfg()
        evaluator = StochasticEvaluator()

        evaluator._generate_sample = AsyncMock(return_value=("sample text", 0.01))  # type: ignore[method-assign]
        # Both criteria score 1.0
        evaluator._score_criterion = AsyncMock(return_value=(1.0, 0.005))  # type: ignore[method-assign]

        # Act
        sample_score, sample_cost, per_criterion = await evaluator._evaluate_single_sample(
            tmp_path, config, "artifact", "p1", gen_cfg, judge_cfg,
        )

        # Assert — 2 criteria × 1.0 each = 2.0 total
        assert sample_score == pytest.approx(2.0)
        assert len(per_criterion) == 2
        assert per_criterion["clarity"] == pytest.approx(1.0)
        assert per_criterion["accuracy"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_two_votes_forward_reverse_merge_averages_scores(
        self, tmp_path: Path
    ) -> None:
        """judgment_votes=2 takes the forward/reverse split path and averages results."""
        # Arrange
        config = _make_stochastic_config(judgment_votes=2, test_prompts=["p1"])
        gen_cfg, judge_cfg = self._make_gen_judge_cfg()
        evaluator = StochasticEvaluator()

        evaluator._generate_sample = AsyncMock(return_value=("text", 0.01))  # type: ignore[method-assign]

        # forward_votes = 1, reverse_votes = 1
        # clarity forward=1.0, accuracy forward=0.0
        # reversed order: accuracy reverse=1.0, clarity reverse=0.0
        call_order: list[str] = []

        async def _score(
            judge: AgentConfig,
            sample: str,
            criterion: BinaryCriterion,
            wp: Path,
            *,
            votes: int,
            comparison_mode: str,
        ) -> tuple[float, float]:
            call_order.append(criterion.name)
            # Forward pass (first call per criterion) = 1.0, reverse = 0.0
            if call_order.count(criterion.name) == 1:
                return (1.0, 0.005)
            return (0.0, 0.005)

        evaluator._score_criterion = _score  # type: ignore[method-assign]

        # Act
        sample_score, sample_cost, per_criterion = await evaluator._evaluate_single_sample(
            tmp_path, config, "artifact", "p1", gen_cfg, judge_cfg,
        )

        # Assert — each criterion = average of 1.0 and 0.0 = 0.5, total = 1.0
        assert sample_score == pytest.approx(1.0)
        assert per_criterion["clarity"] == pytest.approx(0.5)
        assert per_criterion["accuracy"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_evaluate_fixed_aggregates_samples_correctly(
        self, tmp_path: Path
    ) -> None:
        """_evaluate_fixed collects all sample scores and returns aggregated EvalResult."""
        # Arrange
        config = _make_stochastic_config(judgment_votes=1, test_prompts=["p1", "p2"])
        gen_cfg, judge_cfg = self._make_gen_judge_cfg()
        evaluator = StochasticEvaluator()

        # Both samples: clarity=1.0, accuracy=0.0 → sample_score=1.0
        evaluator._generate_sample = AsyncMock(return_value=("text", 0.01))  # type: ignore[method-assign]

        async def _score(
            judge: AgentConfig,
            sample: str,
            criterion: BinaryCriterion,
            wp: Path,
            *,
            votes: int,
            comparison_mode: str,
        ) -> tuple[float, float]:
            return (1.0 if criterion.name == "clarity" else 0.0, 0.002)

        evaluator._score_criterion = _score  # type: ignore[method-assign]

        # Act
        result = await evaluator._evaluate_fixed(
            tmp_path, config, "artifact", ["p1", "p2"], gen_cfg, judge_cfg,
        )

        # Assert — 2 samples × score 1.0 each → mean 1.0
        assert result.score == pytest.approx(1.0)
        assert result.per_criterion_scores is not None
        assert result.per_criterion_scores["clarity"] == pytest.approx(1.0)
        assert result.per_criterion_scores["accuracy"] == pytest.approx(0.0)
        assert result.raw_scores == [1.0, 1.0]

    @pytest.mark.asyncio
    async def test_evaluate_single_sample_uses_generation_prompt_template(
        self, tmp_path: Path
    ) -> None:
        """_evaluate_single_sample formats the prompt template with test_prompt and artifact."""
        # Arrange
        config = _make_stochastic_config(judgment_votes=1, test_prompts=["hello"])
        gen_cfg, judge_cfg = self._make_gen_judge_cfg()
        evaluator = StochasticEvaluator()

        captured_prompt: list[str] = []

        async def _capture_generate(
            cfg: AgentConfig, prompt: str, fmt: str, wp: Path
        ) -> tuple[str, float]:
            captured_prompt.append(prompt)
            return ("output", 0.01)

        evaluator._generate_sample = _capture_generate  # type: ignore[method-assign]
        evaluator._score_criterion = AsyncMock(return_value=(1.0, 0.001))  # type: ignore[method-assign]

        # Act
        await evaluator._evaluate_single_sample(
            tmp_path, config, "my artifact", "hello", gen_cfg, judge_cfg,
        )

        # Assert — template was expanded
        assert len(captured_prompt) == 1
        assert "hello" in captured_prompt[0]
        assert "my artifact" in captured_prompt[0]


# ---------------------------------------------------------------------------
# Lines 579-580 — _generate_sample API timeout / connection error
# ---------------------------------------------------------------------------


class TestGenerateSampleAPIErrors:
    @pytest.mark.asyncio
    async def test_generate_sample_api_timeout_raises_eval_error(
        self, tmp_path: Path
    ) -> None:
        """openai.APITimeoutError during generation is converted to EvalError."""
        # Arrange
        config = AgentConfig(
            mode="api", model="gpt-4.1", evaluator_model="gpt-4.1", max_budget_usd=0.10,
        )
        evaluator = StochasticEvaluator()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )

        # Act + Assert
        with patch("anneal.engine.eval.make_client", return_value=mock_client):
            with pytest.raises(EvalError, match="Generation API call failed"):
                await evaluator._generate_sample(config, "prompt", "text", tmp_path)

    @pytest.mark.asyncio
    async def test_generate_sample_api_connection_error_raises_eval_error(
        self, tmp_path: Path
    ) -> None:
        """openai.APIConnectionError during generation is converted to EvalError."""
        # Arrange
        config = AgentConfig(
            mode="api", model="gpt-4.1", evaluator_model="gpt-4.1", max_budget_usd=0.10,
        )
        evaluator = StochasticEvaluator()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        # Act + Assert
        with patch("anneal.engine.eval.make_client", return_value=mock_client):
            with pytest.raises(EvalError, match="Generation API call failed"):
                await evaluator._generate_sample(config, "prompt", "text", tmp_path)

    @pytest.mark.asyncio
    async def test_generate_sample_claude_code_mode_dispatches_to_invoker(
        self, tmp_path: Path
    ) -> None:
        """claude_code mode calls AgentInvoker.invoke and returns its raw_output and cost."""
        # Arrange (lines 554-561)
        from anneal.engine.types import AgentInvocationResult

        config = AgentConfig(
            mode="claude_code", model="claude-sonnet", evaluator_model="claude-sonnet",
            max_budget_usd=0.10,
        )
        evaluator = StochasticEvaluator()

        mock_result = AgentInvocationResult(
            success=True, cost_usd=0.07, input_tokens=300, output_tokens=150,
            hypothesis=None, hypothesis_source="synthesized",
            tags=[], raw_output="claude code output",
        )
        evaluator._invoker = AsyncMock()
        evaluator._invoker.invoke = AsyncMock(return_value=mock_result)

        # Act
        text, cost = await evaluator._generate_sample(config, "the prompt", "json", tmp_path)

        # Assert
        assert text == "claude code output"
        assert cost == pytest.approx(0.07)
        evaluator._invoker.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 628-629 — _score_criterion_once API error for scoring
# ---------------------------------------------------------------------------


class TestScoreCriterionAPIErrors:
    @pytest.mark.asyncio
    async def test_score_criterion_once_api_timeout_raises_eval_error(
        self, tmp_path: Path
    ) -> None:
        """openai.APITimeoutError during scoring is converted to EvalError."""
        # Arrange
        config = AgentConfig(
            mode="api", model="gpt-4.1", evaluator_model="gpt-4.1", max_budget_usd=0.10,
        )
        criterion = BinaryCriterion(name="crit", question="Good?")
        evaluator = StochasticEvaluator()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )

        # Act + Assert
        with patch("anneal.engine.eval.make_client", return_value=mock_client):
            with pytest.raises(EvalError, match="Scoring API call failed"):
                await evaluator._score_criterion_once(config, "sample", criterion, tmp_path)

    @pytest.mark.asyncio
    async def test_score_criterion_once_api_connection_error_raises_eval_error(
        self, tmp_path: Path
    ) -> None:
        """openai.APIConnectionError during scoring is converted to EvalError."""
        # Arrange
        config = AgentConfig(
            mode="api", model="gpt-4.1", evaluator_model="gpt-4.1", max_budget_usd=0.10,
        )
        criterion = BinaryCriterion(name="crit", question="Good?")
        evaluator = StochasticEvaluator()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        # Act + Assert
        with patch("anneal.engine.eval.make_client", return_value=mock_client):
            with pytest.raises(EvalError, match="Scoring API call failed"):
                await evaluator._score_criterion_once(config, "sample", criterion, tmp_path)


# ---------------------------------------------------------------------------
# Lines 628-629 — EvalEngine.evaluate stochastic with cache hit
# ---------------------------------------------------------------------------


class TestEvalEngineCacheHit:
    @pytest.mark.asyncio
    async def test_evaluate_stochastic_cache_hit_returns_cached_result(
        self, tmp_path: Path
    ) -> None:
        """EvalEngine returns cached EvalResult when cache has a matching entry."""
        # Arrange
        from anneal.engine.eval_cache import EvalCache

        cache = MagicMock(spec=EvalCache)
        cached_entry = MagicMock()
        cached_entry.score = 0.88
        cached_entry.raw_scores = [0.8, 0.9, 0.9]
        cached_entry.criterion_names = ["clarity", "accuracy"]
        cache.get = MagicMock(return_value=cached_entry)
        cache.hit_rate = 0.5

        engine = EvalEngine(cache=cache)

        stoch_config = _make_stochastic_config()
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stoch_config,
        )

        # Act
        result = await engine.evaluate(tmp_path, eval_config, artifact_content="some content")

        # Assert
        assert result.score == pytest.approx(0.88)
        assert result.raw_scores == [0.8, 0.9, 0.9]
        # stochastic evaluator should NOT have been called (cache hit)
        cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_stochastic_no_artifact_content_raises(
        self, tmp_path: Path
    ) -> None:
        """EvalEngine.evaluate raises EvalError when artifact_content is None for stochastic."""
        # Arrange
        engine = EvalEngine()
        stoch_config = _make_stochastic_config()
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stoch_config,
        )

        # Act + Assert
        with pytest.raises(EvalError, match="Stochastic evaluation requires artifact_content"):
            await engine.evaluate(tmp_path, eval_config, artifact_content=None)

    @pytest.mark.asyncio
    async def test_evaluate_no_config_raises(self, tmp_path: Path) -> None:
        """EvalEngine.evaluate raises EvalError when neither deterministic nor stochastic."""
        # Arrange (line 746)
        engine = EvalEngine()
        eval_config = EvalConfig(
            metric_name="q",
            direction=Direction.HIGHER_IS_BETTER,
        )

        # Act + Assert
        with pytest.raises(EvalError, match="EvalConfig has neither"):
            await engine.evaluate(tmp_path, eval_config)


# ---------------------------------------------------------------------------
# Lines 652-674 — EvalEngine.check_constraints
# ---------------------------------------------------------------------------


class TestEvalEngineCheckConstraints:
    @pytest.mark.asyncio
    async def test_check_constraints_stochastic_min_criterion_scores_pass(
        self, tmp_path: Path
    ) -> None:
        """check_constraints reports pass when per_criterion_scores exceed thresholds."""
        # Arrange (lines 652-674)
        engine = EvalEngine()
        stoch_config = _make_stochastic_config(
            min_criterion_scores={"clarity": 0.7, "accuracy": 0.6},
        )
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stoch_config,
        )

        # Act
        results = await engine.check_constraints(
            tmp_path,
            eval_config,
            per_criterion_scores={"clarity": 0.85, "accuracy": 0.75},
        )

        # Assert
        passed_map = {name: passed for name, passed, _ in results}
        assert passed_map["clarity"] is True
        assert passed_map["accuracy"] is True

    @pytest.mark.asyncio
    async def test_check_constraints_stochastic_min_criterion_scores_fail(
        self, tmp_path: Path
    ) -> None:
        """check_constraints reports failure when per_criterion_scores fall below threshold."""
        # Arrange
        engine = EvalEngine()
        stoch_config = _make_stochastic_config(
            min_criterion_scores={"clarity": 0.9},
        )
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stoch_config,
        )

        # Act
        results = await engine.check_constraints(
            tmp_path,
            eval_config,
            per_criterion_scores={"clarity": 0.5},
        )

        # Assert
        assert len(results) == 1
        name, passed, actual = results[0]
        assert name == "clarity"
        assert passed is False
        assert actual == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_check_constraints_stochastic_missing_criterion_uses_zero(
        self, tmp_path: Path
    ) -> None:
        """Missing criterion in per_criterion_scores defaults to 0.0."""
        # Arrange
        engine = EvalEngine()
        stoch_config = _make_stochastic_config(
            min_criterion_scores={"missing_crit": 0.5},
        )
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stoch_config,
        )

        # Act
        results = await engine.check_constraints(
            tmp_path,
            eval_config,
            per_criterion_scores={},  # missing_crit not present
        )

        # Assert
        name, passed, actual = results[0]
        assert actual == pytest.approx(0.0)
        assert passed is False

    @pytest.mark.asyncio
    async def test_check_constraints_constraint_commands_higher_is_better_pass(
        self, tmp_path: Path
    ) -> None:
        """constraint_commands with HIGHER_IS_BETTER passes when score >= threshold."""
        # Arrange (lines 663-674)
        constraint = ConstraintCommand(
            name="coverage",
            run_command="echo done",
            parse_command="echo 0.9",
            timeout_seconds=10,
            threshold=0.8,
            direction=Direction.HIGHER_IS_BETTER,
        )
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            constraint_commands=[constraint],
        )
        engine = EvalEngine()

        # Mock the deterministic evaluator to return score=0.9
        engine._deterministic.evaluate = AsyncMock(return_value=EvalResult(score=0.9))  # type: ignore[method-assign]

        # Act
        results = await engine.check_constraints(tmp_path, eval_config)

        # Assert
        assert len(results) == 1
        name, passed, actual = results[0]
        assert name == "coverage"
        assert passed is True
        assert actual == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_check_constraints_constraint_commands_lower_is_better_pass(
        self, tmp_path: Path
    ) -> None:
        """constraint_commands with LOWER_IS_BETTER passes when score <= threshold."""
        # Arrange
        constraint = ConstraintCommand(
            name="latency_ms",
            run_command="echo done",
            parse_command="echo 50",
            timeout_seconds=10,
            threshold=100.0,
            direction=Direction.LOWER_IS_BETTER,
        )
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            constraint_commands=[constraint],
        )
        engine = EvalEngine()
        engine._deterministic.evaluate = AsyncMock(return_value=EvalResult(score=50.0))  # type: ignore[method-assign]

        # Act
        results = await engine.check_constraints(tmp_path, eval_config)

        # Assert
        name, passed, actual = results[0]
        assert name == "latency_ms"
        assert passed is True
        assert actual == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_check_constraints_constraint_commands_lower_is_better_fail(
        self, tmp_path: Path
    ) -> None:
        """constraint_commands with LOWER_IS_BETTER fails when score > threshold."""
        # Arrange
        constraint = ConstraintCommand(
            name="latency_ms",
            run_command="echo done",
            parse_command="echo 200",
            timeout_seconds=10,
            threshold=100.0,
            direction=Direction.LOWER_IS_BETTER,
        )
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            constraint_commands=[constraint],
        )
        engine = EvalEngine()
        engine._deterministic.evaluate = AsyncMock(return_value=EvalResult(score=200.0))  # type: ignore[method-assign]

        # Act
        results = await engine.check_constraints(tmp_path, eval_config)

        # Assert
        name, passed, actual = results[0]
        assert passed is False
        assert actual == pytest.approx(200.0)

    @pytest.mark.asyncio
    async def test_check_constraints_combined_stochastic_and_commands(
        self, tmp_path: Path
    ) -> None:
        """check_constraints returns results for both stochastic and command constraints."""
        # Arrange
        constraint = ConstraintCommand(
            name="sec_score",
            run_command="echo done",
            parse_command="echo 0.95",
            timeout_seconds=10,
            threshold=0.9,
            direction=Direction.HIGHER_IS_BETTER,
        )
        stoch_config = _make_stochastic_config(
            min_criterion_scores={"clarity": 0.5},
        )
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stoch_config,
            constraint_commands=[constraint],
        )
        engine = EvalEngine()
        engine._deterministic.evaluate = AsyncMock(return_value=EvalResult(score=0.95))  # type: ignore[method-assign]

        # Act
        results = await engine.check_constraints(
            tmp_path,
            eval_config,
            per_criterion_scores={"clarity": 0.8},
        )

        # Assert — one stochastic + one command result
        assert len(results) == 2
        names = {r[0] for r in results}
        assert "clarity" in names
        assert "sec_score" in names


# ---------------------------------------------------------------------------
# Lines 690, 715 — EvalEngine.evaluate_held_out
# ---------------------------------------------------------------------------


class TestEvalEngineEvaluateHeldOut:
    @pytest.mark.asyncio
    async def test_evaluate_held_out_no_stochastic_raises(
        self, tmp_path: Path
    ) -> None:
        """evaluate_held_out raises EvalError when stochastic is not configured."""
        # Arrange (line 690)
        engine = EvalEngine()
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            deterministic=_make_det_eval(),
        )

        # Act + Assert
        with pytest.raises(EvalError, match="Held-out evaluation requires stochastic config"):
            await engine.evaluate_held_out(tmp_path, eval_config, "artifact")

    @pytest.mark.asyncio
    async def test_evaluate_held_out_no_held_out_prompts_raises(
        self, tmp_path: Path
    ) -> None:
        """evaluate_held_out raises EvalError when held_out_prompts is empty."""
        # Arrange (line 715)
        engine = EvalEngine()
        stoch_config = _make_stochastic_config(held_out_prompts=[])
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stoch_config,
        )

        # Act + Assert
        with pytest.raises(EvalError, match="No held_out_prompts configured"):
            await engine.evaluate_held_out(tmp_path, eval_config, "artifact")

    @pytest.mark.asyncio
    async def test_evaluate_held_out_delegates_to_stochastic_evaluator(
        self, tmp_path: Path
    ) -> None:
        """evaluate_held_out calls stochastic.evaluate_held_out with correct args."""
        # Arrange
        engine = EvalEngine()
        stoch_config = _make_stochastic_config(held_out_prompts=["held out prompt"])
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stoch_config,
        )

        expected_result = EvalResult(score=0.77)
        engine._stochastic.evaluate_held_out = AsyncMock(return_value=expected_result)  # type: ignore[method-assign]

        # Act
        result = await engine.evaluate_held_out(tmp_path, eval_config, "artifact content")

        # Assert
        assert result.score == pytest.approx(0.77)
        engine._stochastic.evaluate_held_out.assert_called_once_with(
            tmp_path, stoch_config, "artifact content"
        )


# ---------------------------------------------------------------------------
# Lines 746, 759 — EvalEngine.evaluate misc branches
# ---------------------------------------------------------------------------


class TestEvalEngineEvaluateMisc:
    @pytest.mark.asyncio
    async def test_evaluate_stochastic_cache_miss_calls_evaluator_and_stores(
        self, tmp_path: Path
    ) -> None:
        """EvalEngine calls stochastic evaluator on cache miss and stores result."""
        # Arrange
        from anneal.engine.eval_cache import EvalCache

        cache = MagicMock(spec=EvalCache)
        cache.get = MagicMock(return_value=None)  # cache miss
        cache.hit_rate = 0.0

        engine = EvalEngine(cache=cache)
        stoch_config = _make_stochastic_config()
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stoch_config,
        )

        expected_result = EvalResult(
            score=0.75,
            raw_scores=[0.7, 0.8],
            criterion_names=["clarity", "accuracy"],
        )
        engine._stochastic.evaluate = AsyncMock(return_value=expected_result)  # type: ignore[method-assign]

        # Act
        result = await engine.evaluate(tmp_path, eval_config, artifact_content="art")

        # Assert
        assert result.score == pytest.approx(0.75)
        engine._stochastic.evaluate.assert_called_once()
        cache.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_stochastic_no_cache_calls_evaluator_directly(
        self, tmp_path: Path
    ) -> None:
        """EvalEngine without cache calls stochastic evaluator directly."""
        # Arrange (line 759 — no cache path)
        engine = EvalEngine(cache=None)
        stoch_config = _make_stochastic_config()
        eval_config = EvalConfig(
            metric_name="quality",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stoch_config,
        )

        expected_result = EvalResult(score=0.65, raw_scores=[0.6, 0.7])
        engine._stochastic.evaluate = AsyncMock(return_value=expected_result)  # type: ignore[method-assign]

        # Act
        result = await engine.evaluate(tmp_path, eval_config, artifact_content="artifact")

        # Assert
        assert result.score == pytest.approx(0.65)
        engine._stochastic.evaluate.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 652-674 — _score_criterion majority vote and bradley_terry paths
# ---------------------------------------------------------------------------


class TestScoreCriterion:
    @pytest.mark.asyncio
    async def test_score_criterion_majority_vote_multi_votes_all_yes_returns_one(
        self, tmp_path: Path
    ) -> None:
        """Multiple votes with majority=YES returns 1.0 for majority_vote mode."""
        # Arrange (lines 655-674 majority vote path)
        evaluator = StochasticEvaluator()
        config = AgentConfig(
            mode="api", model="gpt-4.1", evaluator_model="gpt-4.1", max_budget_usd=0.10,
        )
        criterion = BinaryCriterion(name="c", question="Good?")

        # _score_criterion_once always returns YES=1.0
        evaluator._score_criterion_once = AsyncMock(return_value=(1.0, 0.01))  # type: ignore[method-assign]

        # Act
        score, cost = await evaluator._score_criterion(
            config, "sample", criterion, tmp_path, votes=3, comparison_mode="majority_vote",
        )

        # Assert — all YES → majority = 1.0
        assert score == pytest.approx(1.0)
        assert evaluator._score_criterion_once.call_count == 3

    @pytest.mark.asyncio
    async def test_score_criterion_majority_vote_all_no_returns_zero(
        self, tmp_path: Path
    ) -> None:
        """Multiple votes with majority=NO returns 0.0 for majority_vote mode."""
        # Arrange
        evaluator = StochasticEvaluator()
        config = AgentConfig(
            mode="api", model="gpt-4.1", evaluator_model="gpt-4.1", max_budget_usd=0.10,
        )
        criterion = BinaryCriterion(name="c", question="Good?")
        evaluator._score_criterion_once = AsyncMock(return_value=(0.0, 0.01))  # type: ignore[method-assign]

        # Act
        score, cost = await evaluator._score_criterion(
            config, "sample", criterion, tmp_path, votes=3, comparison_mode="majority_vote",
        )

        # Assert — all NO → majority = 0.0
        assert score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_score_criterion_bradley_terry_early_stop(
        self, tmp_path: Path
    ) -> None:
        """Bradley-Terry mode stops early when confidence is sufficient after 2+ votes."""
        # Arrange (lines 663-666 bradley_terry early stop)
        evaluator = StochasticEvaluator()
        config = AgentConfig(
            mode="api", model="gpt-4.1", evaluator_model="gpt-4.1", max_budget_usd=0.10,
        )
        criterion = BinaryCriterion(name="c", question="Good?")

        # Return YES for all votes — after 2 YES votes, should stop early (high confidence)
        evaluator._score_criterion_once = AsyncMock(return_value=(1.0, 0.01))  # type: ignore[method-assign]

        # Act — with 10 votes, but should stop after 2 if confidence is met
        score, cost = await evaluator._score_criterion(
            config, "sample", criterion, tmp_path, votes=10, comparison_mode="bradley_terry",
        )

        # Assert — score should be high (all YES), stopped before 10 votes
        assert score > 0.5
        # Must have called at least 2 times (first check at i=1)
        assert evaluator._score_criterion_once.call_count >= 2

    @pytest.mark.asyncio
    async def test_score_criterion_bradley_terry_no_early_stop_returns_bt_mean(
        self, tmp_path: Path
    ) -> None:
        """Bradley-Terry mode without early stop returns BT mean after all votes."""
        # Arrange (lines 668-670 bradley_terry no early stop)
        evaluator = StochasticEvaluator()
        config = AgentConfig(
            mode="api", model="gpt-4.1", evaluator_model="gpt-4.1", max_budget_usd=0.10,
        )
        criterion = BinaryCriterion(name="c", question="Good?")

        # Alternate YES/NO to keep uncertainty high (prevent early stop)
        call_count = 0

        async def _alternate(*args: object, **kwargs: object) -> tuple[float, float]:
            nonlocal call_count
            call_count += 1
            return (1.0 if call_count % 2 == 1 else 0.0, 0.01)

        evaluator._score_criterion_once = _alternate  # type: ignore[method-assign]

        # Act — with only 2 votes (minimum to potentially trigger), uncertain result
        score, cost = await evaluator._score_criterion(
            config, "sample", criterion, tmp_path, votes=2, comparison_mode="bradley_terry",
        )

        # Assert — BT mean is returned (not 0.0 or 1.0 exactly)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Line 690 — _extract_cost with no usage returns 0.0
# ---------------------------------------------------------------------------


class TestExtractCost:
    def test_extract_cost_no_usage_returns_zero(self) -> None:
        """_extract_cost returns 0.0 when response has no usage attribute."""
        # Arrange (line 690)
        from anneal.engine.eval import _extract_cost

        response = object()  # no .usage attribute

        # Act
        cost = _extract_cost(response, "gpt-4.1")

        # Assert
        assert cost == pytest.approx(0.0)

    def test_extract_cost_with_usage_delegates_to_compute_cost(self) -> None:
        """_extract_cost passes token counts to compute_cost and returns result."""
        # Arrange
        from anneal.engine.eval import _extract_cost

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        response = MagicMock()
        response.usage = usage

        # Act
        with patch("anneal.engine.eval.compute_cost", return_value=0.002) as mock_cost:
            cost = _extract_cost(response, "gpt-4.1")

        # Assert
        assert cost == pytest.approx(0.002)
        mock_cost.assert_called_once_with("gpt-4.1", 100, 50)
