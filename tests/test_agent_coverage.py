"""Tests targeting uncovered lines in anneal/engine/agent.py.

Covers:
  - _extract_hypothesis edge cases (lines 53-54)
  - _extract_tags edge cases (lines 62-64)
  - invoke() dispatch to api and unknown mode (lines 85-88)
  - _invoke_claude_code allowed tools logic, timeout, error paths (lines 132-137, 182-211)
  - _invoke_api full path (lines 244-272)
  - _build_diagnosis_prompt criterion scores + history (lines 298, 310-316)
  - diagnose() API error handling (lines 346-349)
  - invoke_api_text (lines 368-380)
  - generate_drafts both modes (lines 398-439)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anneal.engine.agent import (
    AgentInvocationError,
    AgentInvoker,
    AgentTimeoutError,
    _extract_hypothesis,
    _extract_tags,
)
from anneal.engine.types import (
    AgentConfig,
    AgentInvocationResult,
    EvalResult,
    ExperimentRecord,
    Outcome,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_config(
    *,
    model: str = "gpt-4.1",
    temperature: float = 0.7,
    max_budget_usd: float = 0.10,
) -> AgentConfig:
    return AgentConfig(
        mode="api",
        model=model,
        evaluator_model="gpt-4.1-mini",
        temperature=temperature,
        max_budget_usd=max_budget_usd,
    )


def _cc_config(*, model: str = "gpt-4.1") -> AgentConfig:
    return AgentConfig(
        mode="claude_code",
        model=model,
        evaluator_model="gpt-4.1-mini",
    )


def _mock_openai_response(
    content: str,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_experiment_record(*, hypothesis: str = "test", score: float = 0.5) -> ExperimentRecord:
    import datetime

    return ExperimentRecord(
        id="rec-1",
        target_id="t",
        git_sha="abc",
        pre_experiment_sha="abc000",
        timestamp=datetime.datetime.now(),
        hypothesis=hypothesis,
        hypothesis_source="agent",
        mutation_diff_summary="diff",
        score=score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=None,
        baseline_score=0.5,
        outcome=Outcome.DISCARDED,
        failure_mode=None,
        duration_seconds=1.0,
        tags=[],
        learnings="",
        cost_usd=0.01,
        bootstrap_seed=42,
    )


# ---------------------------------------------------------------------------
# _extract_hypothesis
# ---------------------------------------------------------------------------


class TestExtractHypothesis:
    def test_extract_hypothesis_present_returns_content(self) -> None:
        # Arrange
        text = "## Hypothesis\nThis is the hypothesis.\n## Tags\ntag1"

        # Act
        result = _extract_hypothesis(text)

        # Assert
        assert result == "This is the hypothesis."

    def test_extract_hypothesis_blank_line_before_next_header_captures_next_section(self) -> None:
        # Arrange — the \s* in the regex consumes blank lines after the header,
        # so the capture group starts at the next ## section when hypothesis body is blank.
        # This verifies the actual (not idealized) regex behavior.
        text = "## Hypothesis\n\n## Tags\ntag1"

        # Act
        result = _extract_hypothesis(text)

        # Assert: regex \s* eats the blank line, capture starts at "## Tags\ntag1"
        assert result == "## Tags\ntag1"

    def test_extract_hypothesis_no_match_returns_none(self) -> None:
        # Arrange — no ## Hypothesis header at all
        text = "Some intro text.\n\n## Tags\ntag1"

        # Act
        result = _extract_hypothesis(text)

        # Assert
        assert result is None

    def test_extract_hypothesis_absent_returns_none(self) -> None:
        # Arrange
        text = "Some text without any headers."

        # Act
        result = _extract_hypothesis(text)

        # Assert
        assert result is None

    def test_extract_hypothesis_multiline_returns_stripped(self) -> None:
        # Arrange
        text = "## Hypothesis\nLine one.\nLine two.\n## Tags\ntag"

        # Act
        result = _extract_hypothesis(text)

        # Assert
        assert result == "Line one.\nLine two."


# ---------------------------------------------------------------------------
# _extract_tags
# ---------------------------------------------------------------------------


class TestExtractTags:
    def test_extract_tags_present_returns_list(self) -> None:
        # Arrange
        text = "## Tags\nfoo, bar, baz"

        # Act
        result = _extract_tags(text)

        # Assert
        assert result == ["foo", "bar", "baz"]

    def test_extract_tags_blank_line_before_next_header_captures_next_section(self) -> None:
        # Arrange — same regex behavior: \s* in the pattern eats the blank line,
        # capture group starts at the next ## header text.
        text = "## Tags\n\n## Next"

        # Act
        result = _extract_tags(text)

        # Assert: "## Next".strip() is truthy, so it's treated as a single tag
        assert result == ["## Next"]

    def test_extract_tags_single_tag_no_comma(self) -> None:
        # Arrange — single tag with no commas
        text = "## Tags\nperformance"

        # Act
        result = _extract_tags(text)

        # Assert
        assert result == ["performance"]

    def test_extract_tags_absent_returns_empty_list(self) -> None:
        # Arrange
        text = "No tags section here."

        # Act
        result = _extract_tags(text)

        # Assert
        assert result == []

    def test_extract_tags_filters_blank_entries(self) -> None:
        # Arrange — trailing comma produces blank item
        text = "## Tags\nfoo, , bar,"

        # Act
        result = _extract_tags(text)

        # Assert
        assert result == ["foo", "bar"]


# ---------------------------------------------------------------------------
# invoke() dispatch
# ---------------------------------------------------------------------------


class TestInvokeDispatch:
    @pytest.mark.asyncio
    async def test_invoke_api_mode_calls_invoke_api(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        expected = AgentInvocationResult(
            success=True, cost_usd=0.01, input_tokens=10, output_tokens=5,
            hypothesis=None, hypothesis_source="synthesized", tags=[], raw_output="x",
        )

        with patch.object(invoker, "_invoke_api", new=AsyncMock(return_value=expected)) as mock_api:
            # Act
            result = await invoker.invoke(config, "prompt", tmp_path, 60)

        # Assert
        mock_api.assert_awaited_once()
        assert result is expected

    @pytest.mark.asyncio
    async def test_invoke_unknown_mode_raises(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = AgentConfig(
            mode="claude_code",  # valid to construct; we'll override below
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
        )
        # Manually set an unsupported mode (bypasses Pydantic literal check)
        object.__setattr__(config, "mode", "unknown_mode")

        # Act / Assert
        with pytest.raises(AgentInvocationError, match="Unknown agent mode"):
            await invoker.invoke(config, "prompt", tmp_path, 60)

    @pytest.mark.asyncio
    async def test_invoke_meta_unknown_mode_raises(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        object.__setattr__(config, "mode", "unsupported")
        program_md = tmp_path / "program.md"
        program_md.write_text("strategy text")

        # Act / Assert
        with pytest.raises(AgentInvocationError, match="Unknown agent mode"):
            await invoker.invoke_meta(config, "meta prompt", tmp_path, 60, program_md)

    @pytest.mark.asyncio
    async def test_invoke_claude_code_mode_calls_invoke_claude_code(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        expected = AgentInvocationResult(
            success=True, cost_usd=0.01, input_tokens=10, output_tokens=5,
            hypothesis=None, hypothesis_source="synthesized", tags=[], raw_output="x",
        )

        with patch.object(invoker, "_invoke_claude_code", new=AsyncMock(return_value=expected)) as mock_cc:
            # Act
            result = await invoker.invoke(config, "prompt", tmp_path, 60)

        # Assert
        mock_cc.assert_awaited_once()
        assert result is expected

    @pytest.mark.asyncio
    async def test_invoke_deployment_delegates_to_invoke_with_deployment_mode(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        expected = AgentInvocationResult(
            success=True, cost_usd=0.01, input_tokens=10, output_tokens=5,
            hypothesis=None, hypothesis_source="synthesized", tags=[], raw_output="x",
        )

        with patch.object(invoker, "_invoke_claude_code", new=AsyncMock(return_value=expected)) as mock_cc:
            # Act
            result = await invoker.invoke_deployment(config, "prompt", tmp_path, 60)

        # Assert: deployment_mode=True must be passed through
        _, kwargs = mock_cc.call_args
        assert kwargs.get("deployment_mode") is True
        assert result is expected

    @pytest.mark.asyncio
    async def test_invoke_meta_claude_code_mode_calls_invoke_claude_code(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        program_md = tmp_path / "program.md"
        program_md.write_text("strategy text")
        expected = AgentInvocationResult(
            success=True, cost_usd=0.01, input_tokens=10, output_tokens=5,
            hypothesis=None, hypothesis_source="synthesized", tags=[], raw_output="x",
        )

        with patch.object(invoker, "_invoke_claude_code", new=AsyncMock(return_value=expected)) as mock_cc:
            # Act
            result = await invoker.invoke_meta(config, "meta prompt", tmp_path, 60, program_md)

        # Assert: meta_mode=True must be passed through
        _, kwargs = mock_cc.call_args
        assert kwargs.get("meta_mode") is True
        assert result is expected

    @pytest.mark.asyncio
    async def test_invoke_meta_api_mode_calls_invoke_api(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        program_md = tmp_path / "program.md"
        program_md.write_text("strategy text")
        expected = AgentInvocationResult(
            success=True, cost_usd=0.01, input_tokens=10, output_tokens=5,
            hypothesis=None, hypothesis_source="synthesized", tags=[], raw_output="x",
        )

        with patch.object(invoker, "_invoke_api", new=AsyncMock(return_value=expected)) as mock_api:
            # Act
            result = await invoker.invoke_meta(config, "meta prompt", tmp_path, 60, program_md)

        # Assert
        mock_api.assert_awaited_once()
        assert result is expected


# ---------------------------------------------------------------------------
# _invoke_claude_code allowed tools logic
# ---------------------------------------------------------------------------


class TestInvokeClaudeCodeAllowedTools:
    def _make_proc(
        self,
        stdout: bytes = b'{"result":"ok","total_cost_usd":0.01,"usage":{"input_tokens":10,"output_tokens":5}}',
        returncode: int = 0,
        stderr: bytes = b"",
    ) -> MagicMock:
        proc = MagicMock()
        proc.pid = 12345
        proc.returncode = returncode
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
        return proc

    @pytest.mark.asyncio
    async def test_meta_mode_uses_edit_tool(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        proc = self._make_proc()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as mock_exec:
            # Act
            await invoker._invoke_claude_code(
                config, "prompt", tmp_path, 60, meta_mode=True
            )

        # Assert: --allowedTools should be "Edit"
        call_args = mock_exec.call_args[0]
        idx = list(call_args).index("--allowedTools")
        assert call_args[idx + 1] == "Edit"

    @pytest.mark.asyncio
    async def test_deployment_mode_uses_read_tool(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        proc = self._make_proc()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as mock_exec:
            # Act
            await invoker._invoke_claude_code(
                config, "prompt", tmp_path, 60, deployment_mode=True
            )

        # Assert: --allowedTools should be "Read"
        call_args = mock_exec.call_args[0]
        idx = list(call_args).index("--allowedTools")
        assert call_args[idx + 1] == "Read"

    @pytest.mark.asyncio
    async def test_normal_mode_uses_edit_write_tools(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        proc = self._make_proc()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as mock_exec:
            # Act
            await invoker._invoke_claude_code(config, "prompt", tmp_path, 60)

        # Assert: --allowedTools should be "Edit,Write"
        call_args = mock_exec.call_args[0]
        idx = list(call_args).index("--allowedTools")
        assert call_args[idx + 1] == "Edit,Write"

    @pytest.mark.asyncio
    async def test_timeout_raises_agent_timeout_error(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        proc = MagicMock()
        proc.pid = 99999
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
            patch("os.killpg", side_effect=ProcessLookupError),
        ):
            # Act / Assert
            with pytest.raises(AgentTimeoutError, match="exceeded time budget"):
                await invoker._invoke_claude_code(config, "prompt", tmp_path, 1)

    @pytest.mark.asyncio
    async def test_nonzero_returncode_raises_invocation_error(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        proc = self._make_proc(
            stdout=b"",
            returncode=1,
            stderr=b"some error",
        )

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            # Act / Assert
            with pytest.raises(AgentInvocationError, match="exited with code 1"):
                await invoker._invoke_claude_code(config, "prompt", tmp_path, 60)

    @pytest.mark.asyncio
    async def test_invalid_json_raises_invocation_error(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        proc = self._make_proc(stdout=b"not-json{{")

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            # Act / Assert
            with pytest.raises(AgentInvocationError, match="Invalid JSON"):
                await invoker._invoke_claude_code(config, "prompt", tmp_path, 60)

    @pytest.mark.asyncio
    async def test_error_subtype_raises_invocation_error(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        payload = json.dumps({
            "is_error": False,
            "subtype": "error_overloaded",
            "total_cost_usd": 0.002,
        }).encode()
        proc = self._make_proc(stdout=payload)

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            # Act / Assert
            with pytest.raises(AgentInvocationError, match="error_overloaded"):
                await invoker._invoke_claude_code(config, "prompt", tmp_path, 60)

    @pytest.mark.asyncio
    async def test_is_error_flag_raises_invocation_error(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        payload = json.dumps({
            "is_error": True,
            "subtype": "",
            "total_cost_usd": 0.0,
        }).encode()
        proc = self._make_proc(stdout=payload)

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            # Act / Assert
            with pytest.raises(AgentInvocationError, match="Claude Code returned error"):
                await invoker._invoke_claude_code(config, "prompt", tmp_path, 60)

    @pytest.mark.asyncio
    async def test_success_response_returns_invocation_result(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        payload = json.dumps({
            "result": "## Hypothesis\nFix the thing.\n## Tags\nfix, quality",
            "total_cost_usd": 0.015,
            "usage": {"input_tokens": 200, "output_tokens": 80},
        }).encode()
        proc = self._make_proc(stdout=payload)

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            # Act
            result = await invoker._invoke_claude_code(config, "prompt", tmp_path, 60)

        # Assert
        assert result.success is True
        assert result.hypothesis == "Fix the thing."
        assert result.hypothesis_source == "agent"
        assert result.tags == ["fix", "quality"]
        assert result.cost_usd == pytest.approx(0.015)
        assert result.input_tokens == 200
        assert result.output_tokens == 80


# ---------------------------------------------------------------------------
# _invoke_api full path
# ---------------------------------------------------------------------------


class TestInvokeApi:
    @pytest.mark.asyncio
    async def test_invoke_api_returns_invocation_result(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config(model="gpt-4.1", temperature=0.5)
        mock_response = _mock_openai_response(
            "## Hypothesis\nImprove clarity.\n## Tags\nclarity",
            prompt_tokens=120,
            completion_tokens=60,
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.compute_cost", return_value=0.007),
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act
            result = await invoker._invoke_api(config, "prompt text", tmp_path, 60)

        # Assert
        assert result.success is True
        assert result.hypothesis == "Improve clarity."
        assert result.hypothesis_source == "agent"
        assert result.tags == ["clarity"]
        assert result.cost_usd == pytest.approx(0.007)
        assert result.input_tokens == 120
        assert result.output_tokens == 60

    @pytest.mark.asyncio
    async def test_invoke_api_no_hypothesis_synthesized_source(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        mock_response = _mock_openai_response("Plain text with no headers.", prompt_tokens=50, completion_tokens=20)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.compute_cost", return_value=0.001),
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act
            result = await invoker._invoke_api(config, "prompt", tmp_path, 60)

        # Assert
        assert result.hypothesis is None
        assert result.hypothesis_source == "synthesized"
        assert result.tags == []

    @pytest.mark.asyncio
    async def test_invoke_api_timeout_raises_agent_timeout_error(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError())

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act / Assert
            with pytest.raises(AgentTimeoutError, match="API agent exceeded time budget"):
                await invoker._invoke_api(config, "prompt", tmp_path, 1)

    @pytest.mark.asyncio
    async def test_invoke_api_null_usage_defaults_to_zero_tokens(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        mock_response = _mock_openai_response("output")
        mock_response.usage = None
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.compute_cost", return_value=0.0) as mock_cost,
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act
            result = await invoker._invoke_api(config, "prompt", tmp_path, 60)

        # Assert
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        mock_cost.assert_called_once_with(config.model, 0, 0)

    @pytest.mark.asyncio
    async def test_invoke_api_null_content_returns_empty_raw_output(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        mock_response = _mock_openai_response("ignored")
        mock_response.choices[0].message.content = None
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.compute_cost", return_value=0.0),
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act
            result = await invoker._invoke_api(config, "prompt", tmp_path, 60)

        # Assert
        assert result.raw_output == ""


# ---------------------------------------------------------------------------
# _build_diagnosis_prompt
# ---------------------------------------------------------------------------


class TestBuildDiagnosisPrompt:
    def test_build_prompt_includes_criterion_scores(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        eval_result = EvalResult(
            score=0.6,
            per_criterion_scores={"clarity": 0.3, "depth": 0.7},
        )

        # Act
        prompt = invoker._build_diagnosis_prompt("artifact text", eval_result, [])

        # Assert
        assert "## Per-Criterion Scores" in prompt
        assert "clarity: 0.3000" in prompt
        assert "depth: 0.7000" in prompt

    def test_build_prompt_includes_ci_when_present(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        eval_result = EvalResult(score=0.5, ci_lower=0.45, ci_upper=0.55)

        # Act
        prompt = invoker._build_diagnosis_prompt("artifact", eval_result, [])

        # Assert
        assert "CI: [0.4500, 0.5500]" in prompt

    def test_build_prompt_omits_ci_when_absent(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        eval_result = EvalResult(score=0.5)

        # Act
        prompt = invoker._build_diagnosis_prompt("artifact", eval_result, [])

        # Assert
        assert "CI:" not in prompt

    def test_build_prompt_includes_recent_history(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        eval_result = EvalResult(score=0.5)
        records = [
            _make_experiment_record(hypothesis="add examples", score=0.6),
            _make_experiment_record(hypothesis="remove fluff", score=0.55),
        ]

        # Act
        prompt = invoker._build_diagnosis_prompt("artifact", eval_result, records)

        # Assert
        assert "## Recent Experiment History" in prompt
        assert "add examples" in prompt
        assert "remove fluff" in prompt

    def test_build_prompt_limits_history_to_last_five(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        eval_result = EvalResult(score=0.5)
        records = [
            _make_experiment_record(hypothesis=f"attempt-{i}", score=0.5 + i * 0.01)
            for i in range(8)
        ]

        # Act
        prompt = invoker._build_diagnosis_prompt("artifact", eval_result, records)

        # Assert: only last 5 (attempt-3 through attempt-7) should appear
        assert "attempt-7" in prompt
        assert "attempt-3" in prompt
        assert "attempt-0" not in prompt
        assert "attempt-1" not in prompt
        assert "attempt-2" not in prompt

    def test_build_prompt_omits_history_section_when_empty(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        eval_result = EvalResult(score=0.5)

        # Act
        prompt = invoker._build_diagnosis_prompt("artifact", eval_result, [])

        # Assert
        assert "## Recent Experiment History" not in prompt

    def test_build_prompt_omits_criterion_scores_when_absent(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        eval_result = EvalResult(score=0.5, per_criterion_scores=None)

        # Act
        prompt = invoker._build_diagnosis_prompt("artifact", eval_result, [])

        # Assert
        assert "## Per-Criterion Scores" not in prompt


# ---------------------------------------------------------------------------
# diagnose() API error handling
# ---------------------------------------------------------------------------


class TestDiagnoseApiErrorHandling:
    @pytest.mark.asyncio
    async def test_diagnose_timeout_raises_agent_timeout_error(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        eval_result = EvalResult(score=0.5)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError())

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act / Assert
            with pytest.raises(AgentTimeoutError, match="Diagnosis agent exceeded 60s timeout"):
                await invoker.diagnose(config, "artifact", eval_result, [], tmp_path)

    @pytest.mark.asyncio
    async def test_diagnose_generic_exception_raises_invocation_error(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        eval_result = EvalResult(score=0.5)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act / Assert
            with pytest.raises(AgentInvocationError, match="Diagnosis API call failed"):
                await invoker.diagnose(config, "artifact", eval_result, [], tmp_path)

    @pytest.mark.asyncio
    async def test_diagnose_uses_diagnosis_model_when_set(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = AgentConfig(
            mode="api",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
            diagnosis_model="gpt-5",
        )
        eval_result = EvalResult(score=0.5)
        payload = {
            "weakest_criteria": ["x"],
            "root_cause": "r",
            "fix_category": "other",
            "suggested_direction": "d",
        }
        mock_response = _mock_openai_response(json.dumps(payload))
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client) as mock_make,
            patch("anneal.engine.agent.compute_cost", return_value=0.0),
            patch("anneal.engine.agent.strip_provider_prefix", side_effect=lambda m: m),
        ):
            # Act
            await invoker.diagnose(config, "artifact", eval_result, [], tmp_path)

        # Assert: make_client called with the diagnosis_model, not the primary model
        mock_make.assert_called_once_with("gpt-5")


# ---------------------------------------------------------------------------
# invoke_api_text
# ---------------------------------------------------------------------------


class TestInvokeApiText:
    @pytest.mark.asyncio
    async def test_invoke_api_text_returns_content(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        mock_response = _mock_openai_response("generated text response")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act
            result = await invoker.invoke_api_text(config, "write something")

        # Assert
        assert result == "generated text response"

    @pytest.mark.asyncio
    async def test_invoke_api_text_null_content_returns_empty_string(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        mock_response = _mock_openai_response("placeholder")
        mock_response.choices[0].message.content = None
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act
            result = await invoker.invoke_api_text(config, "prompt")

        # Assert
        assert result == ""

    @pytest.mark.asyncio
    async def test_invoke_api_text_exception_raises_invocation_error(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("network error")
        )

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act / Assert
            with pytest.raises(AgentInvocationError, match="invoke_api_text failed"):
                await invoker.invoke_api_text(config, "prompt")

    @pytest.mark.asyncio
    async def test_invoke_api_text_passes_correct_temperature(self) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config()
        mock_response = _mock_openai_response("response")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.strip_provider_prefix", return_value="gpt-4.1"),
        ):
            # Act
            await invoker.invoke_api_text(config, "prompt")

        # Assert: temperature=0.7 is hardcoded in invoke_api_text
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# generate_drafts — API mode
# ---------------------------------------------------------------------------


class TestGenerateDraftsApiMode:
    @pytest.mark.asyncio
    async def test_generate_drafts_api_returns_n_drafts(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config(temperature=0.7, max_budget_usd=0.30)
        mock_result = AgentInvocationResult(
            success=True, cost_usd=0.01, input_tokens=10, output_tokens=5,
            hypothesis="h", hypothesis_source="agent", tags=[], raw_output="draft output",
        )
        git = AsyncMock()
        git.rev_parse = AsyncMock(return_value="sha123")

        with patch.object(invoker, "_invoke_api", new=AsyncMock(return_value=mock_result)):
            # Act
            drafts = await invoker.generate_drafts(config, "prompt", tmp_path, 60, n_drafts=3, git=git)

        # Assert
        assert len(drafts) == 3
        for result, diff in drafts:
            assert result is mock_result
            assert diff == "draft output"

    @pytest.mark.asyncio
    async def test_generate_drafts_api_skips_failed_drafts(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config(temperature=0.7, max_budget_usd=0.30)
        good_result = AgentInvocationResult(
            success=True, cost_usd=0.01, input_tokens=10, output_tokens=5,
            hypothesis="h", hypothesis_source="agent", tags=[], raw_output="good",
        )
        git = AsyncMock()
        git.rev_parse = AsyncMock(return_value="sha123")

        call_count = 0

        async def _sometimes_fail(config: AgentConfig, *args: object, **kwargs: object) -> AgentInvocationResult:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise AgentInvocationError("draft 2 failed")
            return good_result

        with patch.object(invoker, "_invoke_api", new=_sometimes_fail):
            # Act
            drafts = await invoker.generate_drafts(config, "prompt", tmp_path, 60, n_drafts=3, git=git)

        # Assert: only 2 successful drafts returned
        assert len(drafts) == 2

    @pytest.mark.asyncio
    async def test_generate_drafts_api_varies_temperature(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _api_config(temperature=0.7, max_budget_usd=0.30)
        captured_configs: list[AgentConfig] = []
        mock_result = AgentInvocationResult(
            success=True, cost_usd=0.01, input_tokens=10, output_tokens=5,
            hypothesis=None, hypothesis_source="synthesized", tags=[], raw_output="x",
        )

        async def _capture(cfg: AgentConfig, *args: object, **kwargs: object) -> AgentInvocationResult:
            captured_configs.append(cfg)
            return mock_result

        git = AsyncMock()
        git.rev_parse = AsyncMock(return_value="sha")

        with patch.object(invoker, "_invoke_api", new=_capture):
            # Act
            await invoker.generate_drafts(config, "prompt", tmp_path, 60, n_drafts=3, git=git)

        # Assert: temperatures differ across drafts
        temps = [c.temperature for c in captured_configs]
        assert len(set(temps)) > 1, "Expected varied temperatures across drafts"


# ---------------------------------------------------------------------------
# generate_drafts — claude_code mode
# ---------------------------------------------------------------------------


class TestGenerateDraftsClaudeCodeMode:
    def _make_proc(
        self,
        stdout: bytes = b'{"result":"ok","total_cost_usd":0.01,"usage":{"input_tokens":10,"output_tokens":5}}',
        returncode: int = 0,
    ) -> MagicMock:
        proc = MagicMock()
        proc.pid = 1
        proc.returncode = returncode
        proc.communicate = AsyncMock(return_value=(stdout, b""))
        return proc

    @pytest.mark.asyncio
    async def test_generate_drafts_cc_returns_diffs(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        config = config.model_copy(update={"max_budget_usd": 0.30})
        proc = self._make_proc()
        git = AsyncMock()
        git.rev_parse = AsyncMock(return_value="sha123")
        git.capture_diff = AsyncMock(return_value="--- diff ---")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            # Act
            drafts = await invoker.generate_drafts(config, "prompt", tmp_path, 60, n_drafts=2, git=git)

        # Assert
        assert len(drafts) == 2
        for _, diff in drafts:
            assert diff == "--- diff ---"

    @pytest.mark.asyncio
    async def test_generate_drafts_cc_resets_worktree_between_drafts(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        config = config.model_copy(update={"max_budget_usd": 0.30})
        proc = self._make_proc()
        git = AsyncMock()
        git.rev_parse = AsyncMock(return_value="sha123")
        git.capture_diff = AsyncMock(return_value="diff")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            # Act
            await invoker.generate_drafts(config, "prompt", tmp_path, 60, n_drafts=3, git=git)

        # Assert: reset_hard and clean_untracked called once per draft
        assert git.reset_hard.await_count == 3
        assert git.clean_untracked.await_count == 3

    @pytest.mark.asyncio
    async def test_generate_drafts_cc_skips_failed_draft_and_still_resets(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        config = config.model_copy(update={"max_budget_usd": 0.30})
        git = AsyncMock()
        git.rev_parse = AsyncMock(return_value="sha123")
        git.capture_diff = AsyncMock(return_value="diff")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()

        call_count = 0

        async def _sometimes_fail(cfg: AgentConfig, *args: object, **kwargs: object) -> AgentInvocationResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise AgentInvocationError("first draft failed")
            return AgentInvocationResult(
                success=True, cost_usd=0.01, input_tokens=10, output_tokens=5,
                hypothesis=None, hypothesis_source="synthesized", tags=[], raw_output="ok",
            )

        with patch.object(invoker, "_invoke_claude_code", new=_sometimes_fail):
            # Act
            drafts = await invoker.generate_drafts(config, "prompt", tmp_path, 60, n_drafts=2, git=git)

        # Assert: only 1 successful draft; reset still happened for both
        assert len(drafts) == 1
        assert git.reset_hard.await_count == 2

    @pytest.mark.asyncio
    async def test_generate_drafts_cc_timeout_error_skips_draft(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _cc_config()
        config = config.model_copy(update={"max_budget_usd": 0.30})
        git = AsyncMock()
        git.rev_parse = AsyncMock(return_value="sha123")
        git.capture_diff = AsyncMock(return_value="diff")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()

        with patch.object(
            invoker, "_invoke_claude_code",
            new=AsyncMock(side_effect=AgentTimeoutError("timed out")),
        ):
            # Act
            drafts = await invoker.generate_drafts(config, "prompt", tmp_path, 60, n_drafts=2, git=git)

        # Assert: all drafts failed, but resets still happened
        assert len(drafts) == 0
        assert git.reset_hard.await_count == 2
