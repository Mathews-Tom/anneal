"""Tests for two-phase mutation: diagnosis step and its integration with the runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anneal.engine.agent import AgentInvocationError, AgentInvoker
from anneal.engine.environment import GitEnvironment
from anneal.engine.eval import EvalEngine
from anneal.engine.registry import Registry
from anneal.engine.runner import ExperimentRunner
from anneal.engine.scope import compute_scope_hash
from anneal.engine.search import GreedySearch
from anneal.engine.types import (
    AgentConfig,
    AgentInvocationResult,
    DeterministicEval,
    DiagnosisResult,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    EvalResult,
    ExperimentRecord,
    OptimizationTarget,
    Outcome,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config(
    *,
    two_phase_mutation: bool = False,
    diagnosis_model: str = "",
    exploration_model: str = "",
    model: str = "gpt-4.1",
) -> AgentConfig:
    return AgentConfig(
        mode="api",
        model=model,
        evaluator_model="gpt-4.1-mini",
        two_phase_mutation=two_phase_mutation,
        diagnosis_model=diagnosis_model,
        exploration_model=exploration_model,
    )


def _make_eval_result(
    score: float = 0.5,
    per_criterion_scores: dict[str, float] | None = None,
) -> EvalResult:
    return EvalResult(
        score=score,
        per_criterion_scores=per_criterion_scores or {"clarity": 0.4, "depth": 0.6},
    )


def _make_mock_openai_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> MagicMock:
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


def _make_agent_invocation_result(hypothesis: str = "improve the structure") -> AgentInvocationResult:
    return AgentInvocationResult(
        success=True,
        cost_usd=0.01,
        input_tokens=100,
        output_tokens=50,
        hypothesis=hypothesis,
        hypothesis_source="agent",
        tags=["test"],
        raw_output=f"## Hypothesis\n{hypothesis}\n## Tags\ntest",
    )


def _make_git_repo(path: Path) -> tuple[Path, str]:
    """Create a minimal git repo; return (path, scope_hash)."""
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, capture_output=True, check=True)
    scope_content = "editable:\n  - artifact.md\nimmutable:\n  - scope.yaml\n"
    (path / "scope.yaml").write_text(scope_content)
    (path / "artifact.md").write_text("# Artifact\nSome content.\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, capture_output=True, check=True)
    scope_hash = compute_scope_hash(path / "scope.yaml")
    return path, scope_hash


def _make_target(
    worktree_path: str,
    *,
    scope_hash: str = "",
    two_phase_mutation: bool = False,
    diagnosis_model: str = "",
    exploration_model: str = "",
    model: str = "gpt-4.1",
    inject_knowledge_context: bool = False,
) -> OptimizationTarget:
    return OptimizationTarget(
        id="test-target",
        domain_tier=DomainTier.SANDBOX,
        artifact_paths=["artifact.md"],
        scope_path="scope.yaml",
        scope_hash=scope_hash,
        eval_mode=EvalMode.DETERMINISTIC,
        eval_config=EvalConfig(
            metric_name="score",
            direction=Direction.HIGHER_IS_BETTER,
            deterministic=DeterministicEval(
                run_command="echo 0.9",
                parse_command="cat",
                timeout_seconds=10,
            ),
        ),
        agent_config=_make_agent_config(
            two_phase_mutation=two_phase_mutation,
            diagnosis_model=diagnosis_model,
            exploration_model=exploration_model,
            model=model,
        ),
        time_budget_seconds=60,
        loop_interval_seconds=0,
        knowledge_path=".anneal/targets/test-target",
        worktree_path=worktree_path,
        git_branch="anneal/test-target",
        baseline_score=0.75,
        max_consecutive_failures=5,
        inject_knowledge_context=inject_knowledge_context,
    )


def _build_runner(
    tmp_path: Path,
) -> tuple[ExperimentRunner, AsyncMock, AsyncMock]:
    git = AsyncMock(spec=GitEnvironment)
    git.rev_parse = AsyncMock(return_value="abc123")

    _call_count = 0

    async def _alternating_status(_worktree: object) -> list[tuple[str, str]]:
        nonlocal _call_count
        _call_count += 1
        if _call_count % 2 == 1:
            return []
        return [("M", "artifact.md")]

    git.status_porcelain = _alternating_status
    git.commit = AsyncMock(return_value="def456")
    git.reset_hard = AsyncMock()
    git.clean_untracked = AsyncMock()
    git.cleanup_index_lock = AsyncMock()
    git.checkout_paths = AsyncMock()

    agent = AsyncMock(spec=AgentInvoker)
    agent.invoke = AsyncMock(return_value=_make_agent_invocation_result())

    registry = MagicMock(spec=Registry)
    registry.update_target = MagicMock()

    runner = ExperimentRunner(
        git=git,
        agent_invoker=agent,
        eval_engine=EvalEngine(),
        search=GreedySearch(),
        registry=registry,
        repo_root=tmp_path,
        knowledge=None,
        notifications=None,
        learning_pool=None,
    )
    return runner, git, agent


# ---------------------------------------------------------------------------
# Unit: diagnose()
# ---------------------------------------------------------------------------


class TestDiagnoseReturnsStructuredResult:
    """diagnose() with a valid JSON response populates DiagnosisResult correctly."""

    @pytest.mark.asyncio
    async def test_diagnosis_returns_structured_result(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _make_agent_config(model="gpt-4.1")
        eval_result = _make_eval_result()
        payload = {
            "weakest_criteria": ["clarity", "depth"],
            "root_cause": "The artifact lacks specificity in key sections.",
            "fix_category": "content",
            "suggested_direction": "Add concrete examples and expand on weak points.",
        }
        mock_response = _make_mock_openai_response(json.dumps(payload))

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.compute_cost", return_value=0.0025),
        ):
            # Act
            result = await invoker.diagnose(config, "artifact text", eval_result, [], tmp_path)

        # Assert
        assert result.weakest_criteria == ["clarity", "depth"]
        assert result.root_cause == "The artifact lacks specificity in key sections."
        assert result.fix_category == "content"
        assert result.suggested_direction == "Add concrete examples and expand on weak points."
        assert result.cost_usd == 0.0025


class TestDiagnoseInvalidJsonRaises:
    """diagnose() raises AgentInvocationError when the API returns malformed JSON."""

    @pytest.mark.asyncio
    async def test_diagnosis_invalid_json_raises(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _make_agent_config(model="gpt-4.1")
        eval_result = _make_eval_result()
        mock_response = _make_mock_openai_response("this is not json {{{")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.compute_cost", return_value=0.0),
        ):
            # Act / Assert
            with pytest.raises(AgentInvocationError, match="invalid JSON"):
                await invoker.diagnose(config, "artifact text", eval_result, [], tmp_path)


class TestDiagnosisCostTracked:
    """diagnose() cost is reflected in the returned DiagnosisResult.cost_usd."""

    @pytest.mark.asyncio
    async def test_diagnosis_cost_tracked(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _make_agent_config(model="gpt-4.1")
        eval_result = _make_eval_result()
        payload = {
            "weakest_criteria": ["tone"],
            "root_cause": "Tone is too casual.",
            "fix_category": "formatting",
            "suggested_direction": "Use formal language throughout.",
        }
        mock_response = _make_mock_openai_response(
            json.dumps(payload), prompt_tokens=200, completion_tokens=80
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        expected_cost = 0.0042

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client),
            patch("anneal.engine.agent.compute_cost", return_value=expected_cost) as mock_cost,
        ):
            # Act
            result = await invoker.diagnose(config, "artifact text", eval_result, [], tmp_path)

        # Assert
        mock_cost.assert_called_once_with(config.model, 200, 80)
        assert result.cost_usd == expected_cost


class TestDiagnosisUsesExplorationModel:
    """diagnosis_model="" + exploration_model set → diagnose() uses exploration_model."""

    @pytest.mark.asyncio
    async def test_diagnosis_uses_exploration_model(self, tmp_path: Path) -> None:
        # Arrange
        invoker = AgentInvoker()
        config = _make_agent_config(
            model="gpt-4.1",
            diagnosis_model="",
            exploration_model="gpt-4.1-mini",
        )
        eval_result = _make_eval_result()
        payload = {
            "weakest_criteria": ["structure"],
            "root_cause": "Missing headers.",
            "fix_category": "structural",
            "suggested_direction": "Add clear section headers.",
        }
        mock_response = _make_mock_openai_response(json.dumps(payload))

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("anneal.engine.agent.make_client", return_value=mock_client) as mock_make_client,
            patch("anneal.engine.agent.compute_cost", return_value=0.001),
        ):
            # Act
            await invoker.diagnose(config, "artifact text", eval_result, [], tmp_path)

        # Assert: make_client called with exploration_model, not primary model
        mock_make_client.assert_called_once_with("gpt-4.1-mini")


# ---------------------------------------------------------------------------
# Integration: runner.run_one with two-phase mutation
# ---------------------------------------------------------------------------


def _make_history_record(
    *,
    score: float = 0.5,
    criterion_names: list[str] | None = None,
    per_criterion_scores: dict[str, float] | None = None,
) -> ExperimentRecord:
    """Build a minimal ExperimentRecord for seeding the knowledge store mock."""
    import datetime

    return ExperimentRecord(
        id="prev-1",
        target_id="test-target",
        git_sha="abc123",
        pre_experiment_sha="abc000",
        timestamp=datetime.datetime.now(),
        hypothesis="previous attempt",
        hypothesis_source="agent",
        mutation_diff_summary="diff",
        score=score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=None,
        baseline_score=0.75,
        outcome=Outcome.DISCARDED,
        failure_mode=None,
        duration_seconds=1.0,
        tags=[],
        learnings="",
        cost_usd=0.01,
        bootstrap_seed=42,
        criterion_names=criterion_names,
        per_criterion_scores=per_criterion_scores,
    )


def _inject_knowledge(runner: ExperimentRunner, records: list[ExperimentRecord]) -> None:
    """Attach a mock knowledge store to runner._knowledge that returns `records`."""
    knowledge = MagicMock()
    knowledge.load_records = MagicMock(return_value=records)
    knowledge.get_context = MagicMock(return_value="")
    knowledge.append_record = MagicMock()
    knowledge.update_index = MagicMock()
    runner._knowledge = knowledge  # noqa: SLF001


class TestMutationPromptContainsDiagnosis:
    """When two_phase_mutation=True, the mutation prompt receives the diagnosis context."""

    @pytest.mark.asyncio
    async def test_two_phase_mutation_injects_diagnosis(self, tmp_path: Path) -> None:
        # Arrange
        _, scope_hash = _make_git_repo(tmp_path)
        runner, git, agent = _build_runner(tmp_path)
        target = _make_target(
            str(tmp_path),
            scope_hash=scope_hash,
            two_phase_mutation=True,
            inject_knowledge_context=True,
        )
        history_record = _make_history_record(
            score=0.5,
            criterion_names=["clarity"],
            per_criterion_scores={"clarity": 0.4},
        )
        _inject_knowledge(runner, [history_record])

        diagnosis_result = DiagnosisResult(
            weakest_criteria=["clarity"],
            root_cause="Sentences are too long and lack examples.",
            fix_category="content",
            suggested_direction="Break up long sentences and add examples.",
            cost_usd=0.003,
        )
        agent.diagnose = AsyncMock(return_value=diagnosis_result)

        captured_prompts: list[str] = []

        async def _capture_invoke(config: AgentConfig, prompt: str, *args: object, **kwargs: object) -> AgentInvocationResult:
            captured_prompts.append(prompt)
            return _make_agent_invocation_result()

        agent.invoke = _capture_invoke

        # Act
        await runner.run_one(target)

        # Assert
        assert len(captured_prompts) == 1
        assert "## Diagnosis" in captured_prompts[0]
        assert "clarity" in captured_prompts[0]
        assert "Sentences are too long" in captured_prompts[0]


class TestSinglePhaseUnchanged:
    """two_phase_mutation=False leaves behavior unchanged — diagnose() is never called."""

    @pytest.mark.asyncio
    async def test_single_phase_unchanged(self, tmp_path: Path) -> None:
        # Arrange
        _, scope_hash = _make_git_repo(tmp_path)
        runner, git, agent = _build_runner(tmp_path)
        target = _make_target(str(tmp_path), scope_hash=scope_hash, two_phase_mutation=False)
        agent.diagnose = AsyncMock()

        captured_prompts: list[str] = []

        async def _capture_invoke(config: AgentConfig, prompt: str, *args: object, **kwargs: object) -> AgentInvocationResult:
            captured_prompts.append(prompt)
            return _make_agent_invocation_result()

        agent.invoke = _capture_invoke

        # Act — no knowledge store, history stays empty; diagnose must not be called
        await runner.run_one(target)

        # Assert
        agent.diagnose.assert_not_called()
        assert len(captured_prompts) == 1
        assert "## Diagnosis" not in captured_prompts[0]


class TestDiagnosisPopulatesLearnings:
    """ExperimentRecord.learnings contains the diagnosis text when two_phase_mutation=True."""

    @pytest.mark.asyncio
    async def test_diagnosis_populates_learnings(self, tmp_path: Path) -> None:
        # Arrange
        _, scope_hash = _make_git_repo(tmp_path)
        runner, git, agent = _build_runner(tmp_path)
        target = _make_target(
            str(tmp_path),
            scope_hash=scope_hash,
            two_phase_mutation=True,
            inject_knowledge_context=True,
        )
        history_record = _make_history_record(
            score=0.5,
            criterion_names=["depth"],
            per_criterion_scores={"depth": 0.3},
        )
        _inject_knowledge(runner, [history_record])

        diagnosis_result = DiagnosisResult(
            weakest_criteria=["depth"],
            root_cause="Coverage of the topic is too shallow.",
            fix_category="coverage",
            suggested_direction="Expand each section with additional detail.",
            cost_usd=0.002,
        )
        agent.diagnose = AsyncMock(return_value=diagnosis_result)
        agent.invoke = AsyncMock(return_value=_make_agent_invocation_result())

        # Act
        record = await runner.run_one(target)

        # Assert
        assert record is not None
        assert "Coverage of the topic is too shallow." in record.learnings
        assert "Expand each section with additional detail." in record.learnings
        assert "coverage" in record.learnings


class TestDiagnosisCostIncludedInRecord:
    """ExperimentRecord.cost_usd includes the diagnosis cost on top of the agent cost."""

    @pytest.mark.asyncio
    async def test_diagnosis_cost_included_in_record(self, tmp_path: Path) -> None:
        # Arrange
        _, scope_hash = _make_git_repo(tmp_path)
        runner, git, agent = _build_runner(tmp_path)
        target = _make_target(
            str(tmp_path),
            scope_hash=scope_hash,
            two_phase_mutation=True,
            inject_knowledge_context=True,
        )
        history_record = _make_history_record(score=0.5)
        _inject_knowledge(runner, [history_record])

        diagnosis_cost = 0.005
        agent_cost = 0.01
        diagnosis_result = DiagnosisResult(
            weakest_criteria=["logic"],
            root_cause="Argument flow is broken.",
            fix_category="logic",
            suggested_direction="Reorder arguments for coherence.",
            cost_usd=diagnosis_cost,
        )
        agent.diagnose = AsyncMock(return_value=diagnosis_result)
        agent.invoke = AsyncMock(
            return_value=AgentInvocationResult(
                success=True,
                cost_usd=agent_cost,
                input_tokens=100,
                output_tokens=50,
                hypothesis="improve logic",
                hypothesis_source="agent",
                tags=[],
                raw_output="## Hypothesis\nimprove logic\n",
            )
        )

        # Act
        record = await runner.run_one(target)

        # Assert
        assert record is not None
        # Record cost must include both diagnosis cost and agent invocation cost
        assert record.cost_usd >= diagnosis_cost + agent_cost
