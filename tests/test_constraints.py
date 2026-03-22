"""Tests for CompositeMetric constraint mode (F2): EvalEngine.check_constraints."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from anneal.engine.eval import EvalEngine
from anneal.engine.types import (
    BinaryCriterion,
    ConstraintCommand,
    Direction,
    EvalConfig,
    StochasticEval,
)


def _make_eval_config(
    *,
    min_criterion_scores: dict[str, float] | None = None,
    constraint_commands: list[ConstraintCommand] | None = None,
    stochastic: bool = False,
) -> EvalConfig:
    stochastic_cfg = None
    if stochastic or min_criterion_scores:
        stochastic_cfg = StochasticEval(
            sample_count=5,
            criteria=[BinaryCriterion(name="relevant", question="Is it relevant?")],
            test_prompts=["prompt"],
            generation_prompt_template="Generate: {test_prompt}",
            output_format="text",
            min_criterion_scores=min_criterion_scores or {},
        )
    return EvalConfig(
        metric_name="accuracy",
        direction=Direction.HIGHER_IS_BETTER,
        stochastic=stochastic_cfg,
        constraint_commands=constraint_commands or [],
    )


class TestCheckConstraintsStochastic:
    """Tests for stochastic min_criterion_scores constraints."""

    @pytest.mark.asyncio
    async def test_all_criteria_pass(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config(
            min_criterion_scores={"relevant": 0.7, "coherent": 0.5},
        )
        per_criterion = {"relevant": 0.8, "coherent": 0.6}
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores=per_criterion,
        )
        assert len(results) == 2
        assert all(passed for _, passed, _ in results)

    @pytest.mark.asyncio
    async def test_one_criterion_fails(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config(
            min_criterion_scores={"relevant": 0.7, "coherent": 0.9},
        )
        per_criterion = {"relevant": 0.8, "coherent": 0.5}
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores=per_criterion,
        )
        names_and_pass = {name: passed for name, passed, _ in results}
        assert names_and_pass["relevant"] is True
        assert names_and_pass["coherent"] is False

    @pytest.mark.asyncio
    async def test_missing_criterion_defaults_to_zero(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config(
            min_criterion_scores={"missing_criterion": 0.5},
        )
        per_criterion = {"relevant": 0.8}
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores=per_criterion,
        )
        assert len(results) == 1
        name, passed, actual = results[0]
        assert name == "missing_criterion"
        assert passed is False
        assert actual == 0.0

    @pytest.mark.asyncio
    async def test_exact_threshold_passes(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config(min_criterion_scores={"relevant": 0.7})
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores={"relevant": 0.7},
        )
        _, passed, _ = results[0]
        assert passed is True

    @pytest.mark.asyncio
    async def test_no_constraints_returns_empty(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config()
        results = await engine.check_constraints(tmp_path, config)
        assert results == []

    @pytest.mark.asyncio
    async def test_no_per_criterion_scores_skips_stochastic(self, tmp_path: Path) -> None:
        engine = EvalEngine()
        config = _make_eval_config(min_criterion_scores={"relevant": 0.7})
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores=None,
        )
        assert results == []


class TestCheckConstraintsDeterministic:
    """Tests for deterministic constraint_commands."""

    @pytest.mark.asyncio
    async def test_constraint_command_passes(self, tmp_path: Path) -> None:
        script = tmp_path / "check.sh"
        script.write_text("#!/bin/bash\necho 95")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        cmd = ConstraintCommand(
            name="coverage",
            run_command=f"bash {script}",
            parse_command="cat",
            timeout_seconds=10,
            threshold=90.0,
            direction=Direction.HIGHER_IS_BETTER,
        )
        engine = EvalEngine()
        config = _make_eval_config(constraint_commands=[cmd])
        results = await engine.check_constraints(tmp_path, config)
        assert len(results) == 1
        name, passed, actual = results[0]
        assert name == "coverage"
        assert passed is True
        assert actual == pytest.approx(95.0)

    @pytest.mark.asyncio
    async def test_constraint_command_fails(self, tmp_path: Path) -> None:
        script = tmp_path / "check.sh"
        script.write_text("#!/bin/bash\necho 85")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        cmd = ConstraintCommand(
            name="coverage",
            run_command=f"bash {script}",
            parse_command="cat",
            timeout_seconds=10,
            threshold=90.0,
            direction=Direction.HIGHER_IS_BETTER,
        )
        engine = EvalEngine()
        config = _make_eval_config(constraint_commands=[cmd])
        results = await engine.check_constraints(tmp_path, config)
        name, passed, actual = results[0]
        assert name == "coverage"
        assert passed is False
        assert actual == pytest.approx(85.0)

    @pytest.mark.asyncio
    async def test_lower_is_better_direction(self, tmp_path: Path) -> None:
        script = tmp_path / "latency.sh"
        script.write_text("#!/bin/bash\necho 50")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        cmd = ConstraintCommand(
            name="latency",
            run_command=f"bash {script}",
            parse_command="cat",
            timeout_seconds=10,
            threshold=100.0,
            direction=Direction.LOWER_IS_BETTER,
        )
        engine = EvalEngine()
        config = _make_eval_config(constraint_commands=[cmd])
        results = await engine.check_constraints(tmp_path, config)
        _, passed, _ = results[0]
        assert passed is True

    @pytest.mark.asyncio
    async def test_mixed_stochastic_and_deterministic(self, tmp_path: Path) -> None:
        script = tmp_path / "check.sh"
        script.write_text("#!/bin/bash\necho 95")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        cmd = ConstraintCommand(
            name="coverage",
            run_command=f"bash {script}",
            parse_command="cat",
            timeout_seconds=10,
            threshold=90.0,
            direction=Direction.HIGHER_IS_BETTER,
        )
        config = EvalConfig(
            metric_name="accuracy",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=StochasticEval(
                sample_count=5,
                criteria=[BinaryCriterion(name="relevant", question="?")],
                test_prompts=["p"],
                generation_prompt_template="G: {test_prompt}",
                output_format="text",
                min_criterion_scores={"relevant": 0.7},
            ),
            constraint_commands=[cmd],
        )
        engine = EvalEngine()
        results = await engine.check_constraints(
            tmp_path, config, per_criterion_scores={"relevant": 0.8},
        )
        assert len(results) == 2
        names = [name for name, _, _ in results]
        assert "relevant" in names
        assert "coverage" in names


# ---------------------------------------------------------------------------
# Step 6.3: Fast constraint pre-check skips expensive eval
# ---------------------------------------------------------------------------


class TestFastConstraintPreCheck:
    """Verify that a failing constraint pre-check skips the evaluate call."""

    @pytest.mark.asyncio
    async def test_failing_constraint_skips_evaluate(self, tmp_path: Path) -> None:
        import subprocess
        import stat as stat_mod
        from unittest.mock import AsyncMock, MagicMock, patch

        from anneal.engine.agent import AgentInvoker, AgentInvocationResult
        from anneal.engine.environment import GitEnvironment
        from anneal.engine.eval import EvalEngine
        from anneal.engine.registry import Registry
        from anneal.engine.runner import ExperimentRunner
        from anneal.engine.scope import compute_scope_hash
        from anneal.engine.search import GreedySearch
        from anneal.engine.types import (
            AgentConfig,
            ConstraintCommand,
            DeterministicEval,
            Direction,
            DomainTier,
            EvalConfig,
            EvalMode,
            OptimizationTarget,
            Outcome,
        )

        # Set up a real git repo for scope verification
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True, check=True)
        scope = tmp_path / "scope.yaml"
        scope.write_text("editable:\n  - artifact.md\nimmutable:\n  - scope.yaml\n")
        (tmp_path / "artifact.md").write_text("# Artifact\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True)

        # Constraint script that fails
        script = tmp_path / "check.sh"
        script.write_text("#!/bin/bash\necho 10")
        script.chmod(script.stat().st_mode | stat_mod.S_IEXEC)

        constraint = ConstraintCommand(
            name="coverage",
            run_command=f"bash {script}",
            parse_command="cat",
            timeout_seconds=10,
            threshold=90.0,
            direction=Direction.HIGHER_IS_BETTER,
        )

        target = OptimizationTarget(
            id="test-target",
            domain_tier=DomainTier.SANDBOX,
            artifact_paths=["artifact.md"],
            scope_path="scope.yaml",
            scope_hash=compute_scope_hash(scope),
            eval_mode=EvalMode.DETERMINISTIC,
            eval_config=EvalConfig(
                metric_name="score",
                direction=Direction.HIGHER_IS_BETTER,
                deterministic=DeterministicEval(
                    run_command="echo 0.9",
                    parse_command="cat",
                    timeout_seconds=10,
                ),
                constraint_commands=[constraint],
            ),
            agent_config=AgentConfig(
                mode="api",
                model="gpt-4.1",
                evaluator_model="gpt-4.1-mini",
            ),
            time_budget_seconds=60,
            loop_interval_seconds=10,
            knowledge_path=".anneal/targets/test-target",
            worktree_path=str(tmp_path),
            git_branch="anneal/test-target",
            baseline_score=0.75,
            meta_depth=0,
            max_consecutive_failures=5,
        )

        # Build runner with mocked git and agent
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
        agent.invoke = AsyncMock(return_value=AgentInvocationResult(
            success=True,
            cost_usd=0.01,
            input_tokens=100,
            output_tokens=50,
            hypothesis="test mutation",
            hypothesis_source="agent",
            tags=["test"],
            raw_output="## Hypothesis\ntest mutation\n## Tags\ntest",
        ))

        registry = MagicMock(spec=Registry)
        registry.update_target = MagicMock()

        eval_engine = EvalEngine()
        evaluate_mock = AsyncMock(side_effect=AssertionError("evaluate must not be called when fast constraint fails"))

        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=eval_engine,
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
        )

        with patch.object(eval_engine, "evaluate", evaluate_mock):
            record = await runner.run_one(target)

        assert record.outcome is Outcome.DISCARDED
        assert record.failure_mode == "constraint_violated:coverage"
        evaluate_mock.assert_not_called()
