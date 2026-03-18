"""Runner integration tests for post-MVP features.

Tests that features F1-F8 are correctly wired through ExperimentRunner.run_one()
and run_loop(). Uses a mock-based approach: mock git/agent I/O boundaries while
letting feature logic (constraints, deployment mode, learning extraction,
held-out eval, meta-optimization) execute for real.
"""

from __future__ import annotations

import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from anneal.engine.agent import AgentInvoker, AgentInvocationResult
from anneal.engine.environment import GitEnvironment
from anneal.engine.eval import EvalEngine
from anneal.engine.learning_pool import LearningPool
from anneal.engine.registry import Registry
from anneal.engine.runner import ExperimentRunner
from anneal.engine.search import GreedySearch, PopulationSearch
from anneal.engine.types import (
    AgentConfig,
    BinaryCriterion,
    ConstraintCommand,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    MetricConstraint,
    OptimizationTarget,
    Outcome,
    StochasticEval,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scope_yaml(path: Path) -> Path:
    scope = path / "scope.yaml"
    scope.write_text(
        "editable:\n  - artifact.md\nimmutable:\n  - scope.yaml\n",
    )
    return scope


def _make_git_repo(path: Path) -> Path:
    """Create a minimal git repo with scope.yaml and artifact.md."""
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=path, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path, capture_output=True, check=True,
    )
    _make_scope_yaml(path)
    (path / "artifact.md").write_text("# Artifact\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=path, capture_output=True, check=True,
    )
    return path


def _make_target(
    worktree_path: str,
    *,
    domain_tier: DomainTier = DomainTier.SANDBOX,
    meta_depth: int = 0,
    constraints: list[MetricConstraint] | None = None,
    constraint_commands: list[ConstraintCommand] | None = None,
    held_out_prompts: list[str] | None = None,
    held_out_interval: int = 1,
    approval_callback: Callable[[str], bool] | None = None,
) -> OptimizationTarget:
    return OptimizationTarget(
        id="test-target",
        domain_tier=domain_tier,
        artifact_paths=["artifact.md"],
        scope_path="scope.yaml",
        scope_hash="",  # Will be set by test
        eval_mode=EvalMode.DETERMINISTIC,
        eval_config=EvalConfig(
            metric_name="score",
            direction=Direction.HIGHER_IS_BETTER,
            deterministic=DeterministicEval(
                run_command="echo 0.9",
                parse_command="cat",
                timeout_seconds=10,
            ),
            constraints=constraints or [],
            constraint_commands=constraint_commands or [],
            held_out_interval=held_out_interval,
            stochastic=StochasticEval(
                sample_count=1,
                criteria=[BinaryCriterion(name="q", question="?")],
                test_prompts=["p"],
                generation_prompt_template="{test_prompt}",
                output_format="text",
                held_out_prompts=held_out_prompts or [],
            ) if held_out_prompts else None,
        ),
        agent_config=AgentConfig(
            mode="api",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
        ),
        time_budget_seconds=60,
        loop_interval_seconds=10,
        knowledge_path=".anneal/targets/test-target",
        worktree_path=worktree_path,
        git_branch="anneal/test-target",
        baseline_score=0.75,
        meta_depth=meta_depth,
        max_consecutive_failures=5,
        approval_callback=approval_callback,
    )


def _mock_agent_result(hypothesis: str = "test mutation") -> AgentInvocationResult:
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


def _build_runner(
    tmp_path: Path,
    *,
    search: GreedySearch | PopulationSearch | None = None,
    learning_pool: LearningPool | None = None,
) -> tuple[ExperimentRunner, AsyncMock, AsyncMock]:
    """Build a runner with mocked git and agent, real eval engine."""
    git = AsyncMock(spec=GitEnvironment)
    git.rev_parse = AsyncMock(return_value="abc123")
    # Alternates: pre-agent snapshot (clean), post-agent (modified)
    _status_call_count = 0

    async def _alternating_status(_worktree: object) -> list[tuple[str, str]]:
        nonlocal _status_call_count
        _status_call_count += 1
        if _status_call_count % 2 == 1:
            return []  # Pre-agent: clean
        return [("M", "artifact.md")]  # Post-agent: modified

    git.status_porcelain = _alternating_status
    git.commit = AsyncMock(return_value="def456")
    git.reset_hard = AsyncMock()
    git.clean_untracked = AsyncMock()
    git.cleanup_index_lock = AsyncMock()
    git.checkout_paths = AsyncMock()

    agent = AsyncMock(spec=AgentInvoker)
    agent.invoke = AsyncMock(return_value=_mock_agent_result())
    agent.invoke_deployment = AsyncMock(return_value=_mock_agent_result())
    agent.invoke_meta = AsyncMock(return_value=_mock_agent_result("meta mutation"))

    registry = MagicMock(spec=Registry)
    registry.update_target = MagicMock()

    runner = ExperimentRunner(
        git=git,
        agent_invoker=agent,
        eval_engine=EvalEngine(),
        search=search or GreedySearch(),
        registry=registry,
        repo_root=tmp_path,
        knowledge=None,
        notifications=None,
        learning_pool=learning_pool,
    )
    return runner, git, agent


# =========================================================================
# F2: Constraint enforcement in run_one
# =========================================================================


class TestRunnerConstraintEnforcement:
    """Verify that constraint violations cause DISCARDED outcome in run_one."""

    @pytest.mark.asyncio
    async def test_failing_constraint_discards(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        # Constraint: coverage >= 90, but script outputs 50
        script = tmp_path / "check_coverage.sh"
        script.write_text("#!/bin/bash\necho 50")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        target = _make_target(
            str(tmp_path),
            constraint_commands=[ConstraintCommand(
                name="coverage",
                run_command=f"bash {script}",
                parse_command="cat",
                timeout_seconds=10,
                threshold=90.0,
                direction=Direction.HIGHER_IS_BETTER,
            )],
        )

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)

        record = await runner.run_one(target)

        assert record.outcome is Outcome.DISCARDED
        assert record.failure_mode is not None
        assert "constraint_violated:coverage" in record.failure_mode

    @pytest.mark.asyncio
    async def test_passing_constraint_keeps(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        script = tmp_path / "check_coverage.sh"
        script.write_text("#!/bin/bash\necho 95")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        target = _make_target(
            str(tmp_path),
            constraint_commands=[ConstraintCommand(
                name="coverage",
                run_command=f"bash {script}",
                parse_command="cat",
                timeout_seconds=10,
                threshold=90.0,
                direction=Direction.HIGHER_IS_BETTER,
            )],
        )

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        record = await runner.run_one(target)

        assert record.outcome is Outcome.KEPT
        assert record.failure_mode is None


# =========================================================================
# F5: Learning pool extraction in run_one
# =========================================================================


class TestRunnerLearningPoolExtraction:
    """Verify that run_one extracts learnings into the pool."""

    @pytest.mark.asyncio
    async def test_learning_extracted_on_kept(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        pool = LearningPool()
        target = _make_target(str(tmp_path))

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path, learning_pool=pool)
        await runner.run_one(target)

        assert pool.count == 1

    @pytest.mark.asyncio
    async def test_learning_extracted_on_discarded(self, tmp_path: Path) -> None:
        """Learnings are extracted even for DISCARDED experiments."""
        _make_git_repo(tmp_path)

        pool = LearningPool()
        # Set eval score lower than baseline to trigger DISCARDED
        target = _make_target(str(tmp_path))
        target.baseline_score = 99.0  # Eval returns 0.9, much lower

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path, learning_pool=pool)
        await runner.run_one(target)

        assert pool.count == 1

    @pytest.mark.asyncio
    async def test_no_pool_no_extraction(self, tmp_path: Path) -> None:
        """When no learning_pool is provided, no extraction occurs."""
        _make_git_repo(tmp_path)

        target = _make_target(str(tmp_path))

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path, learning_pool=None)
        record = await runner.run_one(target)
        # No error — just succeeds without pool
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED)


# =========================================================================
# F6: Deployment-domain runner in run_one
# =========================================================================


class TestRunnerDeploymentMode:
    """Verify deployment mode invocation and approval callback."""

    @pytest.mark.asyncio
    async def test_deployment_tier_uses_invoke_deployment(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        target = _make_target(str(tmp_path), domain_tier=DomainTier.DEPLOYMENT)

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, agent = _build_runner(tmp_path)
        await runner.run_one(target)

        agent.invoke_deployment.assert_called_once()
        agent.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_approval_rejected_discards(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        target = _make_target(
            str(tmp_path),
            domain_tier=DomainTier.DEPLOYMENT,
            approval_callback=lambda _output: False,
        )

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        record = await runner.run_one(target)

        assert record.outcome is Outcome.DISCARDED
        assert record.failure_mode == "approval_rejected"

    @pytest.mark.asyncio
    async def test_approval_accepted_continues(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        target = _make_target(
            str(tmp_path),
            domain_tier=DomainTier.DEPLOYMENT,
            approval_callback=lambda _output: True,
        )

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        record = await runner.run_one(target)

        # Approval passed — should proceed to eval and potentially KEPT
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED)
        assert record.failure_mode != "approval_rejected"

    @pytest.mark.asyncio
    async def test_sandbox_tier_uses_invoke(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        target = _make_target(str(tmp_path), domain_tier=DomainTier.SANDBOX)

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, agent = _build_runner(tmp_path)
        await runner.run_one(target)

        agent.invoke.assert_called_once()
        agent.invoke_deployment.assert_not_called()


# =========================================================================
# F8: Meta-optimization plateau detection in run_loop
# =========================================================================


class TestRunnerMetaOptimization:
    """Verify meta-optimization triggers on plateau in run_loop."""

    @pytest.mark.asyncio
    async def test_meta_triggers_on_plateau(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        target = _make_target(
            str(tmp_path),
            meta_depth=1,
        )
        target.max_consecutive_failures = 3
        target.baseline_score = 99.0  # All experiments will DISCARD (score=0.9)

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        # Create program.md so meta-optimization can find it
        program_dir = tmp_path / target.knowledge_path
        program_dir.mkdir(parents=True, exist_ok=True)
        (program_dir / "program.md").write_text("# Optimization Strategy\n")

        runner, _, agent = _build_runner(tmp_path)

        # Patch _write_status to avoid filesystem issues
        runner._write_status = AsyncMock()

        # Run 4 experiments — all DISCARDED, plateau threshold is 3
        records = await runner.run_loop(target, max_experiments=4)

        # Meta should have been triggered (3 consecutive non-KEPT >= meta_m=3)
        agent.invoke_meta.assert_called()

    @pytest.mark.asyncio
    async def test_meta_not_triggered_when_depth_zero(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        target = _make_target(str(tmp_path), meta_depth=0)
        target.baseline_score = 99.0
        target.max_consecutive_failures = 10

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, agent = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        await runner.run_loop(target, max_experiments=4)

        agent.invoke_meta.assert_not_called()


# =========================================================================
# F1: Held-out eval in run_loop (via kept count tracking)
# =========================================================================


class TestRunnerHeldOutTracking:
    """Verify kept_count tracking and held-out interval logic in run_loop."""

    @pytest.mark.asyncio
    async def test_kept_count_increments(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        target = _make_target(str(tmp_path))

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        records = await runner.run_loop(target, max_experiments=3)

        # With eval returning 0.9 > baseline 0.75, all should be KEPT
        kept = [r for r in records if r.outcome is Outcome.KEPT]
        assert len(kept) >= 1


# =========================================================================
# Cross-feature: Constraint + Learning Pool
# =========================================================================


class TestConstraintWithLearningPool:
    """Verify constraint-violated experiments still extract learnings."""

    @pytest.mark.asyncio
    async def test_constraint_violation_still_extracts_learning(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        pool = LearningPool()
        script = tmp_path / "fail_check.sh"
        script.write_text("#!/bin/bash\necho 10")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        target = _make_target(
            str(tmp_path),
            constraint_commands=[ConstraintCommand(
                name="quality_gate",
                run_command=f"bash {script}",
                parse_command="cat",
                timeout_seconds=10,
                threshold=50.0,
                direction=Direction.HIGHER_IS_BETTER,
            )],
        )

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path, learning_pool=pool)
        record = await runner.run_one(target)

        assert record.outcome is Outcome.DISCARDED
        assert "constraint_violated" in (record.failure_mode or "")
        assert pool.count == 1  # Learning still extracted


# =========================================================================
# Cross-feature: Deployment + Constraint
# =========================================================================


class TestDeploymentWithConstraint:
    """Verify deployment approval rejection short-circuits before constraints."""

    @pytest.mark.asyncio
    async def test_rejection_bypasses_eval_and_constraints(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)

        target = _make_target(
            str(tmp_path),
            domain_tier=DomainTier.DEPLOYMENT,
            approval_callback=lambda _output: False,
            constraint_commands=[ConstraintCommand(
                name="never_reached",
                run_command="exit 1",  # Would fail if reached
                parse_command="cat",
                timeout_seconds=10,
                threshold=0.0,
                direction=Direction.HIGHER_IS_BETTER,
            )],
        )

        from anneal.engine.scope import compute_scope_hash
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        record = await runner.run_one(target)

        # Rejected at approval gate, never reaches eval or constraint check
        assert record.outcome is Outcome.DISCARDED
        assert record.failure_mode == "approval_rejected"
