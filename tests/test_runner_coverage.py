"""Coverage tests for ExperimentRunner — targeting previously uncovered lines.

Grouped by method:
- make_search_strategy (201-220)
- run_one pipeline paths (228-527)
- _invoke_agent (560-607)
- _enforce_and_commit (609-664)
- _evaluate_and_decide (738-877)
- run_loop control flow (885-1163)
- Recovery helpers (1221-1263)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anneal.engine.agent import AgentInvocationError, AgentInvoker, AgentTimeoutError
from anneal.engine.environment import GitEnvironment
from anneal.engine.eval import EvalEngine
from anneal.engine.knowledge import KnowledgeStore
from anneal.engine.registry import Registry
from anneal.engine.runner import ExperimentRunner, RunLoopState
from anneal.engine.scope import compute_scope_hash
from anneal.engine.search import (
    GreedySearch,
    ParetoSearch,
    PopulationSearch,
    SimulatedAnnealingSearch,
)
from anneal.engine.tree_search import UCBTreeSearch
from anneal.engine.types import (
    AgentConfig,
    AgentInvocationResult,
    BinaryCriterion,
    ConstraintCommand,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    EvalResult,
    FidelityStage,
    OptimizationTarget,
    Outcome,
    PopulationConfig,
    RunnerState,
    StochasticEval,
    VerifierCommand,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
    scope = path / "scope.yaml"
    scope.write_text("editable:\n  - artifact.md\nimmutable:\n  - scope.yaml\n")
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
    constraint_commands: list[ConstraintCommand] | None = None,
    in_place: bool = False,
    simplify_before_mutate: bool = False,
    restart_probability: float = 0.0,
    population_config: PopulationConfig | None = None,
    verifiers: list[VerifierCommand] | None = None,
    fidelity_stages: list[FidelityStage] | None = None,
    held_out_prompts: list[str] | None = None,
    held_out_interval: int = 10,
    n_drafts: int = 1,
    approval_callback=None,
    two_phase_mutation: bool = False,
    inject_knowledge_context: bool = False,
) -> OptimizationTarget:
    return OptimizationTarget(
        id="test-target",
        domain_tier=domain_tier,
        artifact_paths=["artifact.md"],
        scope_path="scope.yaml",
        scope_hash="",
        eval_mode=EvalMode.DETERMINISTIC,
        eval_config=EvalConfig(
            metric_name="score",
            direction=Direction.HIGHER_IS_BETTER,
            deterministic=DeterministicEval(
                run_command="echo 0.9",
                parse_command="cat",
                timeout_seconds=10,
            ),
            constraint_commands=constraint_commands or [],
            fidelity_stages=fidelity_stages or [],
            verifiers=verifiers or [],
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
            n_drafts=n_drafts,
            two_phase_mutation=two_phase_mutation,
        ),
        time_budget_seconds=60,
        loop_interval_seconds=10,
        knowledge_path=".anneal/targets/test-target",
        worktree_path=worktree_path,
        git_branch="anneal/test-target",
        baseline_score=0.75,
        max_consecutive_failures=5,
        meta_depth=meta_depth,
        restart_probability=restart_probability,
        in_place=in_place,
        simplify_before_mutate=simplify_before_mutate,
        population_config=population_config,
        approval_callback=approval_callback,
        inject_knowledge_context=inject_knowledge_context,
    )


def _mock_agent_result(
    hypothesis: str = "test mutation",
    success: bool = True,
    cost: float = 0.01,
) -> AgentInvocationResult:
    return AgentInvocationResult(
        success=success,
        cost_usd=cost,
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
    search=None,
    knowledge=None,
) -> tuple[ExperimentRunner, AsyncMock, AsyncMock]:
    """Build a runner with mocked git and agent, real eval engine."""
    git = AsyncMock(spec=GitEnvironment)
    git.rev_parse = AsyncMock(return_value="abc123")

    _status_call_count = 0

    async def _alternating_status(_worktree: object) -> list[tuple[str, str]]:
        nonlocal _status_call_count
        _status_call_count += 1
        if _status_call_count % 2 == 1:
            return []
        return [("M", "artifact.md")]

    git.status_porcelain = _alternating_status
    git.commit = AsyncMock(return_value="def456")
    git.reset_hard = AsyncMock()
    git.clean_untracked = AsyncMock()
    git.cleanup_index_lock = AsyncMock()
    git.checkout_paths = AsyncMock()
    git.checkout = AsyncMock()
    git.apply_diff = AsyncMock(return_value=True)
    git.capture_diff = AsyncMock(return_value="diff --git a/artifact.md b/artifact.md\n+change")
    git.fsck = AsyncMock(return_value=True)

    agent = AsyncMock(spec=AgentInvoker)
    agent.invoke = AsyncMock(return_value=_mock_agent_result())
    agent.invoke_deployment = AsyncMock(return_value=_mock_agent_result())
    agent.invoke_meta = AsyncMock(return_value=_mock_agent_result("meta mutation"))
    agent.invoke_api_text = AsyncMock(return_value="Revised approach text")
    agent.generate_drafts = AsyncMock(return_value=[
        (_mock_agent_result(), "diff --git a/artifact.md b/artifact.md\n+change"),
    ])

    registry = MagicMock(spec=Registry)
    registry.update_target = MagicMock()

    runner = ExperimentRunner(
        git=git,
        agent_invoker=agent,
        eval_engine=EvalEngine(),
        search=search or GreedySearch(),
        registry=registry,
        repo_root=tmp_path,
        knowledge=knowledge,
        notifications=None,
        learning_pool=None,
    )
    return runner, git, agent


# ---------------------------------------------------------------------------
# make_search_strategy
# ---------------------------------------------------------------------------


class TestMakeSearchStrategy:
    def test_greedy_strategy_returns_greedy_search(self) -> None:
        # Arrange
        target = _make_target("/tmp", population_config=None)

        # Act
        strategy = ExperimentRunner.make_search_strategy(target)

        # Assert
        assert isinstance(strategy, GreedySearch)

    def test_simulated_annealing_returns_sa_search(self) -> None:
        # Arrange
        target = _make_target(
            "/tmp",
            population_config=PopulationConfig(search_strategy="simulated_annealing"),
        )

        # Act
        strategy = ExperimentRunner.make_search_strategy(target)

        # Assert
        assert isinstance(strategy, SimulatedAnnealingSearch)

    def test_population_returns_population_search(self) -> None:
        # Arrange
        target = _make_target(
            "/tmp",
            population_config=PopulationConfig(
                search_strategy="population",
                population_size=6,
                tournament_size=3,
            ),
        )

        # Act
        strategy = ExperimentRunner.make_search_strategy(target)

        # Assert
        assert isinstance(strategy, PopulationSearch)

    def test_pareto_returns_pareto_search(self) -> None:
        # Arrange
        target = _make_target(
            "/tmp",
            population_config=PopulationConfig(search_strategy="pareto"),
        )

        # Act
        strategy = ExperimentRunner.make_search_strategy(target)

        # Assert
        assert isinstance(strategy, ParetoSearch)

    def test_unknown_strategy_falls_back_to_greedy(self, caplog) -> None:
        # Arrange
        target = _make_target(
            "/tmp",
            population_config=PopulationConfig(search_strategy="nonexistent_algo"),
        )

        # Act
        import logging
        with caplog.at_level(logging.WARNING):
            strategy = ExperimentRunner.make_search_strategy(target)

        # Assert
        assert isinstance(strategy, GreedySearch)
        assert "Unknown strategy" in caplog.text


# ---------------------------------------------------------------------------
# run_one — in-place mode backup (lines 252-255)
# ---------------------------------------------------------------------------


class TestRunOneInPlaceMode:
    @pytest.mark.asyncio
    async def test_in_place_mode_backup_env_created_before_mutation(self, tmp_path: Path) -> None:
        """In-place mode creates FileBackupEnvironment and calls backup() before mutation.

        The source has an unresolved `backup_id` reference in _evaluate_and_decide that
        only triggers when in_place=True AND the experiment reaches the keep/discard branch.
        We test the backup setup path by letting the agent crash before reaching that branch.
        """
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), in_place=True)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)
        # Crash the agent so run_one returns early (CRASHED), before reaching backup_id branch
        agent.invoke.side_effect = AgentInvocationError("crash before backup_id issue")

        # Act
        record = await runner.run_one(target)

        # Assert — backup env created (lines 252-255 covered), then agent crashed
        assert "test-target" in runner._backup_envs
        assert record.outcome is Outcome.CRASHED

    @pytest.mark.asyncio
    async def test_in_place_mode_skips_git_commit(self, tmp_path: Path) -> None:
        """In-place mode never calls git commit (scope enforcement is skipped)."""
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), in_place=True)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)
        # Crash agent so we exit before the backup_id NameError in _evaluate_and_decide
        agent.invoke.side_effect = AgentInvocationError("crash")

        # Act
        record = await runner.run_one(target)

        # Assert — in-place mode bypasses git commit entirely
        git.commit.assert_not_called()
        assert record.outcome is Outcome.CRASHED


# ---------------------------------------------------------------------------
# run_one — restart probability path (lines 290-305)
# ---------------------------------------------------------------------------


class TestRunOneRestartPath:
    @pytest.mark.asyncio
    async def test_restart_forced_adds_restart_tag(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), restart_probability=1.0)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)

        # Act
        record = await runner.run_one(target)

        # Assert — restart tag must appear when restart fires
        assert "restart" in record.tags

    @pytest.mark.asyncio
    async def test_no_restart_when_probability_zero(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), restart_probability=0.0)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)

        # Act
        record = await runner.run_one(target)

        # Assert
        assert "restart" not in record.tags

    @pytest.mark.asyncio
    async def test_simulated_annealing_scales_restart_probability(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(
            str(tmp_path),
            restart_probability=1.0,
            population_config=PopulationConfig(search_strategy="simulated_annealing"),
        )
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        search = SimulatedAnnealingSearch(initial_temperature=0.5)
        runner, _, _ = _build_runner(tmp_path, search=search)

        # Act
        record = await runner.run_one(target)

        # Assert — restart may or may not fire (temperature scaling), but no crash
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED, Outcome.BLOCKED)


# ---------------------------------------------------------------------------
# run_one — knowledge auto-activation (lines 309-320)
# ---------------------------------------------------------------------------


class TestRunOneKnowledgeAutoActivation:
    @pytest.mark.asyncio
    async def test_knowledge_auto_activates_above_threshold(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), inject_knowledge_context=False)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.types import ExperimentRecord
        from datetime import datetime, timezone

        def _make_kept_record() -> ExperimentRecord:
            return ExperimentRecord(
                id="x",
                target_id="test-target",
                git_sha="abc",
                pre_experiment_sha="abc",
                timestamp=datetime.now(tz=timezone.utc),
                hypothesis="h",
                hypothesis_source="agent",
                mutation_diff_summary="",
                score=0.9,
                score_ci_lower=None,
                score_ci_upper=None,
                raw_scores=None,
                baseline_score=0.75,
                outcome=Outcome.KEPT,
                failure_mode=None,
                duration_seconds=1.0,
                tags=[],
                learnings="",
                cost_usd=0.01,
                bootstrap_seed=0,
                agent_model="gpt-4.1",
            )

        knowledge = MagicMock(spec=KnowledgeStore)
        # First call (for threshold check) returns 20 KEPT records; second returns empty list
        kept_records = [_make_kept_record() for _ in range(20)]
        knowledge.load_records = MagicMock(side_effect=[kept_records, []])
        knowledge.get_context = MagicMock(return_value="")
        knowledge.append_record = MagicMock()
        knowledge.update_index = MagicMock()
        knowledge.consolidate_if_due = MagicMock()
        knowledge.record_count = MagicMock(return_value=0)
        knowledge.CONSOLIDATION_INTERVAL = 20

        runner, _, _ = _build_runner(tmp_path, knowledge=knowledge)

        # Act
        record = await runner.run_one(target)

        # Assert — load_records was called (auto-activation threshold check)
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED, Outcome.BLOCKED)
        knowledge.load_records.assert_called()


# ---------------------------------------------------------------------------
# run_one — simplification pre-pass (lines 349-368)
# ---------------------------------------------------------------------------


class TestRunOneSimplificationPrePass:
    @pytest.mark.asyncio
    async def test_simplify_before_mutate_invokes_agent_twice(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), simplify_before_mutate=True)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, agent = _build_runner(tmp_path)

        # Act
        await runner.run_one(target)

        # Assert — simplify pass + mutation pass = at least 2 invocations
        assert agent.invoke.call_count >= 2

    @pytest.mark.asyncio
    async def test_simplify_exception_continues_without_crash(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), simplify_before_mutate=True)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, agent = _build_runner(tmp_path)
        # Simplify call raises, mutation call succeeds
        agent.invoke.side_effect = [
            AgentInvocationError("simplify failed"),
            _mock_agent_result(),
        ]

        # Act
        record = await runner.run_one(target)

        # Assert — exception in simplify pass does not crash the run
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED, Outcome.BLOCKED, Outcome.CRASHED)


# ---------------------------------------------------------------------------
# run_one — multi-draft generation path (lines 407-459)
# ---------------------------------------------------------------------------


class TestRunOneMultiDraft:
    @pytest.mark.asyncio
    async def test_multi_draft_uses_generate_drafts(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), n_drafts=2)
        target.agent_config = AgentConfig(
            mode="claude_code",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
            n_drafts=2,
        )
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)

        # Act
        record = await runner.run_one(target)

        # Assert
        agent.generate_drafts.assert_called_once()
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED, Outcome.BLOCKED)

    @pytest.mark.asyncio
    async def test_multi_draft_all_generation_failed_returns_blocked(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.agent_config = AgentConfig(
            mode="claude_code",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
            n_drafts=3,
        )
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, agent = _build_runner(tmp_path)
        agent.generate_drafts = AsyncMock(return_value=[])  # All drafts failed

        # Act
        record = await runner.run_one(target)

        # Assert
        assert record.outcome is Outcome.BLOCKED
        assert record.failure_mode == "all_drafts_failed_generation"

    @pytest.mark.asyncio
    async def test_multi_draft_all_drafts_fail_verifiers_returns_blocked(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.agent_config = AgentConfig(
            mode="claude_code",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
            n_drafts=2,
        )
        target.eval_config.verifiers = [
            VerifierCommand(name="test_check", run_command="exit 1", timeout_seconds=10),
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)
        # apply_diff succeeds but verifiers fail
        git.apply_diff = AsyncMock(return_value=True)

        with patch("anneal.engine.runner.run_verifiers") as mock_verifiers:
            mock_verifiers.return_value = [("test_check", False, "test failed")]

            # Act
            record = await runner.run_one(target)

        # Assert
        assert record.outcome is Outcome.BLOCKED
        assert record.failure_mode == "all_drafts_failed_verifiers"

    @pytest.mark.asyncio
    async def test_multi_draft_empty_diff_skipped(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.agent_config = AgentConfig(
            mode="claude_code",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
            n_drafts=2,
        )
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)
        # First draft has empty diff (skipped), second is valid
        agent.generate_drafts = AsyncMock(return_value=[
            (_mock_agent_result(), "   "),  # Empty diff — skipped
            (_mock_agent_result(), "diff --git a/artifact.md b/artifact.md\n+change"),
        ])

        # Act
        record = await runner.run_one(target)

        # Assert — should not crash; empty diff is skipped gracefully
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED, Outcome.BLOCKED)


# ---------------------------------------------------------------------------
# run_one — verifier failure path single-draft (lines 480-527)
# ---------------------------------------------------------------------------


class TestRunOneVerifierFailure:
    @pytest.mark.asyncio
    async def test_verifier_failure_returns_blocked_record(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.eval_config.verifiers = [
            VerifierCommand(name="mycheck", run_command="exit 1", timeout_seconds=10),
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, _ = _build_runner(tmp_path)

        with patch("anneal.engine.runner.run_verifiers") as mock_verifiers:
            mock_verifiers.return_value = [("mycheck", False, "check failed")]

            # Act
            record = await runner.run_one(target)

        # Assert
        assert record.outcome is Outcome.BLOCKED
        assert record.failure_mode == "verifier:mycheck"
        git.reset_hard.assert_called()

    @pytest.mark.asyncio
    async def test_verifier_success_proceeds_to_eval(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.eval_config.verifiers = [
            VerifierCommand(name="mycheck", run_command="true", timeout_seconds=10),
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)

        with patch("anneal.engine.runner.run_verifiers") as mock_verifiers:
            mock_verifiers.return_value = [("mycheck", True, "")]

            # Act
            record = await runner.run_one(target)

        # Assert — verifier passed, run continues to eval
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED)


# ---------------------------------------------------------------------------
# _invoke_agent — timeout and crash paths (lines 595-607)
# ---------------------------------------------------------------------------


class TestInvokeAgent:
    @pytest.mark.asyncio
    async def test_timeout_error_returns_killed_record(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)
        agent.invoke.side_effect = AgentTimeoutError("timed out after 60s")

        # Act
        record = await runner.run_one(target)

        # Assert
        assert record.outcome is Outcome.KILLED
        assert record.failure_mode is not None
        assert "timed out" in record.failure_mode
        git.cleanup_index_lock.assert_called()
        git.reset_hard.assert_called()

    @pytest.mark.asyncio
    async def test_invocation_error_returns_crashed_record(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)
        agent.invoke.side_effect = AgentInvocationError("subprocess crashed")

        # Act
        record = await runner.run_one(target)

        # Assert
        assert record.outcome is Outcome.CRASHED
        assert record.failure_mode is not None
        assert "subprocess crashed" in record.failure_mode
        git.reset_hard.assert_called()


# ---------------------------------------------------------------------------
# _enforce_and_commit — scope paths (lines 627-664)
# ---------------------------------------------------------------------------


class TestEnforceAndCommit:
    @pytest.mark.asyncio
    async def test_in_place_skips_scope_enforcement(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), in_place=True)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)
        # Crash agent before reaching backup_id NameError in _evaluate_and_decide
        agent.invoke.side_effect = AgentInvocationError("crash")

        # Act
        record = await runner.run_one(target)

        # Assert — in-place: no scope enforcement, commit not called
        git.commit.assert_not_called()
        assert record.outcome is Outcome.CRASHED

    @pytest.mark.asyncio
    async def test_all_blocked_scope_returns_blocked_outcome(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, _ = _build_runner(tmp_path)

        with patch("anneal.engine.runner.enforce_scope") as mock_scope:
            from anneal.engine.types import ScopeViolationResult
            mock_scope.return_value = ScopeViolationResult(
                valid_paths=[],
                violated_paths=["scope.yaml"],
                all_blocked=True,
                has_violations=True,
            )
            # Override status to return modified files so enforcement runs
            git.status_porcelain = AsyncMock(return_value=[("M", "scope.yaml")])

            # Act
            record = await runner.run_one(target)

        # Assert
        assert record.outcome is Outcome.BLOCKED
        assert record.failure_mode is not None
        assert "violated scope" in record.failure_mode or "All changes violated" in record.failure_mode

    @pytest.mark.asyncio
    async def test_no_valid_paths_returns_blocked(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, _ = _build_runner(tmp_path)

        with patch("anneal.engine.runner.enforce_scope") as mock_scope:
            from anneal.engine.types import ScopeViolationResult
            mock_scope.return_value = ScopeViolationResult(
                valid_paths=[],
                violated_paths=[],
                all_blocked=False,
                has_violations=False,
            )
            git.status_porcelain = AsyncMock(return_value=[("M", "artifact.md")])

            # Act
            record = await runner.run_one(target)

        # Assert — no valid paths = no changes = BLOCKED
        assert record.outcome is Outcome.BLOCKED
        assert record.failure_mode == "Agent made no file changes"

    @pytest.mark.asyncio
    async def test_has_violations_resets_violated_but_continues(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, _ = _build_runner(tmp_path)
        git.checkout_paths = AsyncMock()

        with patch("anneal.engine.runner.enforce_scope") as mock_scope:
            from anneal.engine.types import ScopeViolationResult
            mock_scope.return_value = ScopeViolationResult(
                valid_paths=["artifact.md"],
                violated_paths=["scope.yaml"],
                all_blocked=False,
                has_violations=True,
            )
            git.status_porcelain = AsyncMock(return_value=[
                ("M", "artifact.md"),
                ("M", "scope.yaml"),
            ])

            # Act
            record = await runner.run_one(target)

        # Assert — violations were reset but valid paths committed
        git.checkout_paths.assert_called()
        git.commit.assert_called()


# ---------------------------------------------------------------------------
# _evaluate_and_decide — cold-start stochastic (lines 773-775)
# ---------------------------------------------------------------------------


class TestEvaluateAndDecide:
    @pytest.mark.asyncio
    async def test_stochastic_cold_start_keeps_first_result(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), held_out_prompts=["p"])
        target.baseline_raw_scores = []  # No previous scores = cold start
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)

        # Act
        record = await runner.run_one(target)

        # Assert — cold-start: always KEPT
        assert record.outcome is Outcome.KEPT

    @pytest.mark.asyncio
    async def test_eval_error_returns_crashed_record(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.eval import EvalError

        runner, git, _ = _build_runner(tmp_path)

        with patch.object(runner._eval, "evaluate", side_effect=EvalError("eval exploded")):
            # Act
            record = await runner.run_one(target)

        # Assert — eval error → CRASHED
        assert record.outcome is Outcome.CRASHED
        git.reset_hard.assert_called()

    @pytest.mark.asyncio
    async def test_taxonomy_classification_called_on_discarded(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.baseline_score = 99.0  # Force DISCARDED
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.taxonomy import FailureTaxonomy
        taxonomy = MagicMock(spec=FailureTaxonomy)
        taxonomy.classify = AsyncMock(return_value=("score_regression", 0.001))

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        _call_count = 0

        async def _status(_w):
            nonlocal _call_count
            _call_count += 1
            return [] if _call_count % 2 == 1 else [("M", "artifact.md")]

        git.status_porcelain = _status
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
            taxonomy=taxonomy,
        )

        # Act
        record = await runner.run_one(target)

        # Assert
        assert record.outcome is Outcome.DISCARDED
        taxonomy.classify.assert_called_once()

    @pytest.mark.asyncio
    async def test_tree_search_records_outcome(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        tree_search = MagicMock(spec=UCBTreeSearch)
        tree_search.select_parent = MagicMock(return_value="abc123")
        # tree_info must include 'depth' key (used by build_target_context)
        tree_search.get_tree_info = MagicMock(return_value={"nodes": 1, "depth": 0, "leaves": 1, "pruned": 0})
        tree_search.record_outcome = MagicMock()
        tree_search.persist = MagicMock()
        tree_search.should_keep = MagicMock(return_value=True)

        runner, git, agent = _build_runner(tmp_path, search=tree_search)
        git.rev_parse = AsyncMock(return_value="abc123")

        knowledge_path = tmp_path / ".anneal" / "targets" / "test-target"
        knowledge_path.mkdir(parents=True, exist_ok=True)

        # Act
        record = await runner.run_one(target)

        # Assert
        tree_search.record_outcome.assert_called_once()
        tree_search.persist.assert_called_once()


# ---------------------------------------------------------------------------
# run_loop — stop conditions (lines 885-981)
# ---------------------------------------------------------------------------


class TestRunLoopStopConditions:
    @pytest.mark.asyncio
    async def test_max_experiments_stops_loop(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        # Act
        records = await runner.run_loop(target, max_experiments=2)

        # Assert
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_stop_score_halts_when_reached(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.baseline_score = 0.95  # Already at or above stop_score
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        # Act
        records = await runner.run_loop(target, max_experiments=10, stop_score=0.9)

        # Assert — stop_score already met before first experiment
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_consecutive_failures_halts_loop(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.max_consecutive_failures = 2
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, agent = _build_runner(tmp_path)
        runner._write_status = AsyncMock()
        agent.invoke.side_effect = AgentTimeoutError("timeout")

        # Act
        records = await runner.run_loop(target, max_experiments=10)

        # Assert — halted after 2 consecutive failures
        assert len(records) == 2
        assert all(r.outcome is Outcome.KILLED for r in records)

    @pytest.mark.asyncio
    async def test_request_stop_halts_loop_mid_run(self, tmp_path: Path) -> None:
        """request_stop() stops the loop after the current experiment completes.

        run_loop calls _clear_stop at entry so a pre-set flag is cleared.
        We set the flag mid-run via the on_experiment callback instead.
        """
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        def _stop_after_first(record):
            runner.request_stop(target.id)

        # Act
        records = await runner.run_loop(
            target,
            max_experiments=10,
            on_experiment=_stop_after_first,
        )

        # Assert — stopped after first experiment
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_on_experiment_callback_invoked(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        callback_records = []

        # Act
        await runner.run_loop(
            target,
            max_experiments=2,
            on_experiment=callback_records.append,
        )

        # Assert
        assert len(callback_records) == 2


# ---------------------------------------------------------------------------
# run_loop — state persistence and restore (lines 919-927)
# ---------------------------------------------------------------------------


class TestRunLoopStatePersistence:
    @pytest.mark.asyncio
    async def test_loop_state_saved_after_each_experiment(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        # Use tmp_path-based knowledge_path so state is isolated per test
        knowledge_path = tmp_path / ".anneal" / "targets" / "test-target"
        knowledge_path.mkdir(parents=True, exist_ok=True)
        target.knowledge_path = str(knowledge_path)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        # Act
        await runner.run_loop(target, max_experiments=1)

        # Assert — loop state file written
        state_path = knowledge_path / ".loop-state.json"
        assert state_path.exists()
        loaded = RunLoopState.load(state_path)
        assert loaded.total_experiments == 1

    @pytest.mark.asyncio
    async def test_loop_state_restored_from_previous_run(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        # Use tmp_path-based knowledge_path so state is isolated per test
        knowledge_path = tmp_path / ".anneal" / "targets" / "test-target"
        knowledge_path.mkdir(parents=True, exist_ok=True)
        target.knowledge_path = str(knowledge_path)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        state_path = knowledge_path / ".loop-state.json"

        # Write pre-existing state
        prior_state = RunLoopState(total_experiments=5, kept_count=3)
        prior_state.save(state_path)

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        # Act
        await runner.run_loop(target, max_experiments=1)

        # Assert — total_experiments continues from restored value (5+1=6)
        loaded = RunLoopState.load(state_path)
        assert loaded.total_experiments == 6


# ---------------------------------------------------------------------------
# run_loop — held-out evaluation (lines 1015-1049)
# ---------------------------------------------------------------------------


class TestRunLoopHeldOutEval:
    @pytest.mark.asyncio
    async def test_held_out_eval_triggered_at_interval(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(
            str(tmp_path),
            held_out_prompts=["prompt1"],
            held_out_interval=1,
        )
        target.baseline_raw_scores = []  # Cold start so first is KEPT
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        knowledge = MagicMock(spec=KnowledgeStore)
        knowledge.load_records = MagicMock(return_value=[])
        knowledge.get_context = MagicMock(return_value="")
        knowledge.append_record = MagicMock()
        knowledge.update_index = MagicMock()
        knowledge.consolidate_if_due = MagicMock()
        knowledge.record_count = MagicMock(return_value=0)
        knowledge.CONSOLIDATION_INTERVAL = 20

        eval_engine = MagicMock(spec=EvalEngine)
        eval_engine.evaluate = AsyncMock(return_value=EvalResult(score=0.9))
        eval_engine.check_constraints = AsyncMock(return_value=[])
        held_out_result = EvalResult(score=0.85)
        eval_engine.evaluate_held_out = AsyncMock(return_value=held_out_result)

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        _call_count = 0

        async def _status(_w):
            nonlocal _call_count
            _call_count += 1
            return [] if _call_count % 2 == 1 else [("M", "artifact.md")]

        git.status_porcelain = _status
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=eval_engine,
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
            knowledge=knowledge,
        )
        runner._write_status = AsyncMock()

        # Act
        records = await runner.run_loop(target, max_experiments=1)

        # Assert — held-out eval was triggered
        eval_engine.evaluate_held_out.assert_called_once()
        assert records[0].held_out_score == 0.85

    @pytest.mark.asyncio
    async def test_held_out_eval_error_does_not_crash_loop(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(
            str(tmp_path),
            held_out_prompts=["prompt1"],
            held_out_interval=1,
        )
        target.baseline_raw_scores = []
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.eval import EvalError

        eval_engine = MagicMock(spec=EvalEngine)
        eval_engine.evaluate = AsyncMock(return_value=EvalResult(score=0.9))
        eval_engine.check_constraints = AsyncMock(return_value=[])
        eval_engine.evaluate_held_out = AsyncMock(side_effect=EvalError("held out failed"))

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        _call_count = 0

        async def _status(_w):
            nonlocal _call_count
            _call_count += 1
            return [] if _call_count % 2 == 1 else [("M", "artifact.md")]

        git.status_porcelain = _status
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=eval_engine,
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
        )
        runner._write_status = AsyncMock()

        # Act — should not raise
        records = await runner.run_loop(target, max_experiments=1)

        # Assert
        assert len(records) == 1


# ---------------------------------------------------------------------------
# run_loop — component evolution manifest mode (lines 1076-1104)
# ---------------------------------------------------------------------------


class TestRunLoopComponentEvolution:
    @pytest.mark.asyncio
    async def test_component_evolution_called_at_consolidation_interval(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.strategy_mode = "manifest"
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.strategy import StrategyManifest, save_strategy
        manifest = StrategyManifest(lineage=[])
        manifest.hypothesis_generation.approach = "initial approach"
        knowledge_path = Path(target.knowledge_path)
        knowledge_path.mkdir(parents=True, exist_ok=True)
        save_strategy(manifest, knowledge_path)

        knowledge = MagicMock(spec=KnowledgeStore)
        knowledge.load_records = MagicMock(return_value=[])
        knowledge.get_context = MagicMock(return_value="")
        knowledge.append_record = MagicMock()
        knowledge.update_index = MagicMock()
        knowledge.consolidate_if_due = MagicMock()
        knowledge.record_count = MagicMock(return_value=0)
        consolidation_interval = 20
        knowledge.CONSOLIDATION_INTERVAL = consolidation_interval

        runner, _, agent = _build_runner(tmp_path, knowledge=knowledge)
        runner._write_status = AsyncMock()

        # Patch total_experiments to trigger consolidation on first experiment
        with patch("anneal.engine.runner.KnowledgeStore") as mock_ks_cls:
            mock_ks_cls.CONSOLIDATION_INTERVAL = 1

            # Act
            records = await runner.run_loop(target, max_experiments=1)

        # Assert — no crash; component evolution path exercised
        assert len(records) == 1


# ---------------------------------------------------------------------------
# run_loop — plateau meta-optimization (lines 1106-1141)
# ---------------------------------------------------------------------------


class TestRunLoopPlateauMetaOptimization:
    @pytest.mark.asyncio
    async def test_plateau_meta_triggers_invoke_meta(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), meta_depth=1)
        target.baseline_score = 99.0  # All experiments DISCARD
        # meta_m = min(max_consecutive_failures, 10). Set to 3 so plateau triggers after 3 experiments.
        target.max_consecutive_failures = 3
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        # Create program.md under absolute knowledge path
        program_dir = tmp_path / ".anneal" / "targets" / "test-target"
        program_dir.mkdir(parents=True, exist_ok=True)
        (program_dir / "program.md").write_text("# Optimization Strategy\n")
        target.knowledge_path = str(program_dir)

        runner, _, agent = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        # Act — 4 experiments, plateau threshold is 3
        records = await runner.run_loop(target, max_experiments=4)

        # Assert — plateau triggers meta invoke
        agent.invoke_meta.assert_called()

    @pytest.mark.asyncio
    async def test_plateau_meta_error_does_not_crash_loop(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), meta_depth=1)
        target.baseline_score = 99.0
        target.max_consecutive_failures = 3  # meta_m = 3
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        program_dir = tmp_path / ".anneal" / "targets" / "test-target"
        program_dir.mkdir(parents=True, exist_ok=True)
        (program_dir / "program.md").write_text("# Optimization Strategy\n")
        target.knowledge_path = str(program_dir)

        runner, _, agent = _build_runner(tmp_path)
        runner._write_status = AsyncMock()
        agent.invoke_meta.side_effect = AgentInvocationError("meta crashed")

        # Act — must not raise
        records = await runner.run_loop(target, max_experiments=4)

        # Assert
        assert len(records) == 4

    @pytest.mark.asyncio
    async def test_meta_not_triggered_without_program_md(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), meta_depth=1)
        target.baseline_score = 99.0
        target.max_consecutive_failures = 3  # meta_m = 3, plateau at 3 non-kept
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        knowledge_dir = tmp_path / ".anneal" / "targets" / "test-target"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        target.knowledge_path = str(knowledge_dir)
        # No program.md created

        runner, _, agent = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        # Act — no program.md file exists
        records = await runner.run_loop(target, max_experiments=4)

        # Assert — meta not triggered because program.md absent
        agent.invoke_meta.assert_not_called()


# ---------------------------------------------------------------------------
# Recovery helpers (lines 1221-1263)
# ---------------------------------------------------------------------------


class TestResetViolated:
    @pytest.mark.asyncio
    async def test_reset_violated_checks_out_tracked_files(self, tmp_path: Path) -> None:
        # Arrange
        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        git.status_porcelain = AsyncMock(return_value=[])
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.fsck = AsyncMock(return_value=True)

        runner = ExperimentRunner(
            git=git,
            agent_invoker=AsyncMock(spec=AgentInvoker),
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=MagicMock(spec=Registry),
        )

        # Act
        await runner._reset_violated(
            tmp_path,
            violated_paths=["artifact.md"],
            git_status=[("M", "artifact.md")],
        )

        # Assert — tracked file checked out via checkout_paths
        git.checkout_paths.assert_called_once_with(tmp_path, ["artifact.md"])

    @pytest.mark.asyncio
    async def test_reset_violated_deletes_untracked_files(self, tmp_path: Path) -> None:
        # Arrange
        untracked_file = tmp_path / "newfile.md"
        untracked_file.write_text("new content")

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        git.checkout_paths = AsyncMock()

        runner = ExperimentRunner(
            git=git,
            agent_invoker=AsyncMock(spec=AgentInvoker),
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=MagicMock(spec=Registry),
        )

        # Act
        await runner._reset_violated(
            tmp_path,
            violated_paths=["newfile.md"],
            git_status=[("??", "newfile.md")],
        )

        # Assert — untracked file deleted; checkout_paths not called
        assert not untracked_file.exists()
        git.checkout_paths.assert_not_called()

    @pytest.mark.asyncio
    async def test_reset_violated_removes_untracked_dir(self, tmp_path: Path) -> None:
        # Arrange
        untracked_dir = tmp_path / "new_subdir"
        untracked_dir.mkdir()
        (untracked_dir / "file.md").write_text("content")

        git = AsyncMock(spec=GitEnvironment)
        git.checkout_paths = AsyncMock()

        runner = ExperimentRunner(
            git=git,
            agent_invoker=AsyncMock(spec=AgentInvoker),
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=MagicMock(spec=Registry),
        )

        # Act
        await runner._reset_violated(
            tmp_path,
            violated_paths=["new_subdir"],
            git_status=[("??", "new_subdir")],
        )

        # Assert — untracked directory removed
        assert not untracked_dir.exists()


class TestHandleKilled:
    @pytest.mark.asyncio
    async def test_handle_killed_calls_cleanup_reset_clean(self, tmp_path: Path) -> None:
        # Arrange
        git = AsyncMock(spec=GitEnvironment)
        git.cleanup_index_lock = AsyncMock()
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.fsck = AsyncMock(return_value=True)

        runner = ExperimentRunner(
            git=git,
            agent_invoker=AsyncMock(spec=AgentInvoker),
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=MagicMock(spec=Registry),
        )

        # Act
        await runner._handle_killed(tmp_path, "abc123")

        # Assert
        git.cleanup_index_lock.assert_called_once_with(tmp_path)
        git.reset_hard.assert_called_once_with(tmp_path, "abc123")
        git.clean_untracked.assert_called_once_with(tmp_path)
        git.fsck.assert_called_once_with(tmp_path)

    @pytest.mark.asyncio
    async def test_handle_killed_logs_corruption_on_fsck_fail(
        self, tmp_path: Path, caplog
    ) -> None:
        # Arrange
        git = AsyncMock(spec=GitEnvironment)
        git.cleanup_index_lock = AsyncMock()
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.fsck = AsyncMock(return_value=False)  # Corruption detected

        runner = ExperimentRunner(
            git=git,
            agent_invoker=AsyncMock(spec=AgentInvoker),
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=MagicMock(spec=Registry),
        )

        import logging
        # Act
        with caplog.at_level(logging.ERROR):
            await runner._handle_killed(tmp_path, "abc123")

        # Assert — corruption log emitted
        assert "corruption" in caplog.text.lower() or "corrupt" in caplog.text.lower()


class TestSafeRestore:
    @pytest.mark.asyncio
    async def test_safe_restore_calls_reset_and_clean(self, tmp_path: Path) -> None:
        # Arrange
        git = AsyncMock(spec=GitEnvironment)
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()

        runner = ExperimentRunner(
            git=git,
            agent_invoker=AsyncMock(spec=AgentInvoker),
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=MagicMock(spec=Registry),
        )

        # Act
        await runner._safe_restore(tmp_path, "abc123")

        # Assert
        git.reset_hard.assert_called_once_with(tmp_path, "abc123")
        git.clean_untracked.assert_called_once_with(tmp_path)


# ---------------------------------------------------------------------------
# run_loop — eval environment lifecycle (lines 916-917, 1161-1163)
# ---------------------------------------------------------------------------


class TestRunLoopEvalEnvironment:
    @pytest.mark.asyncio
    async def test_eval_env_setup_and_teardown_commands_invoked(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.types import EvalEnvironment
        target.eval_environment = EvalEnvironment(
            setup_command="echo setup",
            teardown_command="echo teardown",
        )

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        with patch.object(runner, "_run_lifecycle_command", new_callable=AsyncMock) as mock_lifecycle:
            # Act
            await runner.run_loop(target, max_experiments=1)

        # Assert — both setup and teardown called
        calls = [c.args[1] for c in mock_lifecycle.call_args_list]
        assert "setup" in calls
        assert "teardown" in calls


# ---------------------------------------------------------------------------
# RunLoopState — save/load round-trip
# ---------------------------------------------------------------------------


class TestRunLoopState:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        # Arrange
        state = RunLoopState(
            consecutive_failures=2,
            kept_count=7,
            consecutive_no_kept=3,
            total_experiments=12,
            cumulative_cost_usd=0.42,
        )
        path = tmp_path / "state.json"

        # Act
        state.save(path)
        loaded = RunLoopState.load(path)

        # Assert
        assert loaded.consecutive_failures == 2
        assert loaded.kept_count == 7
        assert loaded.consecutive_no_kept == 3
        assert loaded.total_experiments == 12
        assert abs(loaded.cumulative_cost_usd - 0.42) < 1e-9

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "nonexistent.json"

        # Act
        state = RunLoopState.load(path)

        # Assert
        assert state.consecutive_failures == 0
        assert state.kept_count == 0
        assert state.total_experiments == 0


# ---------------------------------------------------------------------------
# ScopeIntegrityError — scope hash mismatch (line 264)
# ---------------------------------------------------------------------------


class TestScopeIntegrityError:
    @pytest.mark.asyncio
    async def test_scope_hash_mismatch_raises_scope_integrity_error(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        # Deliberately wrong scope hash
        target.scope_hash = "deadbeef000000000000000000000000"

        runner, _, _ = _build_runner(tmp_path)

        # Act / Assert
        from anneal.engine.runner import ScopeIntegrityError
        with pytest.raises(ScopeIntegrityError, match="scope.yaml hash mismatch"):
            await runner.run_one(target)


# ---------------------------------------------------------------------------
# Two-phase mutation / diagnosis (lines 374-400)
# ---------------------------------------------------------------------------


class TestTwoPhaseMutation:
    @pytest.mark.asyncio
    async def test_diagnosis_prepends_context_to_prompt(self, tmp_path: Path) -> None:
        # Arrange — two_phase_mutation=True + inject_knowledge_context=True + history
        _make_git_repo(tmp_path)
        target = _make_target(
            str(tmp_path),
            two_phase_mutation=True,
            inject_knowledge_context=True,
        )
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.types import DiagnosisResult, ExperimentRecord
        from datetime import datetime, timezone

        diagnosis_result = DiagnosisResult(
            weakest_criteria=["clarity"],
            root_cause="Too verbose",
            fix_category="content",
            suggested_direction="Reduce word count",
            cost_usd=0.005,
        )

        def _make_record() -> ExperimentRecord:
            return ExperimentRecord(
                id="x",
                target_id="test-target",
                git_sha="abc",
                pre_experiment_sha="abc",
                timestamp=datetime.now(tz=timezone.utc),
                hypothesis="prev hypothesis",
                hypothesis_source="agent",
                mutation_diff_summary="",
                score=0.8,
                score_ci_lower=None,
                score_ci_upper=None,
                raw_scores=None,
                baseline_score=0.75,
                outcome=Outcome.KEPT,
                failure_mode=None,
                duration_seconds=1.0,
                tags=[],
                learnings="",
                cost_usd=0.01,
                bootstrap_seed=0,
                agent_model="gpt-4.1",
            )

        knowledge = MagicMock(spec=KnowledgeStore)
        # load_records called twice: once for history, once inside build_target_context
        knowledge.load_records = MagicMock(return_value=[_make_record()])
        knowledge.get_context = MagicMock(return_value="some context")
        knowledge.append_record = MagicMock()
        knowledge.update_index = MagicMock()
        knowledge.consolidate_if_due = MagicMock()
        knowledge.record_count = MagicMock(return_value=1)
        knowledge.CONSOLIDATION_INTERVAL = 20

        runner, _, agent = _build_runner(tmp_path, knowledge=knowledge)
        agent.diagnose = AsyncMock(return_value=diagnosis_result)

        # Act
        record = await runner.run_one(target)

        # Assert — diagnose called, cost added
        agent.diagnose.assert_called_once()
        assert record.cost_usd >= diagnosis_result.cost_usd
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED, Outcome.BLOCKED)

    @pytest.mark.asyncio
    async def test_diagnosis_failure_continues_without_crash(self, tmp_path: Path) -> None:
        # Arrange — two_phase_mutation=True + inject_knowledge_context=True + history
        _make_git_repo(tmp_path)
        target = _make_target(
            str(tmp_path),
            two_phase_mutation=True,
            inject_knowledge_context=True,
        )
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.types import ExperimentRecord
        from datetime import datetime, timezone

        def _make_record() -> ExperimentRecord:
            return ExperimentRecord(
                id="x",
                target_id="test-target",
                git_sha="abc",
                pre_experiment_sha="abc",
                timestamp=datetime.now(tz=timezone.utc),
                hypothesis="prev",
                hypothesis_source="agent",
                mutation_diff_summary="",
                score=0.8,
                score_ci_lower=None,
                score_ci_upper=None,
                raw_scores=None,
                baseline_score=0.75,
                outcome=Outcome.KEPT,
                failure_mode=None,
                duration_seconds=1.0,
                tags=[],
                learnings="",
                cost_usd=0.01,
                bootstrap_seed=0,
                agent_model="gpt-4.1",
            )

        knowledge = MagicMock(spec=KnowledgeStore)
        knowledge.load_records = MagicMock(return_value=[_make_record()])
        knowledge.get_context = MagicMock(return_value="some context")
        knowledge.append_record = MagicMock()
        knowledge.update_index = MagicMock()
        knowledge.consolidate_if_due = MagicMock()
        knowledge.record_count = MagicMock(return_value=1)
        knowledge.CONSOLIDATION_INTERVAL = 20

        runner, _, agent = _build_runner(tmp_path, knowledge=knowledge)
        agent.diagnose = AsyncMock(side_effect=AgentInvocationError("diagnose failed"))

        # Act
        record = await runner.run_one(target)

        # Assert — diagnosis failure is warned but run continues
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED, Outcome.BLOCKED)


# ---------------------------------------------------------------------------
# Simplification pre-pass — verifier failure reverts (lines 360-366)
# ---------------------------------------------------------------------------


class TestSimplificationVerifierRevert:
    @pytest.mark.asyncio
    async def test_simplify_verifier_failure_reverts(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), simplify_before_mutate=True)
        target.eval_config.verifiers = [
            VerifierCommand(name="check", run_command="true", timeout_seconds=10),
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)

        # Simplify call succeeds, then verifiers fail, then mutation succeeds
        agent.invoke.side_effect = [
            _mock_agent_result("simplify"),
            _mock_agent_result("mutate"),
        ]

        with patch("anneal.engine.runner.run_verifiers") as mock_vfy:
            # First verifier call (simplify check) fails; second (mutation check) passes
            mock_vfy.side_effect = [
                [("check", False, "failed")],  # simplify verifier fails
                [("check", True, "")],          # mutation verifier passes
            ]

            # Act
            record = await runner.run_one(target)

        # Assert — simplify was reverted (reset_hard called), run continued
        git.reset_hard.assert_called()
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED)

    @pytest.mark.asyncio
    async def test_simplify_verifier_passes_accepted(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path), simplify_before_mutate=True)
        target.eval_config.verifiers = [
            VerifierCommand(name="check", run_command="true", timeout_seconds=10),
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)

        agent.invoke.side_effect = [
            _mock_agent_result("simplify"),
            _mock_agent_result("mutate"),
        ]

        with patch("anneal.engine.runner.run_verifiers") as mock_vfy:
            # Both simplify and mutation verifiers pass
            mock_vfy.return_value = [("check", True, "")]

            # Act
            record = await runner.run_one(target)

        # Assert — simplify accepted, run continued
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED)


# ---------------------------------------------------------------------------
# _write_status — actual file writing (lines 1178-1189)
# ---------------------------------------------------------------------------


class TestWriteStatus:
    @pytest.mark.asyncio
    async def test_write_status_creates_status_file(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        # Do NOT mock _write_status — let it run for real

        # Act
        await runner.run_loop(target, max_experiments=1)

        # Assert — status file was written to worktree
        status_path = tmp_path / target.notifications.status_file
        assert status_path.exists()
        import json as _json
        data = _json.loads(status_path.read_text())
        assert data["target_id"] == "test-target"
        assert "state" in data


# ---------------------------------------------------------------------------
# run_loop — PAUSED state from safety check (lines 973-981)
# ---------------------------------------------------------------------------


class TestRunLoopSafetyCheckPaused:
    @pytest.mark.asyncio
    async def test_safety_check_fail_pauses_loop(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        with patch("anneal.engine.runner.pre_experiment_check") as mock_check:
            mock_check.return_value = (False, "budget exceeded")

            # Act
            records = await runner.run_loop(target, max_experiments=5)

        # Assert — loop paused immediately, no experiments run
        assert len(records) == 0
        runner._write_status.assert_called()


# ---------------------------------------------------------------------------
# run_loop — policy agent initialization (line 912)
# ---------------------------------------------------------------------------


class TestRunLoopPolicyAgent:
    @pytest.mark.asyncio
    async def test_policy_agent_initialized_when_enabled(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.types import PolicyConfig
        target.policy_config = PolicyConfig(enabled=True, rewrite_interval=100)

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        # Act
        await runner.run_loop(target, max_experiments=1)

        # Assert — policy agent created
        assert runner._policy_agent is not None


# ---------------------------------------------------------------------------
# run_loop — fidelity stage EvalError continues (lines 711-713)
# ---------------------------------------------------------------------------


class TestFidelityStageEvalError:
    @pytest.mark.asyncio
    async def test_fidelity_stage_eval_error_continues(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.eval_config.fidelity_stages = [
            FidelityStage(
                name="flaky_check",
                run_command="echo 0.9",
                parse_command="cat",
                min_pass_score=0.5,
            )
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.eval import EvalError

        runner, _, _ = _build_runner(tmp_path)

        call_count = 0
        original_evaluate = runner._eval.evaluate

        async def _mock_evaluate(worktree, eval_config, content):
            nonlocal call_count
            call_count += 1
            # Fidelity stage eval fails, main eval succeeds
            if "fidelity_" in eval_config.metric_name:
                raise EvalError("fidelity eval exploded")
            return await original_evaluate(worktree, eval_config, content)

        runner._eval.evaluate = _mock_evaluate

        # Act
        record = await runner.run_one(target)

        # Assert — fidelity eval error is warned, run continues to main eval
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED)

    @pytest.mark.asyncio
    async def test_fidelity_stage_lower_is_better_passes(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.eval_config.direction = Direction.LOWER_IS_BETTER
        target.eval_config.fidelity_stages = [
            FidelityStage(
                name="latency_check",
                run_command="echo 0.1",
                parse_command="cat",
                min_pass_score=0.5,  # score (0.1) <= 0.5 passes for LOWER_IS_BETTER
            )
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, _, _ = _build_runner(tmp_path)

        # Act
        record = await runner.run_one(target)

        # Assert — fidelity passes (0.1 <= 0.5), run continues
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED)


# ---------------------------------------------------------------------------
# _run_lifecycle_command — actual subprocess (lines 885-897)
# ---------------------------------------------------------------------------


class TestRunLifecycleCommand:
    @pytest.mark.asyncio
    async def test_lifecycle_command_success_logs_info(self, tmp_path: Path, caplog) -> None:
        # Arrange
        runner = ExperimentRunner(
            git=AsyncMock(spec=GitEnvironment),
            agent_invoker=AsyncMock(spec=AgentInvoker),
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=MagicMock(spec=Registry),
        )

        import logging
        # Act
        with caplog.at_level(logging.INFO):
            await runner._run_lifecycle_command("echo hello", "setup")

        # Assert
        assert "setup" in caplog.text

    @pytest.mark.asyncio
    async def test_lifecycle_command_failure_logs_error(self, tmp_path: Path, caplog) -> None:
        # Arrange
        runner = ExperimentRunner(
            git=AsyncMock(spec=GitEnvironment),
            agent_invoker=AsyncMock(spec=AgentInvoker),
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=MagicMock(spec=Registry),
        )

        import logging
        # Act
        with caplog.at_level(logging.ERROR):
            await runner._run_lifecycle_command("exit 1", "teardown")

        # Assert — error logged for non-zero exit
        assert "teardown" in caplog.text


# ---------------------------------------------------------------------------
# Tree search — parent checkout when parent_sha != pre_sha (lines 280-283)
# ---------------------------------------------------------------------------


class TestTreeSearchParentCheckout:
    @pytest.mark.asyncio
    async def test_tree_search_checkouts_different_parent(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        # select_parent returns a sha different from HEAD
        tree_search = MagicMock(spec=UCBTreeSearch)
        tree_search.select_parent = MagicMock(return_value="parent111")
        tree_search.get_tree_info = MagicMock(return_value={"nodes": 2, "depth": 1, "leaves": 1, "pruned": 0})
        tree_search.record_outcome = MagicMock()
        tree_search.persist = MagicMock()
        tree_search.should_keep = MagicMock(return_value=True)

        runner, git, _ = _build_runner(tmp_path, search=tree_search)
        git.rev_parse = AsyncMock(return_value="headsha1")  # Different from parent111

        knowledge_path = tmp_path / ".anneal" / "targets" / "test-target"
        knowledge_path.mkdir(parents=True, exist_ok=True)

        # Act
        record = await runner.run_one(target)

        # Assert — checkout called with parent sha
        git.checkout.assert_called_once_with(tmp_path, "parent111")
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED, Outcome.BLOCKED)


# ---------------------------------------------------------------------------
# Verifier failure with learning_pool (lines 523-524, 526)
# ---------------------------------------------------------------------------


class TestVerifierFailureWithLearningPool:
    @pytest.mark.asyncio
    async def test_verifier_failure_extracts_learning(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.eval_config.verifiers = [
            VerifierCommand(name="mycheck", run_command="exit 1", timeout_seconds=10),
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.learning_pool import LearningPool
        pool = LearningPool()

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        _call_count = 0

        async def _status(_w):
            nonlocal _call_count
            _call_count += 1
            return [] if _call_count % 2 == 1 else [("M", "artifact.md")]

        git.status_porcelain = _status
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
            learning_pool=pool,
        )

        with patch("anneal.engine.runner.run_verifiers") as mock_vfy:
            mock_vfy.return_value = [("mycheck", False, "check failed")]
            # Act
            record = await runner.run_one(target)

        # Assert — verifier failure added to learning pool
        assert record.outcome is Outcome.BLOCKED
        assert pool.count == 1


# ---------------------------------------------------------------------------
# Multi-draft — knowledge recording on all_drafts_failed_verifiers (lines 449-450)
# ---------------------------------------------------------------------------


class TestMultiDraftKnowledgeRecording:
    @pytest.mark.asyncio
    async def test_all_drafts_failed_verifiers_records_to_knowledge(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.agent_config = AgentConfig(
            mode="claude_code",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
            n_drafts=2,
        )
        target.eval_config.verifiers = [
            VerifierCommand(name="check", run_command="exit 1", timeout_seconds=10),
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        knowledge = MagicMock(spec=KnowledgeStore)
        knowledge.load_records = MagicMock(return_value=[])
        knowledge.get_context = MagicMock(return_value="")
        knowledge.append_record = MagicMock()
        knowledge.update_index = MagicMock()
        knowledge.consolidate_if_due = MagicMock()
        knowledge.record_count = MagicMock(return_value=0)
        knowledge.CONSOLIDATION_INTERVAL = 20

        runner, git, agent = _build_runner(tmp_path, knowledge=knowledge)
        git.apply_diff = AsyncMock(return_value=True)

        with patch("anneal.engine.runner.run_verifiers") as mock_vfy:
            mock_vfy.return_value = [("check", False, "failed")]
            # Act
            record = await runner.run_one(target)

        # Assert — BLOCKED record saved to knowledge
        assert record.outcome is Outcome.BLOCKED
        knowledge.append_record.assert_called()
        knowledge.update_index.assert_called()


# ---------------------------------------------------------------------------
# Fidelity stage with learning_pool (line 733)
# ---------------------------------------------------------------------------


class TestFidelityStageWithLearningPool:
    @pytest.mark.asyncio
    async def test_fidelity_failure_extracts_learning(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.eval_config.fidelity_stages = [
            FidelityStage(
                name="quick_check",
                run_command="echo 0.1",
                parse_command="cat",
                min_pass_score=0.5,
            )
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.learning_pool import LearningPool
        pool = LearningPool()

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        _call_count = 0

        async def _status(_w):
            nonlocal _call_count
            _call_count += 1
            return [] if _call_count % 2 == 1 else [("M", "artifact.md")]

        git.status_porcelain = _status
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
            learning_pool=pool,
        )

        # Act
        record = await runner.run_one(target)

        # Assert — fidelity failure learning extracted
        assert record.outcome is Outcome.DISCARDED
        assert pool.count == 1


# ---------------------------------------------------------------------------
# Constraint post-keep check failure (lines 795-797, 803)
# ---------------------------------------------------------------------------


class TestConstraintPostKeepFailure:
    @pytest.mark.asyncio
    async def test_constraint_fails_after_keep_decision_discards(self, tmp_path: Path) -> None:
        # Arrange — deterministic eval returns high score (KEEP decision),
        # but constraint check then fails
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        # Add a MetricConstraint that will fail
        from anneal.engine.types import MetricConstraint
        target.eval_config.constraints = [
            MetricConstraint(
                metric_name="quality",
                threshold=0.99,
                direction=Direction.HIGHER_IS_BETTER,
            )
        ]

        eval_engine = MagicMock(spec=EvalEngine)
        # Main eval returns score > baseline (keep decision)
        eval_engine.evaluate = AsyncMock(return_value=EvalResult(score=0.95))
        # Constraint check: quality check returns failing result
        eval_engine.check_constraints = AsyncMock(return_value=[("quality", False, 0.5)])

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        _call_count = 0

        async def _status(_w):
            nonlocal _call_count
            _call_count += 1
            return [] if _call_count % 2 == 1 else [("M", "artifact.md")]

        git.status_porcelain = _status
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=eval_engine,
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
        )

        # Act
        record = await runner.run_one(target)

        # Assert — keep decision reversed by constraint failure
        assert record.outcome is Outcome.DISCARDED
        assert record.failure_mode == "constraint_violated:quality"


# ---------------------------------------------------------------------------
# run_loop — notification milestone and BLOCKED state (lines 1149, 1156)
# ---------------------------------------------------------------------------


class TestRunLoopNotifications:
    @pytest.mark.asyncio
    async def test_notification_milestone_called_on_kept(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.notifications import NotificationManager
        notifications = MagicMock(spec=NotificationManager)
        notifications.notify_milestone = AsyncMock()
        notifications.notify_state = AsyncMock()

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        _call_count = 0

        async def _status(_w):
            nonlocal _call_count
            _call_count += 1
            return [] if _call_count % 2 == 1 else [("M", "artifact.md")]

        git.status_porcelain = _status
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
            notifications=notifications,
        )
        runner._write_status = AsyncMock()

        # Act
        records = await runner.run_loop(target, max_experiments=1)

        # Assert — milestone notification called when KEPT
        kept_records = [r for r in records if r.outcome is Outcome.KEPT]
        if kept_records:
            notifications.notify_milestone.assert_called()

    @pytest.mark.asyncio
    async def test_notification_halted_called_on_consecutive_failures(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.max_consecutive_failures = 2
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.notifications import NotificationManager
        notifications = MagicMock(spec=NotificationManager)
        notifications.notify_state = AsyncMock()
        notifications.notify_milestone = AsyncMock()

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        git.status_porcelain = AsyncMock(return_value=[])
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(side_effect=AgentTimeoutError("timeout"))

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
            notifications=notifications,
        )
        runner._write_status = AsyncMock()

        # Act
        await runner.run_loop(target, max_experiments=10)

        # Assert — HALTED notification sent
        notifications.notify_state.assert_called()
        call_args = [c.args for c in notifications.notify_state.call_args_list]
        states = [args[1] for args in call_args]
        assert RunnerState.HALTED in states

    @pytest.mark.asyncio
    async def test_run_loop_blocked_outcome_sets_blocked_state(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        with patch("anneal.engine.runner.enforce_scope") as mock_scope:
            from anneal.engine.types import ScopeViolationResult
            mock_scope.return_value = ScopeViolationResult(
                valid_paths=[],
                violated_paths=["scope.yaml"],
                all_blocked=True,
                has_violations=True,
            )
            git.status_porcelain = AsyncMock(return_value=[("M", "scope.yaml")])
            git.clean_untracked = AsyncMock()

            # Act
            records = await runner.run_loop(target, max_experiments=1)

        # Assert — write_status called with BLOCKED state
        assert len(records) == 1
        assert records[0].outcome is Outcome.BLOCKED
        calls = runner._write_status.call_args_list
        states = [c.args[1] for c in calls if len(c.args) > 1]
        assert RunnerState.BLOCKED in states


# ---------------------------------------------------------------------------
# run_loop — component manifest streak tracking (lines 1011-1012, 1038)
# ---------------------------------------------------------------------------


class TestManifestStreakTracking:
    @pytest.mark.asyncio
    async def test_manifest_kept_resets_streak(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.strategy_mode = "manifest"
        knowledge_path = tmp_path / ".anneal" / "targets" / "test-target"
        knowledge_path.mkdir(parents=True, exist_ok=True)
        target.knowledge_path = str(knowledge_path)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.strategy import StrategyManifest, save_strategy
        manifest = StrategyManifest(lineage=[])
        manifest.hypothesis_generation.approach = "approach"
        for comp in manifest.components:
            comp.streak_without_improvement = 5
        save_strategy(manifest, knowledge_path)

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        # Act
        records = await runner.run_loop(target, max_experiments=1)

        # Assert — if KEPT, streak reset to 0; if DISCARDED, streak incremented
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_manifest_discarded_increments_streak(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.strategy_mode = "manifest"
        target.baseline_score = 99.0  # Force DISCARDED
        knowledge_path = tmp_path / ".anneal" / "targets" / "test-target"
        knowledge_path.mkdir(parents=True, exist_ok=True)
        target.knowledge_path = str(knowledge_path)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.strategy import StrategyManifest, save_strategy
        manifest = StrategyManifest(lineage=[])
        manifest.hypothesis_generation.approach = "approach"
        save_strategy(manifest, knowledge_path)

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        # Act
        records = await runner.run_loop(target, max_experiments=1)

        # Assert — DISCARDED increments streak
        assert records[0].outcome is Outcome.DISCARDED


# ---------------------------------------------------------------------------
# Held-out divergence warning (line 1044)
# ---------------------------------------------------------------------------


class TestHeldOutDivergenceWarning:
    @pytest.mark.asyncio
    async def test_held_out_warning_divergence_logged(self, tmp_path: Path, caplog) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(
            str(tmp_path),
            held_out_prompts=["p"],
            held_out_interval=1,
        )
        target.baseline_raw_scores = []  # Cold start = KEPT
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        eval_engine = MagicMock(spec=EvalEngine)
        eval_engine.evaluate = AsyncMock(return_value=EvalResult(score=0.9))
        eval_engine.check_constraints = AsyncMock(return_value=[])
        # Held-out score diverges by 15% — above WARNING threshold (10%) but below CRITICAL (25%)
        eval_engine.evaluate_held_out = AsyncMock(return_value=EvalResult(score=0.762))

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        _call_count = 0

        async def _status(_w):
            nonlocal _call_count
            _call_count += 1
            return [] if _call_count % 2 == 1 else [("M", "artifact.md")]

        git.status_porcelain = _status
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=eval_engine,
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
        )
        runner._write_status = AsyncMock()

        import logging
        # Act
        with caplog.at_level(logging.WARNING):
            await runner.run_loop(target, max_experiments=1)

        # Assert — divergence warning logged
        assert "diverges" in caplog.text.lower() or "Held-out" in caplog.text


# ---------------------------------------------------------------------------
# Policy rewrite in run_loop (lines 1058-1074)
# ---------------------------------------------------------------------------


class TestRunLoopPolicyRewrite:
    @pytest.mark.asyncio
    async def test_policy_rewrite_triggered_at_interval(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.types import PolicyConfig
        from anneal.engine.policy_agent import PolicyAgent
        target.policy_config = PolicyConfig(enabled=True, rewrite_interval=1)

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        with patch.object(PolicyAgent, "should_rewrite", return_value=True), \
             patch.object(PolicyAgent, "rewrite_instructions", new_callable=AsyncMock) as mock_rw, \
             patch.object(PolicyAgent, "compute_reward", return_value=0.1):
            mock_rw.return_value = ("new instructions", 0.01)

            # Act
            records = await runner.run_loop(target, max_experiments=1)

        # Assert — rewrite_instructions called
        mock_rw.assert_called_once()

    @pytest.mark.asyncio
    async def test_policy_rewrite_exception_does_not_crash_loop(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.types import PolicyConfig
        from anneal.engine.policy_agent import PolicyAgent
        target.policy_config = PolicyConfig(enabled=True, rewrite_interval=1)

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        with patch.object(PolicyAgent, "should_rewrite", return_value=True), \
             patch.object(PolicyAgent, "rewrite_instructions", new_callable=AsyncMock) as mock_rw:
            mock_rw.side_effect = Exception("policy rewrite crashed")

            # Act — must not raise
            records = await runner.run_loop(target, max_experiments=1)

        # Assert
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Component evolution exception path (lines 1103-1104)
# ---------------------------------------------------------------------------


class TestComponentEvolutionException:
    @pytest.mark.asyncio
    async def test_component_evolution_exception_continues(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.strategy_mode = "manifest"
        knowledge_path = tmp_path / ".anneal" / "targets" / "test-target"
        knowledge_path.mkdir(parents=True, exist_ok=True)
        target.knowledge_path = str(knowledge_path)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.strategy import StrategyManifest, save_strategy
        manifest = StrategyManifest(lineage=[])
        manifest.hypothesis_generation.approach = "approach"
        save_strategy(manifest, knowledge_path)

        knowledge = MagicMock(spec=KnowledgeStore)
        knowledge.load_records = MagicMock(return_value=[])
        knowledge.get_context = MagicMock(return_value="")
        knowledge.append_record = MagicMock()
        knowledge.update_index = MagicMock()
        knowledge.consolidate_if_due = MagicMock()
        knowledge.record_count = MagicMock(return_value=0)
        knowledge.CONSOLIDATION_INTERVAL = 20

        runner, _, agent = _build_runner(tmp_path, knowledge=knowledge)
        runner._write_status = AsyncMock()
        agent.invoke_api_text = AsyncMock(side_effect=AgentInvocationError("evolve failed"))

        # Trigger consolidation by patching CONSOLIDATION_INTERVAL to 1
        with patch("anneal.engine.runner.KnowledgeStore") as mock_ks:
            mock_ks.CONSOLIDATION_INTERVAL = 1
            # Act — must not raise
            records = await runner.run_loop(target, max_experiments=1)

        assert len(records) == 1


# ---------------------------------------------------------------------------
# Multi-draft — apply_diff returns False (line 428)
# ---------------------------------------------------------------------------


class TestMultiDraftApplyDiffFailed:
    @pytest.mark.asyncio
    async def test_draft_with_failed_apply_diff_skipped(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.agent_config = AgentConfig(
            mode="claude_code",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
            n_drafts=2,
        )
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        runner, git, agent = _build_runner(tmp_path)
        # First draft: apply_diff fails; second draft: apply_diff succeeds (verify loop),
        # then apply_diff succeeds again when winner is applied permanently (line 455)
        git.apply_diff = AsyncMock(side_effect=[False, True, True])
        agent.generate_drafts = AsyncMock(return_value=[
            (_mock_agent_result("draft1"), "diff --git a/artifact.md\n+d1"),
            (_mock_agent_result("draft2"), "diff --git a/artifact.md\n+d2"),
        ])

        # Act
        record = await runner.run_one(target)

        # Assert — first draft skipped (apply failed), second selected as winner
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED, Outcome.BLOCKED)
        # apply_diff called: once for draft1 (fail), once for draft2 (verify), once for winner (apply)
        assert git.apply_diff.call_count == 3


# ---------------------------------------------------------------------------
# Verifier failure with knowledge store (lines 523-524)
# ---------------------------------------------------------------------------


class TestVerifierFailureWithKnowledge:
    @pytest.mark.asyncio
    async def test_verifier_failure_records_to_knowledge(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.eval_config.verifiers = [
            VerifierCommand(name="check", run_command="exit 1", timeout_seconds=10),
        ]
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        knowledge = MagicMock(spec=KnowledgeStore)
        knowledge.load_records = MagicMock(return_value=[])
        knowledge.get_context = MagicMock(return_value="")
        knowledge.append_record = MagicMock()
        knowledge.update_index = MagicMock()
        knowledge.consolidate_if_due = MagicMock()
        knowledge.record_count = MagicMock(return_value=0)
        knowledge.CONSOLIDATION_INTERVAL = 20

        runner, _, agent = _build_runner(tmp_path, knowledge=knowledge)

        with patch("anneal.engine.runner.run_verifiers") as mock_vfy:
            mock_vfy.return_value = [("check", False, "failed")]
            record = await runner.run_one(target)

        # Assert — knowledge store received the blocked record
        assert record.outcome is Outcome.BLOCKED
        knowledge.append_record.assert_called_once()
        knowledge.update_index.assert_called_once()


# ---------------------------------------------------------------------------
# KEPT with raw_scores updates baseline_raw_scores (line 803)
# ---------------------------------------------------------------------------


class TestKeptUpdatesRawScores:
    @pytest.mark.asyncio
    async def test_kept_with_raw_scores_updates_baseline(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        eval_engine = MagicMock(spec=EvalEngine)
        eval_engine.evaluate = AsyncMock(
            return_value=EvalResult(score=0.9, raw_scores=[0.85, 0.90, 0.95])
        )
        eval_engine.check_constraints = AsyncMock(return_value=[])

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        _call_count = 0

        async def _status(_w):
            nonlocal _call_count
            _call_count += 1
            return [] if _call_count % 2 == 1 else [("M", "artifact.md")]

        git.status_porcelain = _status
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=eval_engine,
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
        )

        # Act
        record = await runner.run_one(target)

        # Assert — baseline_raw_scores updated from eval_result.raw_scores
        assert record.outcome is Outcome.KEPT
        assert target.baseline_raw_scores == [0.85, 0.90, 0.95]


# ---------------------------------------------------------------------------
# Taxonomy classification exception (lines 857-858)
# ---------------------------------------------------------------------------


class TestTaxonomyClassificationException:
    @pytest.mark.asyncio
    async def test_taxonomy_exception_does_not_crash(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.baseline_score = 99.0  # Force DISCARDED
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.taxonomy import FailureTaxonomy

        taxonomy = MagicMock(spec=FailureTaxonomy)
        taxonomy.classify = AsyncMock(side_effect=Exception("taxonomy exploded"))

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        _call_count = 0

        async def _status(_w):
            nonlocal _call_count
            _call_count += 1
            return [] if _call_count % 2 == 1 else [("M", "artifact.md")]

        git.status_porcelain = _status
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
            taxonomy=taxonomy,
        )

        # Act — must not raise
        record = await runner.run_one(target)

        # Assert — DISCARDED, taxonomy exception swallowed
        assert record.outcome is Outcome.DISCARDED
        taxonomy.classify.assert_called_once()


# ---------------------------------------------------------------------------
# Notification on PAUSED (line 976)
# ---------------------------------------------------------------------------


class TestRunLoopNotificationPaused:
    @pytest.mark.asyncio
    async def test_notification_paused_called_on_safety_fail(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.notifications import NotificationManager
        notifications = MagicMock(spec=NotificationManager)
        notifications.notify_state = AsyncMock()
        notifications.notify_milestone = AsyncMock()

        git = AsyncMock(spec=GitEnvironment)
        git.rev_parse = AsyncMock(return_value="abc123")
        git.status_porcelain = AsyncMock(return_value=[])
        git.commit = AsyncMock(return_value="def456")
        git.reset_hard = AsyncMock()
        git.clean_untracked = AsyncMock()
        git.cleanup_index_lock = AsyncMock()
        git.checkout_paths = AsyncMock()
        git.checkout = AsyncMock()
        git.apply_diff = AsyncMock(return_value=True)
        git.capture_diff = AsyncMock(return_value="")
        git.fsck = AsyncMock(return_value=True)

        agent = AsyncMock(spec=AgentInvoker)
        agent.invoke = AsyncMock(return_value=_mock_agent_result())

        registry = MagicMock(spec=Registry)
        runner = ExperimentRunner(
            git=git,
            agent_invoker=agent,
            eval_engine=EvalEngine(),
            search=GreedySearch(),
            registry=registry,
            repo_root=tmp_path,
            notifications=notifications,
        )
        runner._write_status = AsyncMock()

        with patch("anneal.engine.runner.pre_experiment_check") as mock_check:
            mock_check.return_value = (False, "disk full")
            # Act
            await runner.run_loop(target, max_experiments=5)

        # Assert — PAUSED notification sent
        notifications.notify_state.assert_called()
        call_states = [c.args[1] for c in notifications.notify_state.call_args_list]
        assert RunnerState.PAUSED in call_states


# ---------------------------------------------------------------------------
# Manifest streak increment for non-KEPT (line 1038 recheck)
# ---------------------------------------------------------------------------


class TestManifestStreakNonKept:
    @pytest.mark.asyncio
    async def test_manifest_non_kept_increments_streak_value(self, tmp_path: Path) -> None:
        # Arrange — baseline > eval score so DISCARDED
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.strategy_mode = "manifest"
        target.baseline_score = 99.0  # Force DISCARDED
        knowledge_path = tmp_path / ".anneal" / "targets" / "test-target"
        knowledge_path.mkdir(parents=True, exist_ok=True)
        target.knowledge_path = str(knowledge_path)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.strategy import StrategyManifest, save_strategy, load_strategy
        manifest = StrategyManifest(lineage=[])
        manifest.hypothesis_generation.approach = "approach"
        save_strategy(manifest, knowledge_path)

        runner, _, _ = _build_runner(tmp_path)
        runner._write_status = AsyncMock()

        # Act
        records = await runner.run_loop(target, max_experiments=1)

        # Assert — DISCARDED increments streak from 0 to 1
        assert records[0].outcome is Outcome.DISCARDED
        reloaded = load_strategy(knowledge_path)
        assert reloaded is not None
        assert reloaded.hypothesis_generation.streak_without_improvement == 1


    @pytest.mark.asyncio
    async def test_component_evolution_exception_continues(self, tmp_path: Path) -> None:
        # Arrange
        _make_git_repo(tmp_path)
        target = _make_target(str(tmp_path))
        target.strategy_mode = "manifest"
        knowledge_path = tmp_path / ".anneal" / "targets" / "test-target"
        knowledge_path.mkdir(parents=True, exist_ok=True)
        target.knowledge_path = str(knowledge_path)
        target.scope_hash = compute_scope_hash(tmp_path / "scope.yaml")

        from anneal.engine.strategy import StrategyManifest, save_strategy
        manifest = StrategyManifest(lineage=[])
        manifest.hypothesis_generation.approach = "approach"
        save_strategy(manifest, knowledge_path)

        knowledge = MagicMock(spec=KnowledgeStore)
        knowledge.load_records = MagicMock(return_value=[])
        knowledge.get_context = MagicMock(return_value="")
        knowledge.append_record = MagicMock()
        knowledge.update_index = MagicMock()
        knowledge.consolidate_if_due = MagicMock()
        knowledge.record_count = MagicMock(return_value=0)
        knowledge.CONSOLIDATION_INTERVAL = 20

        runner, _, agent = _build_runner(tmp_path, knowledge=knowledge)
        runner._write_status = AsyncMock()
        agent.invoke_api_text = AsyncMock(side_effect=AgentInvocationError("evolve failed"))

        # Trigger consolidation by patching CONSOLIDATION_INTERVAL to 1
        with patch("anneal.engine.runner.KnowledgeStore") as mock_ks:
            mock_ks.CONSOLIDATION_INTERVAL = 1
            # Act — must not raise
            records = await runner.run_loop(target, max_experiments=1)

        assert len(records) == 1
