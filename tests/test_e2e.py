"""End-to-end tests for the full experiment cycle.

Uses real git operations and deterministic eval commands.
Agent I/O is mocked to avoid LLM API calls.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anneal.engine.agent import AgentInvoker
from anneal.engine.environment import GitEnvironment
from anneal.engine.eval import EvalEngine
from anneal.engine.knowledge import KnowledgeStore
from anneal.engine.runner import ExperimentRunner
from anneal.engine.scope import compute_scope_hash
from anneal.engine.search import GreedySearch
from anneal.engine.types import (
    AgentConfig,
    AgentInvocationResult,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    OptimizationTarget,
    Outcome,
)


def _make_agent_invoker(worktree_ref: list[Path]) -> AgentInvoker:
    """Build a real AgentInvoker subclass that mutates the artifact without LLM calls."""

    class _MockInvoker(AgentInvoker):
        async def invoke(  # type: ignore[override]
            self,
            config: AgentConfig,
            prompt: str,
            worktree_path: Path,
            time_budget_seconds: int,
            deployment_mode: bool = False,
        ) -> AgentInvocationResult:
            artifact = worktree_path / "artifact.md"
            current = artifact.read_text()
            artifact.write_text(current + "additional content line\n")
            worktree_ref.append(worktree_path)
            return AgentInvocationResult(
                success=True,
                cost_usd=0.01,
                input_tokens=100,
                output_tokens=50,
                hypothesis="add more content to artifact",
                hypothesis_source="agent",
                tags=["e2e-test"],
                raw_output=(
                    "## Hypothesis\nadd more content to artifact\n## Tags\ne2e-test"
                ),
            )

    return _MockInvoker()


def _build_target(tmp_path: Path, target_id: str = "e2e-target") -> OptimizationTarget:
    """Build an OptimizationTarget for the e2e test repo."""
    knowledge_path = tmp_path / ".anneal" / "knowledge"
    scope_hash = compute_scope_hash(tmp_path / "scope.yaml")
    return OptimizationTarget(
        id=target_id,
        domain_tier=DomainTier.SANDBOX,
        artifact_paths=["artifact.md"],
        scope_path="scope.yaml",
        scope_hash=scope_hash,
        eval_mode=EvalMode.DETERMINISTIC,
        eval_config=EvalConfig(
            metric_name="word_count",
            direction=Direction.HIGHER_IS_BETTER,
            deterministic=DeterministicEval(
                run_command="wc -w < artifact.md",
                parse_command="cat",
                timeout_seconds=10,
            ),
        ),
        agent_config=AgentConfig(
            mode="api",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
        ),
        time_budget_seconds=60,
        loop_interval_seconds=0,
        knowledge_path=str(knowledge_path),
        worktree_path=str(tmp_path),
        git_branch="main",
        baseline_score=2.0,
    )


def _build_runner(
    tmp_path: Path,
    knowledge: KnowledgeStore,
    target_id: str = "e2e-target",
) -> tuple[ExperimentRunner, list[Path]]:
    """Build an ExperimentRunner with a mock agent and real everything else."""
    worktree_ref: list[Path] = []
    agent = _make_agent_invoker(worktree_ref)

    registry_mock = MagicMock()
    registry_mock.update_target = MagicMock()

    runner = ExperimentRunner(
        git=GitEnvironment(),
        agent_invoker=agent,
        eval_engine=EvalEngine(),
        search=GreedySearch(),
        registry=registry_mock,
        repo_root=tmp_path,
        knowledge=knowledge,
        notifications=None,
        learning_pool=None,
    )
    return runner, worktree_ref


@pytest.mark.e2e
class TestFullExperimentCycle:
    """End-to-end tests of the complete experiment pipeline."""

    @pytest.mark.asyncio
    async def test_run_one_produces_record(self, e2e_git_repo: Path) -> None:
        """Full cycle: git repo -> mock agent -> real eval -> search -> knowledge."""
        tmp_path = e2e_git_repo
        knowledge_path = tmp_path / ".anneal" / "knowledge"
        knowledge = KnowledgeStore(knowledge_path)
        target = _build_target(tmp_path)
        runner, _ = _build_runner(tmp_path, knowledge)

        record = await runner.run_one(target)

        assert record is not None
        assert record.target_id == "e2e-target"
        assert record.outcome in (Outcome.KEPT, Outcome.DISCARDED)
        assert record.score > 0
        assert record.duration_seconds > 0
        assert record.cost_usd >= 0

    @pytest.mark.asyncio
    async def test_run_one_persists_to_knowledge_store(self, e2e_git_repo: Path) -> None:
        """Record is appended to the knowledge store after run_one."""
        tmp_path = e2e_git_repo
        knowledge_path = tmp_path / ".anneal" / "knowledge"
        knowledge = KnowledgeStore(knowledge_path)
        target = _build_target(tmp_path)
        runner, _ = _build_runner(tmp_path, knowledge)

        record = await runner.run_one(target)

        assert knowledge.record_count() == 1
        stored = knowledge.load_records()
        assert stored[0].id == record.id

    @pytest.mark.asyncio
    async def test_run_one_kept_increases_word_count(self, e2e_git_repo: Path) -> None:
        """When KEPT, the artifact score reflects the appended content."""
        tmp_path = e2e_git_repo
        knowledge_path = tmp_path / ".anneal" / "knowledge"
        knowledge = KnowledgeStore(knowledge_path)
        target = _build_target(tmp_path)
        original_baseline = target.baseline_score
        runner, _ = _build_runner(tmp_path, knowledge)

        record = await runner.run_one(target)

        # "hello world\nadditional content line\n" = 5 words (wc -w counts words)
        # original baseline was 2.0 ("hello world")
        if record.outcome is Outcome.KEPT:
            assert record.score > original_baseline

    @pytest.mark.asyncio
    async def test_run_loop_completes_n_experiments(self, e2e_git_repo: Path) -> None:
        """run_loop completes the requested number of experiments."""
        tmp_path = e2e_git_repo
        knowledge_path = tmp_path / ".anneal" / "knowledge"
        knowledge = KnowledgeStore(knowledge_path)
        target = _build_target(tmp_path, target_id="e2e-loop")
        runner, _ = _build_runner(tmp_path, knowledge, target_id="e2e-loop")

        records = await runner.run_loop(target, max_experiments=3)

        assert len(records) == 3
        assert knowledge.record_count() == 3

    @pytest.mark.asyncio
    async def test_run_loop_first_experiment_kept(self, e2e_git_repo: Path) -> None:
        """First experiment is KEPT because word count strictly increases."""
        tmp_path = e2e_git_repo
        knowledge_path = tmp_path / ".anneal" / "knowledge"
        knowledge = KnowledgeStore(knowledge_path)
        target = _build_target(tmp_path, target_id="e2e-kept")
        runner, _ = _build_runner(tmp_path, knowledge, target_id="e2e-kept")

        records = await runner.run_loop(target, max_experiments=1)

        assert len(records) == 1
        assert records[0].outcome is Outcome.KEPT

    @pytest.mark.asyncio
    async def test_run_one_outcome_not_crashed(self, e2e_git_repo: Path) -> None:
        """The pipeline must not crash — outcome must be a terminal non-error state."""
        tmp_path = e2e_git_repo
        knowledge_path = tmp_path / ".anneal" / "knowledge"
        knowledge = KnowledgeStore(knowledge_path)
        target = _build_target(tmp_path)
        runner, _ = _build_runner(tmp_path, knowledge)

        record = await runner.run_one(target)

        assert record.outcome not in (Outcome.CRASHED, Outcome.KILLED)
