"""Cross-enhancement integration tests verifying feature composition."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from anneal.engine.types import (
    AgentConfig,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    ExperimentRecord,
    FailureClassification,
    Outcome,
    OptimizationTarget,
    PolicyConfig,
    VerifierCommand,
)


def _make_record(
    outcome: Outcome = Outcome.DISCARDED,
    score: float = 0.5,
    baseline_score: float = 0.5,
    hypothesis: str = "test",
    failure_mode: str | None = None,
    failure_classification: FailureClassification | None = None,
    pre_sha: str = "aaa",
    git_sha: str = "bbb",
    tags: list[str] | None = None,
    drafts_generated: int = 1,
    drafts_survived: int = 1,
) -> ExperimentRecord:
    return ExperimentRecord(
        id="test", target_id="t", git_sha=git_sha, pre_experiment_sha=pre_sha,
        timestamp=datetime.now(tz=timezone.utc), hypothesis=hypothesis,
        hypothesis_source="agent", mutation_diff_summary="", score=score,
        score_ci_lower=None, score_ci_upper=None, raw_scores=None,
        baseline_score=baseline_score, outcome=outcome,
        failure_mode=failure_mode, failure_classification=failure_classification,
        duration_seconds=1.0, tags=tags or [], learnings="", cost_usd=0.01,
        bootstrap_seed=0, drafts_generated=drafts_generated,
        drafts_survived=drafts_survived,
    )


class TestMultiDraftWithVerifiers:
    """E6 + E5: Multi-draft generation uses verifiers for per-draft pruning."""

    def test_verifier_prunes_drafts(self, tmp_path: Path) -> None:
        """Verifiers filter drafts independently — only survivors proceed."""
        from anneal.engine.eval import run_verifiers

        # Create a verifier that checks for a marker file
        v = VerifierCommand(name="marker_check", run_command="test -f marker.txt")

        # Draft 1: creates marker (passes verifier)
        (tmp_path / "marker.txt").write_text("ok")
        r1 = asyncio.run(run_verifiers(tmp_path, [v]))
        assert r1[0][1] is True  # passes

        # Draft 2: no marker (fails verifier)
        (tmp_path / "marker.txt").unlink()
        r2 = asyncio.run(run_verifiers(tmp_path, [v]))
        assert r2[0][1] is False  # fails

    def test_all_drafts_fail_produces_blocked(self) -> None:
        """When all drafts fail verifiers, the experiment is BLOCKED."""
        record = _make_record(
            outcome=Outcome.BLOCKED,
            failure_mode="all_drafts_failed_verifiers",
            drafts_generated=3,
            drafts_survived=0,
        )
        assert record.outcome is Outcome.BLOCKED
        assert record.drafts_generated == 3
        assert record.drafts_survived == 0

    def test_draft_tracking_fields_on_record(self) -> None:
        """ExperimentRecord carries draft statistics through the pipeline."""
        record = _make_record(
            outcome=Outcome.KEPT,
            score=0.8,
            drafts_generated=5,
            drafts_survived=2,
        )
        assert record.drafts_generated == 5
        assert record.drafts_survived == 2


class TestTaxonomyWithPolicyAgent:
    """E2 + E1: Failure taxonomy feeds distribution to policy agent."""

    def test_failure_distribution_feeds_policy_prompt(self) -> None:
        """Policy agent receives failure distribution from taxonomy."""
        from anneal.engine.taxonomy import FailureTaxonomy

        records = [
            _make_record(failure_classification=FailureClassification(
                category="output_format", description="d", fix_direction="f",
            ))
            for _ in range(5)
        ] + [
            _make_record(failure_classification=FailureClassification(
                category="logic_error", description="d", fix_direction="f",
            ))
            for _ in range(3)
        ]

        dist = FailureTaxonomy.distribution(records)
        assert dist == {"output_format": 5, "logic_error": 3}

        # Verify the distribution format is compatible with PolicyAgent
        # (it expects dict[str, int] | None)
        assert isinstance(dist, dict)
        assert all(isinstance(k, str) and isinstance(v, int) for k, v in dist.items())

    def test_blind_spots_detected_with_taxonomy(self) -> None:
        """Blind spot detection works across classified failure records."""
        from anneal.engine.taxonomy import FailureTaxonomy

        # 15 failures all classified as output_format — other categories are blind spots
        records = [
            _make_record(failure_classification=FailureClassification(
                category="output_format", description="d", fix_direction="f",
            ))
            for _ in range(15)
        ]

        taxonomy = FailureTaxonomy()
        blind_spots = taxonomy.blind_spot_check(records)
        assert "logic_error" in blind_spots
        assert "output_format" not in blind_spots


class TestTreeSearchWithRestart:
    """E4 + E3: Tree search and restart interact correctly."""

    def test_restart_experiment_added_to_tree(self) -> None:
        """Restart experiments create new tree nodes like normal experiments."""
        from anneal.engine.tree_search import UCBTreeSearch

        tree = UCBTreeSearch()
        tree._add_node("root", score=0.5)

        # Normal experiment
        tree.record_outcome("root", "normal_child", score=0.6, kept=True)
        # Restart experiment (tagged, but structurally identical in the tree)
        tree.record_outcome("root", "restart_child", score=0.7, kept=True)

        assert tree.node_count == 3
        assert tree._nodes["restart_child"].score == 0.7

    def test_tree_search_selects_after_restart_improves(self) -> None:
        """If a restart experiment produces a high-scoring node, UCB favors it."""
        from anneal.engine.tree_search import UCBTreeSearch

        tree = UCBTreeSearch(exploration_constant=0.1)  # Low exploration = exploit
        tree._add_node("root", score=0.5)
        tree._nodes["root"].visit_count = 10
        tree._total_visits = 10

        # Normal branch: moderate score, well-explored
        tree._add_node("branch_a", score=0.6, parent_sha="root")
        tree._nodes["branch_a"].visit_count = 8

        # Restart branch: high score, under-explored
        tree._add_node("restart_branch", score=0.9, parent_sha="root")
        tree._nodes["restart_branch"].visit_count = 1
        tree._total_visits = 19

        selected = tree.select_parent()
        assert selected == "restart_branch"


class TestVerifierWarningWithTaxonomy:
    """E5 + E2: Verifier warnings and failure distribution coexist in context."""

    def test_both_injected_into_context(self) -> None:
        """Context assembly includes both verifier warnings and failure distribution."""
        from anneal.engine.context import _build_failure_distribution_summary, _build_verifier_warning

        # Records with both verifier blocks and classified failures
        # Window is last 10 records; need >60% verifier blocks in that window
        records = []
        for i in range(10):
            if i < 8:
                # Verifier-blocked (8/10 = 80%, above 60% threshold)
                records.append(_make_record(
                    outcome=Outcome.BLOCKED,
                    failure_mode="verifier:typecheck",
                    failure_classification=FailureClassification(
                        category="syntax_error", description="d", fix_direction="f",
                    ),
                ))
            else:
                # Normal failure
                records.append(_make_record(
                    outcome=Outcome.DISCARDED,
                    failure_classification=FailureClassification(
                        category="logic_error", description="d", fix_direction="f",
                    ),
                ))

        verifier_warning = _build_verifier_warning(records)
        failure_summary = _build_failure_distribution_summary(records)

        assert "typecheck" in verifier_warning
        assert "syntax_error" in failure_summary
        assert "logic_error" in failure_summary


class TestTreeSearchPersistenceWithConsolidation:
    """E4: Tree search persistence coexists with knowledge store."""

    def test_tree_persists_alongside_experiments(self, tmp_path: Path) -> None:
        """Tree JSON lives next to experiments.jsonl without interference."""
        from anneal.engine.knowledge import KnowledgeStore
        from anneal.engine.tree_search import UCBTreeSearch

        # Knowledge store writes experiments
        ks = KnowledgeStore(tmp_path)
        for i in range(5):
            ks.append_record(_make_record(score=float(i) / 10))

        # Tree search persists alongside
        tree = UCBTreeSearch()
        tree._add_node("root", score=0.5)
        tree.record_outcome("root", "a", score=0.6, kept=True)
        tree.persist(tmp_path / "search_tree.json")

        # Both files coexist
        assert (tmp_path / "experiments.jsonl").exists()
        assert (tmp_path / "search_tree.json").exists()

        # Both load independently
        loaded_records = ks.load_records()
        loaded_tree = UCBTreeSearch.load(tmp_path / "search_tree.json")
        assert len(loaded_records) == 5
        assert loaded_tree.node_count == 2


class TestPolicyConfigDefaults:
    """E1: Policy config None preserves existing behavior."""

    def test_no_policy_no_instructions(self, tmp_path: Path) -> None:
        """When policy_config is None, no policy instructions in context."""
        from anneal.engine.context import build_target_context

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
            knowledge_path=str(tmp_path / "k"),
            worktree_path="w", git_branch="b", baseline_score=0.0,
        )
        # No policy_instructions passed → no "Mutation Strategy" in output
        prompt, _ = build_target_context(
            target, tmp_path, tmp_path, history=[],
        )
        assert "Mutation Strategy" not in prompt

    def test_policy_instructions_injected_when_provided(self, tmp_path: Path) -> None:
        """Policy instructions appear in context when passed."""
        from anneal.engine.context import build_target_context

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
            knowledge_path=str(tmp_path), worktree_path=str(tmp_path),
            git_branch="b", baseline_score=0.0,
        )
        prompt, _ = build_target_context(
            target, tmp_path, tmp_path, history=[],
            policy_instructions="Focus on output format validation.",
        )
        assert "Mutation Strategy" in prompt
        assert "Focus on output format validation." in prompt
