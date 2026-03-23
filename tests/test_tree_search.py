"""Tests for UCB tree search over artifact space (E4)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from anneal.engine.tree_search import TreeNode, UCBTreeSearch
from anneal.engine.types import Direction, EvalResult, ExperimentRecord, Outcome


def _make_record(
    pre_sha: str,
    git_sha: str,
    score: float,
    baseline_score: float = 0.0,
    outcome: Outcome = Outcome.KEPT,
) -> ExperimentRecord:
    return ExperimentRecord(
        id="test", target_id="t", git_sha=git_sha, pre_experiment_sha=pre_sha,
        timestamp=datetime.now(tz=timezone.utc), hypothesis="h",
        hypothesis_source="agent", mutation_diff_summary="", score=score,
        score_ci_lower=None, score_ci_upper=None, raw_scores=None,
        baseline_score=baseline_score, outcome=outcome, failure_mode=None,
        duration_seconds=1.0, tags=[], learnings="", cost_usd=0.01,
        bootstrap_seed=0,
    )


class TestUCBSelection:
    def test_prefers_unvisited_nodes(self) -> None:
        tree = UCBTreeSearch()
        tree._add_node("root", score=0.5)
        tree._add_node("a", score=0.6, parent_sha="root")
        tree._add_node("b", score=0.4, parent_sha="root")
        tree._nodes["root"].visit_count = 10
        tree._nodes["a"].visit_count = 10
        tree._nodes["b"].visit_count = 0
        tree._total_visits = 10
        selected = tree.select_parent()
        assert selected == "b"  # Unvisited gets inf UCB

    def test_balances_exploitation_and_exploration(self) -> None:
        tree = UCBTreeSearch(exploration_constant=1.0)
        tree._add_node("root", score=0.5)
        tree._add_node("a", score=0.9, parent_sha="root")
        tree._add_node("b", score=0.3, parent_sha="root")
        tree._nodes["a"].visit_count = 50
        tree._nodes["b"].visit_count = 2
        tree._nodes["root"].visit_count = 52
        tree._total_visits = 52
        # b is low-scoring but under-explored
        selected = tree.select_parent()
        # UCB(a) = 0.9 + 1.0*sqrt(ln(52)/50) ~ 0.9 + 0.28 = 1.18
        # UCB(b) = 0.3 + 1.0*sqrt(ln(52)/2) ~ 0.3 + 1.40 = 1.70
        assert selected == "b"

    def test_empty_tree_raises(self) -> None:
        tree = UCBTreeSearch()
        with pytest.raises(ValueError, match="empty"):
            tree.select_parent()


class TestRecordOutcome:
    def test_adds_child_node(self) -> None:
        tree = UCBTreeSearch()
        tree._add_node("root", score=0.5)
        tree.record_outcome("root", "child1", score=0.7, kept=True)
        assert "child1" in tree._nodes
        assert tree._nodes["child1"].score == 0.7
        assert tree._nodes["child1"].parent_sha == "root"

    def test_increments_visit_counts(self) -> None:
        tree = UCBTreeSearch()
        tree._add_node("root", score=0.5)
        tree.record_outcome("root", "c1", score=0.6, kept=True)
        tree.record_outcome("root", "c2", score=0.4, kept=False)
        assert tree._nodes["root"].visit_count >= 2
        assert tree._total_visits >= 2


class TestPruning:
    def test_pruned_nodes_not_selected(self) -> None:
        tree = UCBTreeSearch()
        tree._add_node("root", score=0.5)
        tree._add_node("a", score=0.6, parent_sha="root")
        tree._add_node("b", score=0.4, parent_sha="root")
        tree._nodes["a"].visit_count = 1
        tree._nodes["b"].visit_count = 1
        tree._total_visits = 2
        tree.prune_subtree("a")
        assert tree._nodes["a"].pruned is True
        selected = tree.select_parent()
        assert selected != "a"

    def test_root_not_pruned(self) -> None:
        tree = UCBTreeSearch()
        tree._add_node("root", score=0.5)
        tree.prune_subtree("root")
        assert tree._nodes["root"].pruned is False

    def test_prune_cascades_to_children(self) -> None:
        tree = UCBTreeSearch()
        tree._add_node("root", score=0.5)
        tree._add_node("a", score=0.6, parent_sha="root")
        tree._add_node("b", score=0.4, parent_sha="a")
        tree._add_node("c", score=0.3, parent_sha="a")
        tree.prune_subtree("a")
        assert tree._nodes["a"].pruned is True
        assert tree._nodes["b"].pruned is True
        assert tree._nodes["c"].pruned is True


class TestBootstrapFromHistory:
    def test_rebuilds_tree_from_records(self) -> None:
        records = [
            _make_record("root", "a", score=0.6, baseline_score=0.5),
            _make_record("root", "b", score=0.4, baseline_score=0.5),
            _make_record("a", "c", score=0.8, baseline_score=0.6),
        ]
        tree = UCBTreeSearch()
        tree.bootstrap_from_history(records)
        assert tree.node_count == 4  # root, a, b, c
        assert tree._nodes["c"].parent_sha == "a"
        assert tree._nodes["c"].score == 0.8

    def test_empty_history(self) -> None:
        tree = UCBTreeSearch()
        tree.bootstrap_from_history([])
        assert tree.node_count == 0


class TestPersistence:
    def test_persist_and_load_roundtrip(self, tmp_path: Path) -> None:
        tree = UCBTreeSearch(exploration_constant=2.0, prune_threshold=3)
        tree._add_node("root", score=0.5)
        tree._add_node("a", score=0.7, parent_sha="root")
        tree._add_node("b", score=0.3, parent_sha="root")
        tree._nodes["a"].visit_count = 5
        tree._total_visits = 10
        tree.prune_subtree("b")

        path = tmp_path / "search_tree.json"
        tree.persist(path)
        assert path.exists()

        loaded = UCBTreeSearch.load(path)
        assert loaded.node_count == 3
        assert loaded._c == 2.0
        assert loaded._nodes["a"].visit_count == 5
        assert loaded._nodes["b"].pruned is True
        assert loaded._root is not None
        assert loaded._root.sha == "root"
        assert loaded._total_visits == 10

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            UCBTreeSearch.load(path)

    def test_bootstrap_fallback_when_json_missing(self, tmp_path: Path) -> None:
        """When search_tree.json doesn't exist, bootstrap from history."""
        records = [
            _make_record("root", "a", score=0.6),
            _make_record("a", "b", score=0.8),
        ]
        path = tmp_path / "search_tree.json"
        assert not path.exists()

        tree = UCBTreeSearch()
        tree.bootstrap_from_history(records)
        assert tree.node_count == 3


class TestShouldKeep:
    def test_higher_is_better(self) -> None:
        tree = UCBTreeSearch()
        result = EvalResult(score=0.8)
        assert tree.should_keep(result, 0.5, None, Direction.HIGHER_IS_BETTER) is True
        assert tree.should_keep(result, 0.9, None, Direction.HIGHER_IS_BETTER) is False

    def test_lower_is_better(self) -> None:
        tree = UCBTreeSearch()
        result = EvalResult(score=0.3)
        assert tree.should_keep(result, 0.5, None, Direction.LOWER_IS_BETTER) is True
        assert tree.should_keep(result, 0.1, None, Direction.LOWER_IS_BETTER) is False

    def test_threshold(self) -> None:
        tree = UCBTreeSearch()
        result = EvalResult(score=0.51)
        assert tree.should_keep(result, 0.5, None, Direction.HIGHER_IS_BETTER, 0.1) is False


class TestTreeInfo:
    def test_empty_tree_info(self) -> None:
        tree = UCBTreeSearch()
        info = tree.get_tree_info()
        assert info["nodes"] == 0

    def test_populated_tree_info(self) -> None:
        tree = UCBTreeSearch()
        tree._add_node("root", score=0.5)
        tree._add_node("a", score=0.6, parent_sha="root")
        tree._add_node("b", score=0.3, parent_sha="a")
        tree.prune_subtree("b")
        info = tree.get_tree_info()
        assert info["nodes"] == 3
        assert info["depth"] == 2
        assert info["pruned"] == 1
