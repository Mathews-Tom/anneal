"""Tests for cross-project learning pool (F5): GlobalLearningPool and project_id filter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from anneal.engine.learning_pool import (
    GlobalLearningPool,
    Learning,
    LearningPool,
    LearningScope,
    LearningSignal,
    PersistentLearningPool,
    get_global_pool_path,
)


def _make_learning(
    *,
    observation: str = "obs",
    score_delta: float = 0.3,
    project_id: str = "",
    source_target: str = "target-1",
) -> Learning:
    return Learning(
        observation=observation,
        signal=LearningSignal.POSITIVE,
        source_condition="guided",
        source_target=source_target,
        source_experiment_ids=[1],
        score_delta=score_delta,
        criterion_deltas={},
        confidence=1.0,
        tags=[],
        created_at=datetime.now(UTC),
        project_id=project_id,
    )


class TestGetGlobalPoolPath:
    def test_returns_home_based_path(self) -> None:
        path = get_global_pool_path()
        assert path == Path.home() / ".anneal" / "global-learnings.jsonl"

    def test_path_is_absolute(self) -> None:
        path = get_global_pool_path()
        assert path.is_absolute()


class TestGlobalLearningPool:
    def test_init_creates_parent_directory(self, tmp_path: Path) -> None:
        pool_dir = tmp_path / ".anneal"
        with patch(
            "anneal.engine.learning_pool.get_global_pool_path",
            return_value=pool_dir / "global-learnings.jsonl",
        ):
            GlobalLearningPool()
        assert pool_dir.exists()

    def test_add_and_retrieve(self, tmp_path: Path) -> None:
        pool_file = tmp_path / "global-learnings.jsonl"
        with patch(
            "anneal.engine.learning_pool.get_global_pool_path",
            return_value=pool_file,
        ):
            pool = GlobalLearningPool()
            pool.add(_make_learning(observation="global-1", project_id="proj-a"))
            pool.add(_make_learning(observation="global-2", project_id="proj-b"))
            assert pool.count == 2

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        pool_file = tmp_path / "global-learnings.jsonl"
        with patch(
            "anneal.engine.learning_pool.get_global_pool_path",
            return_value=pool_file,
        ):
            pool1 = GlobalLearningPool()
            pool1.add(_make_learning(observation="persist-test"))
            assert pool1.count == 1

            pool2 = GlobalLearningPool()
            assert pool2.count == 1
            results = pool2.retrieve(scope=LearningScope.GLOBAL, k=1)
            assert results[0].observation == "persist-test"


class TestProjectIdFilter:
    def test_filter_by_project_id(self) -> None:
        pool = LearningPool()
        pool.add(_make_learning(observation="a", project_id="proj-a"))
        pool.add(_make_learning(observation="b", project_id="proj-b"))
        pool.add(_make_learning(observation="c", project_id="proj-a"))

        results = pool.retrieve(scope=LearningScope.GLOBAL, k=10, project_id="proj-a")
        assert len(results) == 2
        assert all("proj-a" == r.project_id for r in results)

    def test_no_filter_returns_all(self) -> None:
        pool = LearningPool()
        pool.add(_make_learning(project_id="proj-a"))
        pool.add(_make_learning(project_id="proj-b"))

        results = pool.retrieve(scope=LearningScope.GLOBAL, k=10)
        assert len(results) == 2

    def test_filter_nonexistent_project_returns_empty(self) -> None:
        pool = LearningPool()
        pool.add(_make_learning(project_id="proj-a"))

        results = pool.retrieve(scope=LearningScope.GLOBAL, k=10, project_id="nonexistent")
        assert results == []

    def test_persistent_pool_preserves_project_id(self, tmp_path: Path) -> None:
        pool1 = PersistentLearningPool(tmp_path)
        pool1.add(_make_learning(observation="proj-test", project_id="my-proj"))

        pool2 = PersistentLearningPool(tmp_path)
        results = pool2.retrieve(scope=LearningScope.GLOBAL, k=1)
        assert results[0].project_id == "my-proj"
