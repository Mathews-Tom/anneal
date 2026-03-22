"""Tests for MAP-Elites quality-diversity archive."""
from __future__ import annotations

from pathlib import Path

from anneal.engine.archive import ArchiveEntry, MapElitesArchive


class TestMapElitesArchive:
    def test_add_new_cell_returns_true(self) -> None:
        archive = MapElitesArchive(n_bins=10)
        added = archive.add("exp-1", {"clarity": 0.8, "accuracy": 0.6}, fitness=0.7)
        assert added is True
        assert archive.coverage == 1

    def test_add_better_fitness_replaces(self) -> None:
        archive = MapElitesArchive(n_bins=10)
        archive.add("exp-1", {"a": 0.5}, fitness=0.5)
        replaced = archive.add("exp-2", {"a": 0.5}, fitness=0.8)
        assert replaced is True
        entry = archive.get({"a": 0.5})
        assert entry is not None
        assert entry.record_id == "exp-2"

    def test_add_worse_fitness_rejected(self) -> None:
        archive = MapElitesArchive(n_bins=10)
        archive.add("exp-1", {"a": 0.5}, fitness=0.8)
        rejected = archive.add("exp-2", {"a": 0.5}, fitness=0.3)
        assert rejected is False
        entry = archive.get({"a": 0.5})
        assert entry is not None
        assert entry.record_id == "exp-1"

    def test_different_behaviors_occupy_different_cells(self) -> None:
        archive = MapElitesArchive(n_bins=10)
        archive.add("exp-1", {"a": 0.1}, fitness=0.5)
        archive.add("exp-2", {"a": 0.9}, fitness=0.5)
        assert archive.coverage == 2

    def test_get_diverse_solutions_returns_top_k(self) -> None:
        archive = MapElitesArchive(n_bins=10)
        archive.add("exp-1", {"a": 0.1}, fitness=0.3)
        archive.add("exp-2", {"a": 0.5}, fitness=0.9)
        archive.add("exp-3", {"a": 0.9}, fitness=0.6)
        top = archive.get_diverse_solutions(k=2)
        assert len(top) == 2
        assert top[0].fitness == 0.9
        assert top[1].fitness == 0.6

    def test_discretize_clamps_values(self) -> None:
        archive = MapElitesArchive(n_bins=10)
        # Values outside [0, 1] should be clamped
        archive.add("exp-1", {"a": -0.5, "b": 1.5}, fitness=0.5)
        assert archive.coverage == 1
        entry = archive.get({"a": 0.0, "b": 1.0})
        assert entry is not None

    def test_persist_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "archive.jsonl"
        archive1 = MapElitesArchive(n_bins=10, persist_path=path)
        archive1.add("exp-1", {"a": 0.5, "b": 0.7}, fitness=0.8)
        archive1.add("exp-2", {"a": 0.1, "b": 0.2}, fitness=0.6)

        # Reload from file
        archive2 = MapElitesArchive(n_bins=10, persist_path=path)
        assert archive2.coverage == 2
        entry = archive2.get({"a": 0.5, "b": 0.7})
        assert entry is not None
        assert entry.record_id == "exp-1"

    def test_max_cells_reflects_dimensions(self) -> None:
        archive = MapElitesArchive(n_bins=5)
        archive.add("exp-1", {"a": 0.5, "b": 0.5}, fitness=0.5)
        assert archive.max_cells == 25  # 5^2

    def test_empty_archive_max_cells_zero(self) -> None:
        archive = MapElitesArchive(n_bins=10)
        assert archive.max_cells == 0
