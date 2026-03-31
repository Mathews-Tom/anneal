"""Tests for Pareto front persistence and dashboard visualization."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anneal.engine.dashboard import AnnealStateReader, FilePollingBus
from anneal.engine.search import ParetoSearch


class TestParetoFrontPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        search = ParetoSearch()
        search._pareto_front = [
            {"style": 0.8, "correctness": 0.6},
            {"style": 0.5, "correctness": 0.9},
        ]
        front_path = tmp_path / "pareto_front.json"
        search.save_front(front_path)

        new_search = ParetoSearch()
        new_search.load_front(front_path)
        assert new_search.pareto_front == search.pareto_front

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        search = ParetoSearch()
        search._pareto_front = [{"x": 0.5}]
        front_path = tmp_path / "nested" / "dir" / "pareto_front.json"
        search.save_front(front_path)
        assert front_path.exists()

    def test_load_nonexistent_path(self, tmp_path: Path) -> None:
        search = ParetoSearch()
        search._pareto_front = [{"x": 0.5}]
        search.load_front(tmp_path / "missing.json")
        # Front should remain unchanged
        assert search.pareto_front == [{"x": 0.5}]

    def test_empty_front_roundtrip(self, tmp_path: Path) -> None:
        search = ParetoSearch()
        front_path = tmp_path / "pareto_front.json"
        search.save_front(front_path)

        loaded = ParetoSearch()
        loaded.load_front(front_path)
        assert loaded.pareto_front == []


class TestDashboardParetoFront:
    def test_read_pareto_front_targets_layout(self, tmp_path: Path) -> None:
        reader = AnnealStateReader(tmp_path)
        target_dir = tmp_path / "targets" / "t1"
        target_dir.mkdir(parents=True)
        front = [{"style": 0.8, "correctness": 0.6}]
        (target_dir / "pareto_front.json").write_text(json.dumps(front))

        result = reader._read_pareto_front("t1")
        assert result == front

    def test_read_pareto_front_flat_layout(self, tmp_path: Path) -> None:
        reader = AnnealStateReader(tmp_path)
        target_dir = tmp_path / "t1"
        target_dir.mkdir(parents=True)
        front = [{"x": 0.5, "y": 0.7}]
        (target_dir / "pareto_front.json").write_text(json.dumps(front))

        result = reader._read_pareto_front("t1")
        assert result == front

    def test_read_pareto_front_missing(self, tmp_path: Path) -> None:
        reader = AnnealStateReader(tmp_path)
        assert reader._read_pareto_front("t1") is None

    def test_build_snapshot_includes_pareto(self, tmp_path: Path) -> None:
        reader = AnnealStateReader(tmp_path)
        # Create target with experiments and pareto front
        target_dir = tmp_path / "targets" / "t1"
        target_dir.mkdir(parents=True)
        front = [{"style": 0.8, "correctness": 0.6}]
        (target_dir / "pareto_front.json").write_text(json.dumps(front))
        # Need experiments.jsonl for target discovery
        (target_dir / "experiments.jsonl").write_text("")
        # Need config.toml for discovery
        (tmp_path / "config.toml").write_text('[targets.t1]\neval_mode = "stochastic"\n')

        snapshot = reader.build_snapshot()
        assert "pareto_front" in snapshot["targets"]["t1"]
        assert snapshot["targets"]["t1"]["pareto_front"] == front

    def test_build_snapshot_no_pareto_for_non_pareto_targets(self, tmp_path: Path) -> None:
        reader = AnnealStateReader(tmp_path)
        (tmp_path / "config.toml").write_text('[targets.t1]\neval_mode = "deterministic"\n')
        target_dir = tmp_path / "targets" / "t1"
        target_dir.mkdir(parents=True)
        (target_dir / "experiments.jsonl").write_text("")

        snapshot = reader.build_snapshot()
        assert "pareto_front" not in snapshot["targets"]["t1"]


class TestParetoSSEEvent:
    @pytest.mark.asyncio
    async def test_pareto_update_published(self, tmp_path: Path) -> None:
        reader = AnnealStateReader(tmp_path)
        bus = FilePollingBus(reader)

        # Set up target with experiment and pareto front
        target_dir = tmp_path / "targets" / "t1"
        target_dir.mkdir(parents=True)

        front = [{"style": 0.8, "correctness": 0.6}]
        (target_dir / "pareto_front.json").write_text(json.dumps(front))

        # Write a config and experiment record
        (tmp_path / "config.toml").write_text('[targets.t1]\neval_mode = "stochastic"\n')

        # Initialize cursors
        reader.build_snapshot()

        # Add new experiment record after cursor init
        record = {
            "id": "1", "target_id": "t1", "score": 0.8,
            "baseline_score": 0.5, "outcome": "KEPT",
            "hypothesis": "test", "cost_usd": 0.01,
            "duration_seconds": 5.0,
        }
        (target_dir / "experiments.jsonl").write_text(json.dumps(record) + "\n")

        # Subscribe and manually trigger one poll cycle
        queue = await bus.subscribe()

        # Simulate one poll iteration
        targets_meta = reader.discover_targets()
        for tid in targets_meta:
            new_records = reader.read_new_experiments(tid)
            for rec in new_records:
                bus.publish("experiment_complete", {"target_id": tid, "score": rec.get("score", 0.0)})
                pareto_front = reader._read_pareto_front(tid)
                if pareto_front:
                    bus.publish("pareto_update", {
                        "target_id": tid,
                        "front": pareto_front,
                        "criterion_names": list(pareto_front[0].keys()) if pareto_front else [],
                    })

        # Check events were published
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        event_types = [e[0] for e in events]
        assert "experiment_complete" in event_types
        assert "pareto_update" in event_types

        pareto_event = next(e for e in events if e[0] == "pareto_update")
        assert pareto_event[1]["front"] == front
        assert pareto_event[1]["criterion_names"] == ["style", "correctness"]
