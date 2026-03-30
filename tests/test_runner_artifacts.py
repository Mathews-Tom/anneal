"""Tests for artifact reading fail-fast guards in ExperimentRunner._read_artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from anneal.engine.runner import ExperimentRunner
from anneal.engine.types import ArtifactError


# ---------------------------------------------------------------------------
# _read_artifacts is a @staticmethod — call directly
# ---------------------------------------------------------------------------


class TestReadArtifacts:
    def test_all_present_returns_concatenated_content(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
        (tmp_path / "b.md").write_text("beta", encoding="utf-8")

        result = ExperimentRunner._read_artifacts(tmp_path, ["a.md", "b.md"])

        assert "### a.md" in result
        assert "alpha" in result
        assert "### b.md" in result
        assert "beta" in result

    def test_all_missing_raises_artifact_error(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError, match="All artifact files missing"):
            ExperimentRunner._read_artifacts(tmp_path, ["missing.md"])

    def test_all_missing_includes_paths_in_message(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError, match="missing.md"):
            ExperimentRunner._read_artifacts(tmp_path, ["missing.md", "also_gone.md"])

    def test_partial_missing_warns_but_returns_content(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / "exists.md").write_text("content", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            result = ExperimentRunner._read_artifacts(
                tmp_path, ["exists.md", "missing.md"]
            )

        assert "### exists.md" in result
        assert "content" in result
        assert "missing.md" in caplog.text

    def test_empty_artifact_paths_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError):
            ExperimentRunner._read_artifacts(tmp_path, [])

    def test_nested_paths_resolved_correctly(self, tmp_path: Path) -> None:
        nested = tmp_path / "examples" / "recon"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("skill content", encoding="utf-8")

        result = ExperimentRunner._read_artifacts(
            tmp_path, ["examples/recon/SKILL.md"]
        )

        assert "skill content" in result
        assert "### examples/recon/SKILL.md" in result
