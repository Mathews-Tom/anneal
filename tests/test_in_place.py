"""Tests for in-place optimization mode (no git worktree)."""

from __future__ import annotations

from pathlib import Path

import pytest

from anneal.engine.environment import FileBackupEnvironment
from anneal.engine.scope import validate_scope
from anneal.engine.types import (
    AgentConfig,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    OptimizationTarget,
    ScopeConfig,
)


# ---------------------------------------------------------------------------
# FileBackupEnvironment
# ---------------------------------------------------------------------------


class TestFileBackupEnvironment:
    @pytest.mark.asyncio
    async def test_backup_creates_and_restores(self, tmp_path: Path) -> None:
        backup_dir = tmp_path / "backups"
        env = FileBackupEnvironment(backup_dir)

        artifact = tmp_path / "artifact.md"
        artifact.write_text("original content", encoding="utf-8")

        backup_id = await env.backup(["artifact.md"], tmp_path)
        assert (backup_dir / backup_id / "artifact.md").exists()

        # Mutate the artifact
        artifact.write_text("mutated content", encoding="utf-8")
        assert artifact.read_text() == "mutated content"

        # Restore
        await env.restore(backup_id, ["artifact.md"], tmp_path)
        assert artifact.read_text() == "original content"

    @pytest.mark.asyncio
    async def test_cleanup_removes_backup(self, tmp_path: Path) -> None:
        backup_dir = tmp_path / "backups"
        env = FileBackupEnvironment(backup_dir)

        (tmp_path / "a.md").write_text("data", encoding="utf-8")
        backup_id = await env.backup(["a.md"], tmp_path)
        assert (backup_dir / backup_id).exists()

        await env.cleanup(backup_id)
        assert not (backup_dir / backup_id).exists()

    @pytest.mark.asyncio
    async def test_backup_preserves_nested_paths(self, tmp_path: Path) -> None:
        backup_dir = tmp_path / "backups"
        env = FileBackupEnvironment(backup_dir)

        nested = tmp_path / "examples" / "recon"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("nested content", encoding="utf-8")

        backup_id = await env.backup(["examples/recon/SKILL.md"], tmp_path)
        assert (backup_dir / backup_id / "examples" / "recon" / "SKILL.md").exists()

        # Mutate and restore
        (nested / "SKILL.md").write_text("changed", encoding="utf-8")
        await env.restore(backup_id, ["examples/recon/SKILL.md"], tmp_path)
        assert (nested / "SKILL.md").read_text() == "nested content"

    @pytest.mark.asyncio
    async def test_restore_nonexistent_backup_raises(self, tmp_path: Path) -> None:
        env = FileBackupEnvironment(tmp_path / "backups")
        with pytest.raises(FileNotFoundError, match="Backup not found"):
            await env.restore("nonexistent", ["a.md"], tmp_path)

    @pytest.mark.asyncio
    async def test_backup_skips_missing_source_files(self, tmp_path: Path) -> None:
        """Backup doesn't fail if an artifact file doesn't exist yet."""
        env = FileBackupEnvironment(tmp_path / "backups")
        backup_id = await env.backup(["nonexistent.md"], tmp_path)
        # Backup succeeds but is empty
        assert not (tmp_path / "backups" / backup_id / "nonexistent.md").exists()


# ---------------------------------------------------------------------------
# Scope validation relaxation for in-place
# ---------------------------------------------------------------------------


class TestInPlaceScopeValidation:
    def test_in_place_does_not_require_metrics_yaml(self) -> None:
        scope = ScopeConfig(
            editable=["SKILL.md"],
            immutable=["scope.yaml"],
        )
        errors = validate_scope(scope, EvalMode.DETERMINISTIC, in_place=True)
        assert not any("metrics.yaml" in e for e in errors)

    def test_standard_mode_requires_metrics_yaml(self) -> None:
        scope = ScopeConfig(
            editable=["SKILL.md"],
            immutable=["scope.yaml"],
        )
        errors = validate_scope(scope, EvalMode.DETERMINISTIC, in_place=False)
        assert any("metrics.yaml" in e for e in errors)

    def test_in_place_still_requires_scope_yaml_immutable(self) -> None:
        scope = ScopeConfig(
            editable=["SKILL.md"],
            immutable=[],
        )
        errors = validate_scope(scope, EvalMode.DETERMINISTIC, in_place=True)
        assert any("scope.yaml" in e for e in errors)


# ---------------------------------------------------------------------------
# OptimizationTarget in_place field
# ---------------------------------------------------------------------------


class TestInPlaceTargetField:
    def test_default_is_false(self) -> None:
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
            knowledge_path="k", worktree_path="w", git_branch="b",
            baseline_score=0.0,
        )
        assert target.in_place is False

    def test_in_place_can_be_set(self) -> None:
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
            knowledge_path="k", worktree_path="w", git_branch="b",
            baseline_score=0.0, in_place=True,
        )
        assert target.in_place is True
