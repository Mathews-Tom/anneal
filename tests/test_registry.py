"""Tests for anneal.engine.registry — init_project, Registry CRUD, persistence, scope validation."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import pytest_asyncio

from anneal.engine.registry import Registry, RegistryError, init_project
from anneal.engine.types import (
    AgentConfig,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    OptimizationTarget,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def initialized_repo(git_repo: Path) -> Path:
    """git_repo with init_project() already called."""
    await init_project(git_repo)
    return git_repo


@pytest.fixture
def sample_scope_yaml(git_repo: Path) -> Path:
    """Write a valid scope.yaml into the repo root and return its path."""
    scope_path = git_repo / "scope.yaml"
    scope_path.write_text(
        "editable:\n"
        "  - SKILL.md\n"
        "immutable:\n"
        "  - scope.yaml\n"
        "  - metrics.yaml\n",
        encoding="utf-8",
    )
    return scope_path


def _make_target(
    target_id: str = "test_target",
    scope_path: str = "scope.yaml",
    eval_mode: EvalMode = EvalMode.DETERMINISTIC,
) -> OptimizationTarget:
    """Factory for a valid OptimizationTarget with sensible defaults."""
    return OptimizationTarget(
        id=target_id,
        domain_tier=DomainTier.SANDBOX,
        artifact_paths=["SKILL.md"],
        scope_path=scope_path,
        scope_hash="",
        eval_mode=eval_mode,
        eval_config=EvalConfig(
            metric_name="accuracy",
            direction=Direction.HIGHER_IS_BETTER,
            min_improvement_threshold=0.01,
            deterministic=DeterministicEval(
                run_command="python eval.py",
                parse_command="python parse.py",
                timeout_seconds=60,
            ),
        ),
        agent_config=AgentConfig(
            mode="claude_code",
            model="gpt-4.1",
            evaluator_model="gpt-4.1-mini",
            max_budget_usd=0.10,
        ),
        time_budget_seconds=3600,
        loop_interval_seconds=120,
        knowledge_path="knowledge/",
        worktree_path="",
        git_branch="",
        baseline_score=0.75,
        baseline_raw_scores=[0.7, 0.8, 0.75],
        max_consecutive_failures=5,
    )


# ---------------------------------------------------------------------------
# init_project
# ---------------------------------------------------------------------------


class TestInitProject:
    @pytest.mark.asyncio
    async def test_creates_directory_structure(self, git_repo: Path) -> None:
        await init_project(git_repo)

        assert (git_repo / ".anneal").is_dir()
        assert (git_repo / ".anneal" / "config.toml").is_file()
        assert (git_repo / ".anneal" / "targets").is_dir()
        assert (git_repo / ".anneal" / "templates").is_dir()
        assert (git_repo / ".anneal" / "worktrees").is_dir()

    @pytest.mark.asyncio
    async def test_default_config_has_anneal_section(self, git_repo: Path) -> None:
        await init_project(git_repo)

        config_text = (git_repo / ".anneal" / "config.toml").read_text(encoding="utf-8")
        data = tomllib.loads(config_text)
        assert "anneal" in data
        assert "version" in data["anneal"]

    @pytest.mark.asyncio
    async def test_adds_worktrees_to_gitignore(self, git_repo: Path) -> None:
        await init_project(git_repo)

        gitignore = (git_repo / ".gitignore").read_text(encoding="utf-8")
        assert ".anneal/" in gitignore

    @pytest.mark.asyncio
    async def test_appends_to_existing_gitignore(self, git_repo: Path) -> None:
        gitignore_path = git_repo / ".gitignore"
        gitignore_path.write_text("*.pyc\n", encoding="utf-8")

        await init_project(git_repo)

        gitignore = gitignore_path.read_text(encoding="utf-8")
        assert "*.pyc" in gitignore
        assert ".anneal/" in gitignore

    @pytest.mark.asyncio
    async def test_raises_on_reinit(self, initialized_repo: Path) -> None:
        with pytest.raises(RegistryError, match="already initialized"):
            await init_project(initialized_repo)

    @pytest.mark.asyncio
    async def test_raises_if_not_git_repo(self, tmp_path: Path) -> None:
        with pytest.raises(RegistryError, match="Not a git repository"):
            await init_project(tmp_path)


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------


class TestRegistryCRUD:
    @pytest.mark.asyncio
    async def test_register_target_stores_and_creates_worktree(
        self, initialized_repo: Path, sample_scope_yaml: Path
    ) -> None:
        registry = Registry(initialized_repo)
        target = _make_target()

        await registry.register_target(target)

        stored = registry.get_target("test_target")
        assert stored.id == "test_target"
        assert stored.worktree_path != ""
        assert stored.git_branch == "anneal/test_target"
        assert stored.scope_hash != ""

    @pytest.mark.asyncio
    async def test_get_target_raises_for_unknown(self, initialized_repo: Path) -> None:
        registry = Registry(initialized_repo)

        with pytest.raises(RegistryError, match="Target not found"):
            registry.get_target("nonexistent")

    @pytest.mark.asyncio
    async def test_all_targets_returns_registered(
        self, initialized_repo: Path, sample_scope_yaml: Path
    ) -> None:
        registry = Registry(initialized_repo)
        t1 = _make_target(target_id="alpha")
        t2 = _make_target(target_id="beta")

        await registry.register_target(t1)
        await registry.register_target(t2)

        all_targets = registry.all_targets()
        ids = {t.id for t in all_targets}
        assert ids == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_update_target_persists(
        self, initialized_repo: Path, sample_scope_yaml: Path
    ) -> None:
        registry = Registry(initialized_repo)
        target = _make_target()
        await registry.register_target(target)

        target.baseline_score = 0.95
        registry.update_target(target)

        reloaded = Registry(initialized_repo)
        assert reloaded.get_target("test_target").baseline_score == 0.95

    @pytest.mark.asyncio
    async def test_deregister_target_removes(
        self, initialized_repo: Path, sample_scope_yaml: Path
    ) -> None:
        registry = Registry(initialized_repo)
        target = _make_target()
        await registry.register_target(target)

        await registry.deregister_target("test_target")

        assert registry.all_targets() == []
        with pytest.raises(RegistryError, match="Target not found"):
            registry.get_target("test_target")

    @pytest.mark.asyncio
    async def test_deregister_unknown_raises(self, initialized_repo: Path) -> None:
        registry = Registry(initialized_repo)

        with pytest.raises(RegistryError, match="Target not found"):
            await registry.deregister_target("ghost")

    @pytest.mark.asyncio
    async def test_register_duplicate_raises(
        self, initialized_repo: Path, sample_scope_yaml: Path
    ) -> None:
        registry = Registry(initialized_repo)
        target = _make_target()
        await registry.register_target(target)

        with pytest.raises(RegistryError, match="already registered"):
            await registry.register_target(_make_target())


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    @pytest.mark.asyncio
    async def test_config_on_disk_after_register(
        self, initialized_repo: Path, sample_scope_yaml: Path
    ) -> None:
        registry = Registry(initialized_repo)
        target = _make_target()
        await registry.register_target(target)

        raw = (initialized_repo / ".anneal" / "config.toml").read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))

        assert "targets" in data
        assert "test_target" in data["targets"]
        assert data["targets"]["test_target"]["id"] == "test_target"

    @pytest.mark.asyncio
    async def test_save_load_roundtrip(
        self, initialized_repo: Path, sample_scope_yaml: Path
    ) -> None:
        registry = Registry(initialized_repo)
        target = _make_target()
        await registry.register_target(target)

        registry.save()

        reloaded = Registry(initialized_repo)
        rt = reloaded.get_target("test_target")

        assert rt.id == target.id
        assert rt.domain_tier == target.domain_tier
        assert rt.eval_mode == target.eval_mode
        assert rt.baseline_score == target.baseline_score
        assert rt.baseline_raw_scores == target.baseline_raw_scores
        assert rt.artifact_paths == target.artifact_paths
        assert rt.scope_path == target.scope_path
        assert rt.eval_config.metric_name == target.eval_config.metric_name
        assert rt.eval_config.direction == target.eval_config.direction
        assert rt.agent_config.model == target.agent_config.model
        assert rt.agent_config.mode == target.agent_config.mode
        assert rt.max_consecutive_failures == target.max_consecutive_failures

    @pytest.mark.asyncio
    async def test_config_is_valid_toml(
        self, initialized_repo: Path, sample_scope_yaml: Path
    ) -> None:
        registry = Registry(initialized_repo)
        await registry.register_target(_make_target())

        raw = (initialized_repo / ".anneal" / "config.toml").read_bytes()
        # Must not raise
        data = tomllib.loads(raw.decode("utf-8"))
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Scope validation during registration
# ---------------------------------------------------------------------------


class TestScopeValidation:
    @pytest.mark.asyncio
    async def test_invalid_scope_missing_immutables_raises(
        self, initialized_repo: Path
    ) -> None:
        # Write scope.yaml missing required immutable entries
        scope_path = initialized_repo / "scope.yaml"
        scope_path.write_text(
            "editable:\n"
            "  - SKILL.md\n"
            "immutable:\n"
            "  - other.yaml\n",
            encoding="utf-8",
        )

        registry = Registry(initialized_repo)
        target = _make_target()

        with pytest.raises(RegistryError, match="Scope validation failed"):
            await registry.register_target(target)

    @pytest.mark.asyncio
    async def test_missing_scope_file_raises(self, initialized_repo: Path) -> None:
        registry = Registry(initialized_repo)
        target = _make_target(scope_path="nonexistent_scope.yaml")

        with pytest.raises(RegistryError, match="Scope load failed"):
            await registry.register_target(target)

    @pytest.mark.asyncio
    async def test_valid_scope_succeeds(
        self, initialized_repo: Path, sample_scope_yaml: Path
    ) -> None:
        registry = Registry(initialized_repo)
        target = _make_target()

        await registry.register_target(target)

        assert registry.get_target("test_target").id == "test_target"
