"""Tests for anneal.engine.scope — scope parsing, validation, hashing, and enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio  # noqa: F401 (ensures pytest-asyncio is importable)

from anneal.engine.scope import (
    ScopeError,
    _is_path_editable,
    compute_scope_hash,
    enforce_scope,
    load_scope,
    validate_scope,
    verify_scope_hash,
)
from anneal.engine.types import (
    EvalMode,
    OptimizationTarget,
    ScopeConfig,
    ScopeViolationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_scope_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid scope.yaml and return its path."""
    p = tmp_path / "scope.yaml"
    p.write_text(
        "editable:\n"
        "  - src/\n"
        "  - SKILL.md\n"
        "immutable:\n"
        "  - scope.yaml\n"
        "  - metrics.yaml\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def full_scope_yaml(tmp_path: Path) -> Path:
    """scope.yaml with all optional fields populated."""
    p = tmp_path / "scope.yaml"
    p.write_text(
        "editable:\n"
        "  - src/\n"
        "immutable:\n"
        "  - scope.yaml\n"
        "  - metrics.yaml\n"
        "watch:\n"
        "  - logs/\n"
        "allowed_deletions:\n"
        "  - src/old.py\n"
        "constraints:\n"
        "  - rule: no binary files\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def base_scope() -> ScopeConfig:
    """Minimal valid ScopeConfig for validation tests."""
    return ScopeConfig(
        editable=["src/", "SKILL.md"],
        immutable=["scope.yaml", "metrics.yaml"],
    )


# ---------------------------------------------------------------------------
# _is_path_editable
# ---------------------------------------------------------------------------


class TestIsPathEditable:
    def test_exact_match(self) -> None:
        assert _is_path_editable("SKILL.md", ["SKILL.md"]) is True

    def test_directory_prefix(self) -> None:
        assert _is_path_editable("src/components/Button.tsx", ["src/components/"]) is True

    def test_glob_pattern(self) -> None:
        assert _is_path_editable("src/utils/helper.ts", ["src/**"]) is True

    def test_non_match(self) -> None:
        assert _is_path_editable("scope.yaml", ["SKILL.md"]) is False

    def test_multiple_patterns(self) -> None:
        patterns = ["docs/", "SKILL.md", "src/**"]
        assert _is_path_editable("src/utils/helper.ts", patterns) is True

    def test_nested_directory(self) -> None:
        assert _is_path_editable("a/b/c/d.txt", ["a/b/"]) is True


# ---------------------------------------------------------------------------
# load_scope
# ---------------------------------------------------------------------------


class TestLoadScope:
    def test_valid_scope_parses(self, valid_scope_yaml: Path) -> None:
        cfg = load_scope(valid_scope_yaml)
        assert cfg.editable == ["src/", "SKILL.md"]
        assert cfg.immutable == ["scope.yaml", "metrics.yaml"]

    def test_missing_editable_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "scope.yaml"
        p.write_text(
            "immutable:\n  - scope.yaml\n  - metrics.yaml\n",
            encoding="utf-8",
        )
        with pytest.raises(ScopeError, match="editable"):
            load_scope(p)

    def test_missing_immutable_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "scope.yaml"
        p.write_text(
            "editable:\n  - src/\n",
            encoding="utf-8",
        )
        with pytest.raises(ScopeError, match="immutable"):
            load_scope(p)

    def test_empty_editable_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "scope.yaml"
        p.write_text(
            "editable: []\nimmutable:\n  - scope.yaml\n",
            encoding="utf-8",
        )
        with pytest.raises(ScopeError, match="non-empty"):
            load_scope(p)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "scope.yaml"
        p.write_text("{{invalid yaml: [", encoding="utf-8")
        with pytest.raises(ScopeError, match="YAML parse error"):
            load_scope(p)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.yaml"
        with pytest.raises(ScopeError, match="Cannot read"):
            load_scope(p)

    def test_optional_fields_default_empty(self, valid_scope_yaml: Path) -> None:
        cfg = load_scope(valid_scope_yaml)
        assert cfg.watch == []
        assert cfg.allowed_deletions == []
        assert cfg.constraints == []

    def test_optional_fields_populated(self, full_scope_yaml: Path) -> None:
        cfg = load_scope(full_scope_yaml)
        assert cfg.watch == ["logs/"]
        assert cfg.allowed_deletions == ["src/old.py"]
        assert cfg.constraints == [{"rule": "no binary files"}]


# ---------------------------------------------------------------------------
# validate_scope
# ---------------------------------------------------------------------------


def _make_sibling(
    target_id: str = "sibling",
    scope_path: str = "targets/sibling/scope.yaml",
) -> OptimizationTarget:
    """Build a minimal OptimizationTarget stub for cross-target tests."""
    from anneal.engine.types import AgentConfig, Direction, DeterministicEval, EvalConfig

    return OptimizationTarget(
        id=target_id,
        domain_tier=__import__("anneal.engine.types", fromlist=["DomainTier"]).DomainTier.SANDBOX,
        artifact_paths=["targets/sibling/artifact.py"],
        scope_path=scope_path,
        scope_hash="abc123",
        eval_mode=EvalMode.DETERMINISTIC,
        eval_config=EvalConfig(
            metric_name="accuracy",
            direction=Direction.HIGHER_IS_BETTER,
            deterministic=DeterministicEval(
                run_command="pytest",
                parse_command="parse",
                timeout_seconds=30,
            ),
        ),
        agent_config=AgentConfig(mode="api", model="gpt-4.1", evaluator_model="gpt-4.1-mini"),
        time_budget_seconds=300,
        loop_interval_seconds=10,
        knowledge_path="targets/sibling/knowledge/",
        worktree_path="/tmp/wt",
        git_branch="anneal/sibling",
        baseline_score=0.5,
    )


class TestValidateScope:
    def test_valid_scope_no_errors(self, base_scope: ScopeConfig) -> None:
        errors = validate_scope(base_scope, EvalMode.DETERMINISTIC)
        assert errors == []

    def test_missing_scope_yaml_in_immutable(self) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["metrics.yaml"],
        )
        errors = validate_scope(scope, EvalMode.DETERMINISTIC)
        assert any("scope.yaml" in e for e in errors)

    def test_missing_metrics_yaml_in_immutable(self) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml"],
        )
        errors = validate_scope(scope, EvalMode.DETERMINISTIC)
        assert any("metrics.yaml" in e for e in errors)

    def test_stochastic_missing_eval_criteria(self) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        errors = validate_scope(scope, EvalMode.STOCHASTIC)
        assert any("eval_criteria.toml" in e for e in errors)

    def test_deterministic_does_not_require_eval_criteria(self, base_scope: ScopeConfig) -> None:
        errors = validate_scope(base_scope, EvalMode.DETERMINISTIC)
        assert not any("eval_criteria.toml" in e for e in errors)

    def test_editable_immutable_overlap(self) -> None:
        scope = ScopeConfig(
            editable=["src/", "scope.yaml"],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        errors = validate_scope(scope, EvalMode.DETERMINISTIC)
        assert any("both editable and immutable" in e for e in errors)

    def test_cross_target_scope_path_in_editable(self, base_scope: ScopeConfig) -> None:
        sibling = _make_sibling()
        scope = ScopeConfig(
            editable=["src/", sibling.scope_path],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        errors = validate_scope(scope, EvalMode.DETERMINISTIC, sibling_targets=[sibling])
        assert any("sibling target config" in e for e in errors)


# ---------------------------------------------------------------------------
# Hash computation and verification
# ---------------------------------------------------------------------------


class TestHashing:
    def test_compute_scope_hash_consistent(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("hello world", encoding="utf-8")
        h1 = compute_scope_hash(p)
        h2 = compute_scope_hash(p)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_verify_scope_hash_matches(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("content", encoding="utf-8")
        h = compute_scope_hash(p)
        assert verify_scope_hash(p, h) is True

    def test_verify_scope_hash_mismatch_after_modification(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("original", encoding="utf-8")
        h = compute_scope_hash(p)
        p.write_text("modified", encoding="utf-8")
        assert verify_scope_hash(p, h) is False


# ---------------------------------------------------------------------------
# enforce_scope
# ---------------------------------------------------------------------------


class TestEnforceScope:
    @pytest.mark.asyncio
    async def test_modified_editable_file_valid(self, tmp_path: Path) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        git_status = [("M", "src/main.py")]
        result = await enforce_scope(tmp_path, scope, git_status)
        assert result.has_violations is False
        assert "src/main.py" in result.valid_paths

    @pytest.mark.asyncio
    async def test_modified_non_editable_violated(self, tmp_path: Path) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        git_status = [("M", "scope.yaml")]
        result = await enforce_scope(tmp_path, scope, git_status)
        assert result.has_violations is True
        assert "scope.yaml" in result.violated_paths

    @pytest.mark.asyncio
    async def test_new_untracked_in_editable_valid(self, tmp_path: Path) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        git_status = [("??", "src/new_file.py")]
        result = await enforce_scope(tmp_path, scope, git_status)
        assert result.has_violations is False
        assert "src/new_file.py" in result.valid_paths

    @pytest.mark.asyncio
    async def test_new_untracked_outside_editable_violated(self, tmp_path: Path) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        git_status = [("??", "rogue_file.txt")]
        result = await enforce_scope(tmp_path, scope, git_status)
        assert result.has_violations is True
        assert "rogue_file.txt" in result.violated_paths

    @pytest.mark.asyncio
    async def test_deleted_in_allowed_deletions_valid(self, tmp_path: Path) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml", "metrics.yaml"],
            allowed_deletions=["src/old.py"],
        )
        git_status = [("D", "src/old.py")]
        result = await enforce_scope(tmp_path, scope, git_status)
        assert result.has_violations is False
        assert "src/old.py" in result.valid_paths

    @pytest.mark.asyncio
    async def test_deleted_not_in_allowed_deletions_violated(self, tmp_path: Path) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        git_status = [("D", "src/main.py")]
        result = await enforce_scope(tmp_path, scope, git_status)
        assert result.has_violations is True
        assert "src/main.py" in result.violated_paths

    @pytest.mark.asyncio
    async def test_partial_violation(self, tmp_path: Path) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        git_status = [
            ("M", "src/main.py"),       # valid
            ("M", "scope.yaml"),         # violated
        ]
        result = await enforce_scope(tmp_path, scope, git_status)
        assert result.has_violations is True
        assert result.all_blocked is False

    @pytest.mark.asyncio
    async def test_all_blocked(self, tmp_path: Path) -> None:
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        git_status = [
            ("M", "scope.yaml"),
            ("M", "metrics.yaml"),
        ]
        result = await enforce_scope(tmp_path, scope, git_status)
        assert result.has_violations is True
        assert result.all_blocked is True
        assert result.valid_paths == []

    @pytest.mark.asyncio
    async def test_partial_violation_preserves_valid_edits(self, tmp_path: Path) -> None:
        """Regression: when scope has violations, valid_paths still contains the good files."""
        scope = ScopeConfig(
            editable=["src/", "SKILL.md"],
            immutable=["scope.yaml", "metrics.yaml"],
        )
        git_status = [
            ("M", "src/engine.py"),      # valid
            ("A", "src/helper.py"),       # valid
            ("M", "metrics.yaml"),        # violated
            ("??", "SKILL.md"),           # valid (exact match)
        ]
        result = await enforce_scope(tmp_path, scope, git_status)
        assert result.has_violations is True
        assert result.all_blocked is False
        assert set(result.valid_paths) == {"src/engine.py", "src/helper.py", "SKILL.md"}
        assert result.violated_paths == ["metrics.yaml"]

    @pytest.mark.asyncio
    async def test_agent_cannot_commit_eval_criteria_toml(self, tmp_path: Path) -> None:
        """eval_criteria.toml is immutable — modifying it is a violation."""
        scope = ScopeConfig(
            editable=["src/"],
            immutable=["scope.yaml", "metrics.yaml", "eval_criteria.toml"],
        )
        git_status = [("M", "eval_criteria.toml")]
        result = await enforce_scope(tmp_path, scope, git_status)
        assert result.has_violations is True
        assert "eval_criteria.toml" in result.violated_paths
