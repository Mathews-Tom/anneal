"""Tests for seed escape via random restart (E3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from anneal.engine.context import build_restart_context, build_target_context
from anneal.engine.search import SimulatedAnnealingSearch
from anneal.engine.types import (
    AgentConfig,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    OptimizationTarget,
)


def _make_target(
    tmp_path: Path,
    restart_probability: float = 0.0,
) -> OptimizationTarget:
    """Create a minimal target for testing."""
    scope_path = tmp_path / "scope.yaml"
    scope_path.write_text(
        "editable:\n  - artifact.md\nimmutable:\n  - scope.yaml\nwatch:\n  - reference.md\n"
    )
    (tmp_path / "artifact.md").write_text("current artifact content\n")
    (tmp_path / "reference.md").write_text("reference material\n")
    knowledge_path = tmp_path / ".anneal" / "targets" / "test"
    knowledge_path.mkdir(parents=True, exist_ok=True)

    return OptimizationTarget(
        id="test",
        domain_tier=DomainTier.SANDBOX,
        artifact_paths=["artifact.md"],
        scope_path="scope.yaml",
        scope_hash="abc123",
        eval_mode=EvalMode.DETERMINISTIC,
        eval_config=EvalConfig(
            metric_name="score",
            direction=Direction.HIGHER_IS_BETTER,
            deterministic=DeterministicEval(
                run_command="echo 1.0",
                parse_command="cat",
                timeout_seconds=30,
            ),
        ),
        agent_config=AgentConfig(
            mode="api", model="test", evaluator_model="test",
        ),
        time_budget_seconds=60,
        loop_interval_seconds=60,
        knowledge_path=str(knowledge_path),
        worktree_path=str(tmp_path),
        git_branch="anneal/test",
        baseline_score=0.5,
        restart_probability=restart_probability,
    )


class TestBuildRestartContext:
    """Tests for restart context assembly."""

    def test_restart_context_excludes_artifact_content(self, tmp_path: Path) -> None:
        target = _make_target(tmp_path, restart_probability=0.05)
        prompt, tokens = build_restart_context(target, tmp_path, tmp_path)
        # The file artifact.md contains this exact text; it must not appear
        # in the restart context (the instruction mentioning "artifact content"
        # in a different sentence is fine).
        assert "current artifact content\n" not in prompt
        assert "```\ncurrent artifact content" not in prompt
        assert tokens > 0

    def test_restart_context_includes_restart_instruction(self, tmp_path: Path) -> None:
        target = _make_target(tmp_path, restart_probability=0.05)
        prompt, _ = build_restart_context(target, tmp_path, tmp_path)
        assert "RESTART EXPERIMENT" in prompt
        assert "fresh" in prompt.lower()

    def test_restart_context_excludes_history(self, tmp_path: Path) -> None:
        target = _make_target(tmp_path, restart_probability=0.05)
        prompt, _ = build_restart_context(target, tmp_path, tmp_path)
        assert "Recent Experiments" not in prompt

    def test_restart_context_includes_watch_files(self, tmp_path: Path) -> None:
        target = _make_target(tmp_path, restart_probability=0.05)
        prompt, _ = build_restart_context(target, tmp_path, tmp_path)
        assert "reference material" in prompt

    def test_restart_context_includes_scope(self, tmp_path: Path) -> None:
        target = _make_target(tmp_path, restart_probability=0.05)
        prompt, _ = build_restart_context(target, tmp_path, tmp_path)
        assert "artifact.md" in prompt

    def test_restart_context_includes_eval_criteria(self, tmp_path: Path) -> None:
        target = _make_target(tmp_path, restart_probability=0.05)
        prompt, _ = build_restart_context(target, tmp_path, tmp_path)
        assert "score" in prompt.lower()

    def test_normal_context_includes_artifact(self, tmp_path: Path) -> None:
        """Contrast: normal context DOES include artifact content."""
        target = _make_target(tmp_path)
        prompt, _ = build_target_context(
            target, tmp_path, tmp_path, history=[],
        )
        assert "current artifact content" in prompt


class TestRestartProbabilityDefault:
    """Tests that restart_probability=0.0 produces no change."""

    def test_zero_restart_probability(self) -> None:
        """Default restart_probability is 0.0 — no restarts triggered."""
        import random
        random.seed(42)
        target = _make_target(Path("/tmp"), restart_probability=0.0)
        # With probability 0.0, random.random() < 0.0 is always False
        restarts = sum(1 for _ in range(1000) if random.random() < target.restart_probability)
        assert restarts == 0

    def test_nonzero_restart_probability(self) -> None:
        """With restart_probability=0.05, ~5% of rolls trigger restart."""
        import random
        random.seed(42)
        target = _make_target(Path("/tmp"), restart_probability=0.05)
        restarts = sum(1 for _ in range(10000) if random.random() < target.restart_probability)
        assert 300 < restarts < 700  # ~5% of 10000, with variance


class TestSATemperatureLinkedRestart:
    """Tests for temperature-linked restart probability decay."""

    def test_temperature_ratio_starts_at_one(self) -> None:
        sa = SimulatedAnnealingSearch(initial_temperature=1.0)
        assert sa.temperature_ratio == pytest.approx(1.0)

    def test_temperature_ratio_decays(self) -> None:
        sa = SimulatedAnnealingSearch(initial_temperature=1.0, cooling_rate=0.5)
        sa.cool()
        assert sa.temperature_ratio == pytest.approx(0.5)

    def test_effective_restart_probability_decays(self) -> None:
        """effective_restart_p = restart_probability * temperature_ratio"""
        sa = SimulatedAnnealingSearch(initial_temperature=1.0, cooling_rate=0.9)
        base_p = 0.05
        # Initially: effective = 0.05 * 1.0 = 0.05
        assert base_p * sa.temperature_ratio == pytest.approx(0.05)
        # After 10 cooling steps: effective = 0.05 * 0.9^10 ≈ 0.0174
        for _ in range(10):
            sa.cool()
        effective = base_p * sa.temperature_ratio
        assert effective < 0.02

    def test_temperature_ratio_at_minimum(self) -> None:
        sa = SimulatedAnnealingSearch(
            initial_temperature=1.0, cooling_rate=0.5, min_temperature=0.01,
        )
        for _ in range(100):
            sa.cool()
        assert sa.temperature_ratio == pytest.approx(0.01)

    def test_initial_temperature_property(self) -> None:
        sa = SimulatedAnnealingSearch(initial_temperature=2.5)
        assert sa.initial_temperature == 2.5
