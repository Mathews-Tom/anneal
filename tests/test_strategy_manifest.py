from __future__ import annotations

from pathlib import Path

from anneal.engine.strategy import (
    StrategyManifest,
    load_strategy,
    migrate_program_md,
    render_manifest_as_prompt,
    save_strategy,
)


class TestStrategyManifest:
    def test_strategy_manifest_default_has_four_components(self) -> None:
        manifest = StrategyManifest()
        assert len(manifest.components) == 4

    def test_strategy_manifest_roundtrip(self, tmp_path: Path) -> None:
        # Arrange
        original = StrategyManifest(
            lineage=["gen-0"],
            aggregate_score_delta=1.5,
        )
        original.hypothesis_generation.approach = "custom approach"

        # Act
        save_strategy(original, tmp_path)
        loaded = load_strategy(tmp_path)

        # Assert
        assert loaded is not None
        assert loaded.lineage == ["gen-0"]
        assert loaded.aggregate_score_delta == 1.5
        assert loaded.hypothesis_generation.approach == "custom approach"

    def test_migrate_program_md_content_in_hypothesis_generation(self, tmp_path: Path) -> None:
        # Arrange
        program_md = tmp_path / "program.md"
        program_md.write_text("Always prefer short outputs.")

        # Act
        manifest = migrate_program_md(program_md)

        # Assert
        assert manifest.hypothesis_generation.approach == "Always prefer short outputs."

    def test_weakest_component_by_streak(self) -> None:
        # Arrange
        manifest = StrategyManifest()
        manifest.hypothesis_generation.streak_without_improvement = 0
        manifest.context_assembly.streak_without_improvement = 5
        manifest.mutation_style.streak_without_improvement = 2
        manifest.failure_analysis.streak_without_improvement = 10

        # Act
        weakest = manifest.weakest_component()

        # Assert
        assert weakest.name == "failure_analysis"
        assert weakest.streak_without_improvement == 10

    def test_weakest_component_all_zero(self) -> None:
        # Arrange
        manifest = StrategyManifest()

        # Act
        weakest = manifest.weakest_component()

        # Assert
        assert weakest in manifest.components

    def test_load_strategy_returns_none_when_missing(self, tmp_path: Path) -> None:
        result = load_strategy(tmp_path)
        assert result is None

    def test_save_strategy_creates_yaml_file(self, tmp_path: Path) -> None:
        # Arrange
        manifest = StrategyManifest()

        # Act
        save_strategy(manifest, tmp_path)

        # Assert
        assert (tmp_path / "strategy.yaml").exists()

    def test_render_manifest_as_prompt_includes_all_components(self) -> None:
        # Arrange
        manifest = StrategyManifest()

        # Act
        rendered = render_manifest_as_prompt(manifest)

        # Assert
        for component in manifest.components:
            assert component.approach in rendered

    def test_render_manifest_as_prompt_includes_constraints(self) -> None:
        # Arrange
        manifest = StrategyManifest(constraints=["never change X"])

        # Act
        rendered = render_manifest_as_prompt(manifest)

        # Assert
        assert "## Constraints" in rendered
        assert "never change X" in rendered

    def test_constraints_not_in_components(self) -> None:
        # Arrange
        manifest = StrategyManifest(constraints=["rule one", "rule two"])

        # Act
        component_names = [c.name for c in manifest.components]

        # Assert
        assert "constraints" not in component_names
        assert len(manifest.components) == 4
