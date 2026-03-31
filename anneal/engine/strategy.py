from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class StrategyComponent(BaseModel):
    name: str
    approach: str
    evolved_at: int = 0
    score_contribution: float | None = None
    streak_without_improvement: int = 0


class StrategyManifest(BaseModel):
    version: int = 1
    lineage: list[str] = Field(default_factory=list)
    aggregate_score_delta: float = 0.0

    hypothesis_generation: StrategyComponent = Field(
        default_factory=lambda: StrategyComponent(
            name="hypothesis_generation",
            approach="Generate mutation hypotheses based on evaluation feedback and experiment history.",
        ),
    )
    context_assembly: StrategyComponent = Field(
        default_factory=lambda: StrategyComponent(
            name="context_assembly",
            approach="Assemble relevant context from past experiments, focusing on recent failures and successes.",
        ),
    )
    mutation_style: StrategyComponent = Field(
        default_factory=lambda: StrategyComponent(
            name="mutation_style",
            approach="Make targeted, minimal changes. Prefer editing over rewriting.",
        ),
    )
    failure_analysis: StrategyComponent = Field(
        default_factory=lambda: StrategyComponent(
            name="failure_analysis",
            approach="Analyze failure patterns to avoid repeating unsuccessful approaches.",
        ),
    )
    constraints: list[str] = Field(default_factory=list)

    @property
    def components(self) -> list[StrategyComponent]:
        return [
            self.hypothesis_generation,
            self.context_assembly,
            self.mutation_style,
            self.failure_analysis,
        ]

    def weakest_component(self) -> StrategyComponent:
        return max(self.components, key=lambda c: c.streak_without_improvement)


def load_strategy(knowledge_path: Path) -> StrategyManifest | None:
    strategy_file = knowledge_path / "strategy.yaml"
    if not strategy_file.exists():
        return None
    data = yaml.safe_load(strategy_file.read_text())
    return StrategyManifest.model_validate(data)


def save_strategy(manifest: StrategyManifest, knowledge_path: Path) -> None:
    knowledge_path.mkdir(parents=True, exist_ok=True)
    strategy_file = knowledge_path / "strategy.yaml"
    strategy_file.write_text(yaml.dump(manifest.model_dump(), sort_keys=False))


def migrate_program_md(program_md_path: Path) -> StrategyManifest:
    content = program_md_path.read_text()
    manifest = StrategyManifest()
    manifest.hypothesis_generation.approach = content
    return manifest


def render_manifest_as_prompt(manifest: StrategyManifest) -> str:
    sections: list[str] = []
    for component in manifest.components:
        title = component.name.replace("_", " ").title()
        sections.append(f"## {title}\n{component.approach}")
    if manifest.constraints:
        constraint_lines = "\n".join(f"- {c}" for c in manifest.constraints)
        sections.append(f"## Constraints\n{constraint_lines}")
    return "\n\n".join(sections)
