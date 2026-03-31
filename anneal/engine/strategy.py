from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from anneal.engine.types import ExperimentRecord, Outcome


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


def _summarize_criterion_performance(records: list[ExperimentRecord]) -> str:
    """Build per-criterion PASS/FAIL summary from recent experiment records."""
    if not records:
        return "No experiments available."

    criterion_stats: dict[str, dict[str, int]] = {}
    for r in records:
        if r.per_criterion_scores:
            for name, score in r.per_criterion_scores.items():
                if name not in criterion_stats:
                    criterion_stats[name] = {"pass": 0, "fail": 0}
                if score >= 0.5:
                    criterion_stats[name]["pass"] += 1
                else:
                    criterion_stats[name]["fail"] += 1

    if not criterion_stats:
        return "No per-criterion data available."

    lines: list[str] = []
    for name, stats in sorted(criterion_stats.items(), key=lambda kv: kv[1]["fail"], reverse=True):
        total = stats["pass"] + stats["fail"]
        lines.append(f"- {name}: {stats['pass']}/{total} passed ({stats['fail']} failures)")
    return "\n".join(lines)


def evolve_weakest_component(
    manifest: StrategyManifest,
    records: list[ExperimentRecord],
) -> tuple[StrategyComponent, str]:
    """Identify and prepare evolution prompt for weakest strategy component.

    Returns (target_component, evolution_prompt).
    """
    target = manifest.weakest_component()

    relevant_records = [
        r for r in records
        if r.outcome in (Outcome.KEPT, Outcome.DISCARDED)
    ]

    kept_count = sum(1 for r in relevant_records if r.outcome is Outcome.KEPT)
    scores = [r.score for r in relevant_records] if relevant_records else []

    prompt = (
        f"## Component Evolution: {target.name}\n\n"
        f"### Current approach\n{target.approach}\n\n"
        f"### Performance since last evolution\n"
        f"- Experiments: {len(relevant_records)}\n"
        f"- Kept: {kept_count}\n"
    )
    if scores:
        prompt += f"- Score range: [{min(scores):.4f}, {max(scores):.4f}]\n\n"
    else:
        prompt += "- Score range: N/A\n\n"

    prompt += (
        f"### Per-criterion feedback from relevant experiments\n"
        + _summarize_criterion_performance(relevant_records)
        + "\n\n"
        f"Revise ONLY the '{target.name}' component. "
        f"Keep the revision concise (2-5 sentences). "
        f"Explain what you're changing and why."
    )

    return target, prompt


def render_manifest_as_prompt(manifest: StrategyManifest) -> str:
    sections: list[str] = []
    for component in manifest.components:
        title = component.name.replace("_", " ").title()
        sections.append(f"## {title}\n{component.approach}")
    if manifest.constraints:
        constraint_lines = "\n".join(f"- {c}" for c in manifest.constraints)
        sections.append(f"## Constraints\n{constraint_lines}")
    return "\n\n".join(sections)
