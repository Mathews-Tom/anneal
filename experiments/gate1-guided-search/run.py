"""Gate 1: LLM-guided mutation vs random vs Bayesian search.

Validates the core claim: LLM-guided mutation outperforms random search
on a SKILL.md target. Uses cross-condition learning pool (RANDOM →
BAYESIAN → GUIDED execution order per round).

Usage:
    PYTHONPATH=. uv run python experiments/gate1-guided-search/run.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import tomllib
from pathlib import Path

import openai

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from anneal.engine.eval import StochasticEvaluator, _make_client
from anneal.engine.learning_pool import (
    LearningPool,
    LearningScope,
    extract_learning,
)
from anneal.engine.types import (
    AgentConfig,
    BinaryCriterion,
    EvalResult,
    ExperimentRecord,
    Outcome,
    StochasticEval,
)
from experiments._harness.base import ExperimentHarness, console, load_config, make_progress, update_progress
from experiments._harness.types import ConditionConfig, ExperimentConfig, ResultRecord

EXPERIMENT_DIR = Path(__file__).resolve().parent

# Execution order per round: random and bayesian first so their learnings
# can be injected into the guided agent's context.
EXECUTION_ORDER = ["random", "bayesian", "guided"]


def _load_eval_criteria(
    repo_root: Path, criteria_path: str,
) -> tuple[list[BinaryCriterion], list[str], dict[str, object]]:
    path = repo_root / criteria_path
    data: dict[str, object] = tomllib.loads(path.read_text(encoding="utf-8"))
    criteria = [
        BinaryCriterion(name=c["name"], question=c["question"])
        for c in (data.get("criteria") or [])  # type: ignore[union-attr]
    ]
    test_prompts = [
        tp["prompt"] for tp in (data.get("test_prompts") or [])  # type: ignore[union-attr]
    ]
    generation: dict[str, object] = data.get("generation") or {}  # type: ignore[assignment]
    return criteria, test_prompts, generation


def _build_stochastic_config(
    criteria: list[BinaryCriterion],
    test_prompts: list[str],
    gen_config: dict[str, object],
) -> StochasticEval:
    agent_section = gen_config.get("agent") or {}
    assert isinstance(agent_section, dict)

    gen_agent = AgentConfig(
        mode="api",
        model=str(agent_section.get("model", "gemini-2.5-flash")),
        evaluator_model="gpt-4.1",
        max_budget_usd=0.02,
        temperature=float(agent_section.get("temperature", 0.7)),
    )

    return StochasticEval(
        sample_count=len(test_prompts),
        criteria=criteria,
        test_prompts=test_prompts,
        generation_prompt_template=str(gen_config.get("prompt_template", "")),
        output_format=str(gen_config.get("output_format", "text")),
        confidence_level=0.95,
        generation_agent_config=gen_agent,
    )


async def _mutate_artifact(
    client: openai.AsyncOpenAI,
    model: str,
    artifact_content: str,
    condition: str,
    history_text: str,
    cross_condition_insights: str,
) -> tuple[str, str, float]:
    """Generate a mutated artifact via API call.

    Returns (mutated_content, hypothesis, cost_usd).
    """
    if condition == "random":
        system_prompt = (
            "You modify text artifacts. Make a single random change. "
            "Do NOT use any context or history. Just change something."
        )
        user_prompt = (
            f"Make one random modification to this artifact. "
            f"Output the full modified artifact.\n\n"
            f"## Hypothesis\nState what you changed in one sentence.\n\n"
            f"---\n{artifact_content}"
        )
    elif condition == "bayesian":
        system_prompt = (
            "You optimize text artifacts using systematic parameter search. "
            "Focus on one parameter at a time: structure, wording, constraints, "
            "examples. Explore the parameter space methodically."
        )
        user_prompt = (
            f"Optimize this artifact by adjusting one parameter. "
            f"Output the full modified artifact.\n\n"
            f"## Hypothesis\nState what parameter you changed and why.\n\n"
            f"---\n{artifact_content}"
        )
    else:  # guided
        system_prompt = (
            "You optimize text artifacts using experiment history and "
            "cross-condition insights. Analyze what worked and what didn't, "
            "then make an informed mutation."
        )
        parts = [
            f"Improve this artifact based on the experiment history below. "
            f"Output the full modified artifact.\n\n"
            f"## Hypothesis\nState your hypothesis in one sentence.\n",
        ]
        if history_text:
            parts.append(f"\n## Previous Results\n{history_text}")
        if cross_condition_insights:
            parts.append(f"\n{cross_condition_insights}")
        parts.append(f"\n---\n{artifact_content}")
        user_prompt = "\n".join(parts)

    response = await client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    output = response.choices[0].message.content or ""
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0

    # Extract hypothesis (first line after "## Hypothesis")
    hypothesis = ""
    lines = output.split("\n")
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("## hypothesis"):
            # Take the next non-empty line
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip():
                    hypothesis = lines[j].strip()
                    break
            break
    if not hypothesis:
        hypothesis = f"{condition} mutation"

    # Approximate cost from tokens
    cost = (input_tokens * 0.0000015 + output_tokens * 0.000002)

    return output, hypothesis, cost


class GuidedSearchExperiment(ExperimentHarness):
    """A/B/C comparison: guided vs random vs bayesian."""

    def __init__(self, config: ExperimentConfig, experiment_dir: Path) -> None:
        super().__init__(config, experiment_dir)
        self._repo_root = PROJECT_ROOT
        self._evaluator = StochasticEvaluator()
        self._learning_pool = LearningPool()
        self._midpoint_check = 25

        # Load midpoint from config
        config_path = experiment_dir / "config.toml"
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self._midpoint_check = raw.get("experiment", {}).get("midpoint_check", 25)

    async def run_condition(self, condition: ConditionConfig) -> list[ResultRecord]:
        """Not used directly — run_all handles interleaved execution."""
        raise NotImplementedError("Gate 1 uses interleaved execution via run_all")

    async def run_all(self) -> dict[str, list[ResultRecord]]:
        """Run conditions interleaved: RANDOM → BAYESIAN → GUIDED per round."""
        from datetime import datetime, timezone

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._results_dir.mkdir(parents=True, exist_ok=True)

        # Load artifact and eval config
        criteria, test_prompts, gen_config = _load_eval_criteria(
            self._repo_root, self.config.eval_criteria_path,
        )
        stochastic_config = _build_stochastic_config(criteria, test_prompts, gen_config)
        mutation_model = "gemini-2.5-flash"
        mutation_client = _make_client(mutation_model)

        # Read initial artifact
        original_artifact = ""
        for rel_path in self.config.artifact_paths:
            full = self._repo_root / rel_path
            if full.exists():
                original_artifact += full.read_text(encoding="utf-8")

        # Per-condition state
        condition_map = {c.name: c for c in self.config.conditions}
        artifacts: dict[str, str] = {}
        baselines: dict[str, float] = {}
        histories: dict[str, list[ResultRecord]] = {}
        results: dict[str, list[ResultRecord]] = {}
        experiment_counters: dict[str, int] = {}
        total_costs: dict[str, float] = {}

        for cond_name in EXECUTION_ORDER:
            artifacts[cond_name] = original_artifact
            baselines[cond_name] = 0.0
            histories[cond_name] = []
            results[cond_name] = []
            experiment_counters[cond_name] = 0
            total_costs[cond_name] = 0.0

        # Run baseline eval
        console.print("[dim]Running baseline eval...[/dim]", end=" ")
        baseline_result = await self._evaluator.evaluate(
            worktree_path=self._repo_root,
            config=stochastic_config,
            artifact_content=original_artifact,
        )
        baseline_score = baseline_result.score
        console.print(f"[bold]{baseline_score:.4f}[/bold]")
        for cond_name in EXECUTION_ORDER:
            baselines[cond_name] = baseline_score

        n = self.config.max_experiments_per_condition
        midpoint_warned = False

        # Multi-task progress: one bar per condition
        progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("{task.description:>10s}"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            TextColumn("{task.fields[status]}"),
            TimeElapsedColumn(),
            console=console,
        )
        task_ids: dict[str, object] = {}
        for cond_name in EXECUTION_ORDER:
            task_ids[cond_name] = progress.add_task(
                cond_name, total=n, status=f"baseline={baseline_score:.3f}",
            )

        with progress:
            for round_idx in range(n):
                for cond_name in EXECUTION_ORDER:
                    condition = condition_map[cond_name]
                    exp_idx = experiment_counters[cond_name]
                    experiment_counters[cond_name] += 1

                    # Build history text for guided condition
                    history_text = ""
                    if condition.knowledge_enabled and histories[cond_name]:
                        history_lines: list[str] = []
                        for r in histories[cond_name][-5:]:
                            tag = "KEPT" if r.kept else "DISC"
                            history_lines.append(
                                f"  [{tag}] score={r.score:.4f} | {r.hypothesis[:80]}"
                            )
                        history_text = "\n".join(history_lines)

                    # Cross-condition insights for guided
                    cross_insights = ""
                    if condition.knowledge_enabled:
                        cross_insights = self._learning_pool.summarize(
                            scope=LearningScope.TARGET,
                            k=5,
                            exclude_condition=cond_name,
                        )

                    start = time.monotonic()

                    # Mutate
                    mutated, hypothesis, mutation_cost = await _mutate_artifact(
                        mutation_client,
                        mutation_model,
                        artifacts[cond_name],
                        cond_name,
                        history_text,
                        cross_insights,
                    )

                    # Evaluate mutated artifact
                    eval_result: EvalResult = await self._evaluator.evaluate(
                        worktree_path=self._repo_root,
                        config=stochastic_config,
                        artifact_content=mutated,
                    )
                    elapsed = time.monotonic() - start
                    exp_cost = mutation_cost + eval_result.cost_usd

                    # Keep/discard decision
                    kept = eval_result.score > baselines[cond_name]
                    if kept:
                        artifacts[cond_name] = mutated
                        baselines[cond_name] = eval_result.score

                    # Per-criterion scores
                    per_criterion: dict[str, float] = {}
                    if eval_result.raw_scores:
                        for i, c in enumerate(criteria):
                            if i < len(eval_result.raw_scores):
                                per_criterion[c.name] = eval_result.raw_scores[i]

                    record = ResultRecord(
                        condition=cond_name,
                        experiment_idx=exp_idx + 1,
                        hypothesis=hypothesis,
                        score=eval_result.score,
                        ci_lower=eval_result.ci_lower,
                        ci_upper=eval_result.ci_upper,
                        baseline_score=baselines[cond_name] if not kept else eval_result.score,
                        kept=kept,
                        cost_usd=exp_cost,
                        duration_seconds=elapsed,
                        seed=self.config.seed + round_idx,
                        raw_scores=list(eval_result.raw_scores) if eval_result.raw_scores else [],
                        per_criterion=per_criterion,
                    )
                    results[cond_name].append(record)
                    histories[cond_name].append(record)
                    total_costs[cond_name] += exp_cost

                    # Update progress bar
                    tag = "[green]KEPT[/]" if kept else "[dim]DISC[/]"
                    progress.update(
                        task_ids[cond_name],  # type: ignore[arg-type]
                        advance=1,
                        status=(
                            f"{tag} {eval_result.score:.3f}  "
                            f"best={baselines[cond_name]:.3f}  "
                            f"${total_costs[cond_name]:.3f}"
                        ),
                    )

                    # Extract learning for cross-condition pool
                    engine_record = ExperimentRecord(
                        id=str(exp_idx + 1),
                        target_id="gate1",
                        git_sha="",
                        pre_experiment_sha="",
                        timestamp=datetime.now(tz=timezone.utc),
                        hypothesis=hypothesis,
                        hypothesis_source="agent",
                        mutation_diff_summary="",
                        score=eval_result.score,
                        score_ci_lower=eval_result.ci_lower,
                        score_ci_upper=eval_result.ci_upper,
                        raw_scores=eval_result.raw_scores,
                        baseline_score=baselines[cond_name],
                        outcome=Outcome.KEPT if kept else Outcome.DISCARDED,
                        failure_mode=None,
                        duration_seconds=elapsed,
                        tags=[],
                        learnings="",
                        cost_usd=exp_cost,
                        bootstrap_seed=0,
                    )
                    learning = extract_learning(
                        engine_record,
                        source_condition=cond_name,
                        source_target="gate1",
                    )
                    self._learning_pool.add(learning)

                # Midpoint early warning check
                if round_idx + 1 == self._midpoint_check and not midpoint_warned:
                    midpoint_warned = True
                    self._midpoint_analysis(results, round_idx + 1)

        # Save and report
        self.save_results(results)
        self._final_analysis(results)
        return results

    def _midpoint_analysis(
        self, results: dict[str, list[ResultRecord]], at_experiment: int,
    ) -> None:
        """Check if guided is outperforming random at the midpoint."""
        from rich.panel import Panel

        guided = results.get("guided", [])
        random_cond = results.get("random", [])

        if not guided or not random_cond:
            return

        guided_best = max(r.score for r in guided)
        random_best = max(r.score for r in random_cond)
        random_ci_upper = max(
            (r.ci_upper for r in random_cond if r.ci_upper is not None),
            default=random_best,
        )

        if guided_best <= random_ci_upper:
            verdict = (
                f"[yellow]WARNING: Guided ({guided_best:.4f}) is within random's CI "
                f"({random_ci_upper:.4f}). LLM advantage not yet demonstrated.[/yellow]"
            )
            style = "yellow"
        else:
            verdict = f"[green]Guided exceeds random CI upper. On track.[/green]"
            style = "green"

        console.print(Panel(
            f"  Guided best:      {guided_best:.4f}\n"
            f"  Random best:      {random_best:.4f}\n"
            f"  Random CI upper:  {random_ci_upper:.4f}\n\n"
            f"  {verdict}",
            title=f"Midpoint Check (experiment {at_experiment})",
            style=style,
        ))

    def _final_analysis(self, results: dict[str, list[ResultRecord]]) -> None:
        """Write go/no-go recommendation."""
        from rich.panel import Panel

        condition_bests: dict[str, float] = {}
        for cond_name, records in results.items():
            scores = [r.score for r in records]
            condition_bests[cond_name] = max(scores) if scores else 0.0

        winner = max(condition_bests, key=lambda k: condition_bests[k])
        guided_best = condition_bests.get("guided", 0.0)
        random_best = condition_bests.get("random", 0.0)

        go = guided_best > random_best
        recommendation = (
            "GO — guided mutation outperforms random search."
            if go
            else "NO-GO — guided mutation does not outperform random. "
            "Consider pivoting to simpler infrastructure tool."
        )

        report_lines = [
            "=" * 60,
            "GATE 1 — GO/NO-GO DECISION",
            "=" * 60,
        ]

        panel_lines: list[str] = []
        for cond, best in sorted(condition_bests.items()):
            records = results[cond]
            kept = sum(1 for r in records if r.kept)
            cost = sum(r.cost_usd for r in records)
            panel_lines.append(
                f"  [bold]{cond}[/bold]: best={best:.4f}  "
                f"n={len(records)}  kept={kept}  ${cost:.4f}"
            )
            report_lines.append(
                f"\n  {cond}:"
                f"\n    Best score:     {best:.4f}"
                f"\n    Experiments:    {len(records)}"
                f"\n    Kept:           {kept}"
                f"\n    Cost:           ${cost:.4f}"
            )

        style = "green" if go else "red"
        panel_lines.append("")
        panel_lines.append(f"  Winner: [bold]{winner}[/bold] ({condition_bests[winner]:.4f})")
        panel_lines.append(f"  {recommendation}")
        panel_lines.append(f"  Learning pool: {self._learning_pool.count} learnings")

        console.print()
        console.print(Panel(
            "\n".join(panel_lines),
            title="Gate 1 — GO/NO-GO Decision",
            style=style,
        ))

        report_lines.extend([
            "",
            "-" * 60,
            f"  Winner: {winner} (score: {condition_bests[winner]:.4f})",
            f"  Recommendation: {recommendation}",
            f"  Learning pool: {self._learning_pool.count} learnings extracted",
            "=" * 60,
        ])

        report_text = "\n".join(report_lines) + "\n"
        decision_path = self.results_dir / "decision.txt"
        decision_path.write_text(report_text, encoding="utf-8")


async def main() -> None:
    config = load_config(EXPERIMENT_DIR / "config.toml")
    harness = GuidedSearchExperiment(config, EXPERIMENT_DIR)
    await harness.run_all()


if __name__ == "__main__":
    asyncio.run(main())
