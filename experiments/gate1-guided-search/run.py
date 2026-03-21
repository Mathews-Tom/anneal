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


_EDIT_FORMAT = """You edit text artifacts by producing SEARCH/REPLACE blocks.

For each change, output exactly this format:

<<<SEARCH
exact text to find in the artifact
>>>REPLACE
replacement text
<<<END

Rules:
- The SEARCH block must match the artifact EXACTLY (whitespace, punctuation, case)
- You may output multiple SEARCH/REPLACE blocks for multiple changes
- Keep changes focused and meaningful — each block should serve the hypothesis
- Start your response with ## Hypothesis followed by a one-sentence explanation
- Then output SEARCH/REPLACE blocks"""

_CONDITION_PROMPTS: dict[str, tuple[str, str]] = {
    "random": (
        "You mutate text artifacts randomly to explore the search space. "
        "Make a substantive structural change — reword a section, add a guideline, "
        "reorganize bullet points, change a constraint value, or add an example. "
        "Do NOT make trivial character-level edits.",

        "Make one random but substantive modification to this artifact. "
        "Change something structural: rewrite a section, add or remove a guideline, "
        "adjust a constraint, reorganize the content, or add a concrete example.",
    ),
    "bayesian": (
        "You optimize text artifacts through systematic parameter exploration. "
        "Each mutation should adjust exactly one dimension: structure, wording "
        "precision, constraint values, formatting, examples, or ordering. "
        "Explore the parameter space methodically.",

        "Optimize this artifact by adjusting one specific parameter. Pick one "
        "dimension (structure, wording, constraints, examples, ordering) and "
        "make a targeted change along that dimension.",
    ),
    "guided": (
        "You optimize text artifacts using experiment history and cross-condition "
        "insights. Analyze which changes improved or degraded specific evaluation "
        "criteria, then make an informed mutation that targets the weakest criteria.",

        "Analyze the experiment history and evaluation criteria below. Identify "
        "which criteria are weakest and propose a targeted change to improve them.",
    ),
}


def _apply_edits(artifact: str, edit_blocks: str) -> tuple[str, int, int]:
    """Apply SEARCH/REPLACE blocks to the artifact content.

    Three-tier matching:
      1. Exact string match
      2. Whitespace-normalized line-by-line match
      3. Difflib sequence similarity (threshold 0.85)

    Returns (mutated_artifact, blocks_found, blocks_applied).
    """
    import re
    from difflib import SequenceMatcher

    pattern = r"<<<SEARCH\n(.*?)\n>>>REPLACE\n(.*?)\n<<<END"
    matches = re.findall(pattern, edit_blocks, re.DOTALL)

    if not matches:
        # No edit blocks found — check if the model output a complete artifact
        cleaned = edit_blocks.strip()
        artifact_start = re.search(r"^#\s+\w", cleaned, re.MULTILINE)
        if artifact_start and "<<<SEARCH" not in cleaned:
            candidate = cleaned[artifact_start.start():]
            if len(candidate) > len(artifact) * 0.3:
                return candidate, 0, 0
        return artifact, 0, 0

    result = artifact
    applied = 0
    for search, replace in matches:
        # Tier 1: Exact match
        if search in result:
            result = result.replace(search, replace, 1)
            applied += 1
            continue

        # Tier 2: Whitespace-normalized line-by-line match
        search_lines = search.split("\n")
        result_lines = result.split("\n")
        tier2_matched = False
        for i in range(len(result_lines) - len(search_lines) + 1):
            window = result_lines[i:i + len(search_lines)]
            if all(
                sl.rstrip() == rl.rstrip()
                for sl, rl in zip(search_lines, window)
            ):
                result_lines[i:i + len(search_lines)] = replace.split("\n")
                result = "\n".join(result_lines)
                applied += 1
                tier2_matched = True
                break
        if tier2_matched:
            continue

        # Tier 3: Difflib sequence similarity on sliding window
        # Find the best-matching window of similar line count in the artifact
        search_norm = search.strip()
        best_ratio = 0.0
        best_start = -1
        best_end = -1
        window_sizes = [len(search_lines), len(search_lines) - 1, len(search_lines) + 1]
        result_lines = result.split("\n")
        for wsize in window_sizes:
            if wsize < 1 or wsize > len(result_lines):
                continue
            for i in range(len(result_lines) - wsize + 1):
                window_text = "\n".join(result_lines[i:i + wsize]).strip()
                ratio = SequenceMatcher(None, search_norm, window_text).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i
                    best_end = i + wsize

        if best_ratio >= 0.85 and best_start >= 0:
            result_lines = result.split("\n")
            result_lines[best_start:best_end] = replace.split("\n")
            result = "\n".join(result_lines)
            applied += 1

    return result, len(matches), applied


async def _mutate_artifact(
    client: openai.AsyncOpenAI,
    model: str,
    artifact_content: str,
    condition: str,
    history_text: str,
    cross_condition_insights: str,
    criteria_text: str = "",
) -> tuple[str, str, float]:
    """Generate a mutated artifact via targeted SEARCH/REPLACE edits.

    Returns (mutated_content, hypothesis, cost_usd).
    """
    from anneal.engine.agent import _compute_cost

    system_base, task_instruction = _CONDITION_PROMPTS[condition]
    system_prompt = f"{system_base}\n\n{_EDIT_FORMAT}"

    parts = [task_instruction, ""]

    if condition == "guided":
        if history_text:
            parts.append(f"## Previous Results\n{history_text}\n")
        if cross_condition_insights:
            parts.append(f"{cross_condition_insights}\n")

    if criteria_text:
        parts.append(f"## Evaluation Criteria\n{criteria_text}\n")

    parts.append(f"## Current Artifact\n\n{artifact_content}")

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
    cost = _compute_cost(model, input_tokens, output_tokens)

    # Apply edits to artifact
    mutated, blocks_found, blocks_applied = _apply_edits(artifact_content, output)

    # If no edits were applied and artifact is unchanged, the mutation failed
    if mutated == artifact_content:
        # Return original — the eval will score the same artifact and likely
        # discard it, which is the correct behavior for a failed mutation
        pass

    # Extract hypothesis
    hypothesis = ""
    for line in output.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("## hypothesis"):
            continue
        if hypothesis == "" and stripped and not stripped.startswith("<<<"):
            hypothesis = stripped
            break
    # Search for hypothesis after the header
    import re
    match = re.search(r"## Hypothesis\s*\n(.*?)(?=\n<<<|\n## |\Z)", output, re.DOTALL)
    if match:
        hypothesis = match.group(1).strip().split("\n")[0].strip()
    if not hypothesis:
        hypothesis = f"{condition} mutation"

    return mutated, hypothesis, cost


class GuidedSearchExperiment(ExperimentHarness):
    """A/B/C comparison: guided vs random vs bayesian."""

    def __init__(self, config: ExperimentConfig, experiment_dir: Path) -> None:
        super().__init__(config, experiment_dir)
        self._repo_root = PROJECT_ROOT
        self._evaluator = StochasticEvaluator()
        self._learning_pool = LearningPool()
        self._midpoint_check = 25

        # Load extra settings from config
        config_path = experiment_dir / "config.toml"
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self._midpoint_check = raw.get("experiment", {}).get("midpoint_check", 25)
        self._mutation_model = raw.get("experiment", {}).get("mutation_model", "gemini-2.5-pro")

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
        mutation_model = self._mutation_model
        mutation_client = _make_client(mutation_model)

        # Build criteria text for mutation context
        criteria_text = "\n".join(
            f"- **{c.name}**: {c.question}" for c in criteria
        )

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
                    hypothesis = ""
                    failure_mode = ""

                    try:
                        # Mutate
                        mutated, hypothesis, mutation_cost = await _mutate_artifact(
                            mutation_client,
                            mutation_model,
                            artifacts[cond_name],
                            cond_name,
                            history_text,
                            cross_insights,
                            criteria_text=criteria_text,
                        )

                        # Check if mutation actually changed the artifact
                        mutation_applied = mutated != artifacts[cond_name]

                        # Evaluate mutated artifact
                        eval_result: EvalResult = await self._evaluator.evaluate(
                            worktree_path=self._repo_root,
                            config=stochastic_config,
                            artifact_content=mutated,
                        )
                        elapsed = time.monotonic() - start
                        exp_cost = mutation_cost + eval_result.cost_usd

                        # Keep/discard decision — use CI-aware comparison
                        baseline = baselines[cond_name]
                        ci_lower = eval_result.ci_lower if eval_result.ci_lower is not None else eval_result.score
                        kept = eval_result.score > baseline or ci_lower >= baseline
                        if kept:
                            artifacts[cond_name] = mutated
                            baselines[cond_name] = eval_result.score

                        # Per-criterion scores
                        per_criterion: dict[str, float] = {}
                        if eval_result.raw_scores:
                            for i, c in enumerate(criteria):
                                if i < len(eval_result.raw_scores):
                                    per_criterion[c.name] = eval_result.raw_scores[i]

                        failure_mode = ""

                    except Exception as exc:
                        # API errors, timeout, JSON parse failures — log and continue
                        elapsed = time.monotonic() - start
                        eval_result = EvalResult(score=0.0)
                        if not hypothesis:
                            hypothesis = f"{cond_name} mutation (crashed)"
                        mutation_applied = False
                        kept = False
                        exp_cost = 0.0
                        per_criterion = {}
                        failure_mode = f"{type(exc).__name__}: {str(exc)[:100]}"

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
                        failure_mode=failure_mode,
                    )
                    results[cond_name].append(record)
                    histories[cond_name].append(record)
                    total_costs[cond_name] += exp_cost

                    # Update progress bar
                    if kept:
                        tag = "[green]KEPT[/]"
                    elif failure_mode:
                        tag = "[red]ERR![/]"
                    elif not mutation_applied:
                        tag = "[yellow]NOOP[/]"
                    else:
                        tag = "[dim]DISC[/]"
                    progress.update(
                        task_ids[cond_name],  # type: ignore[arg-type]
                        advance=1,
                        status=(
                            f"{tag} {eval_result.score:.3f}  "
                            f"best={baselines[cond_name]:.3f}  "
                            f"${total_costs[cond_name]:.3f}"
                        ),
                    )

                    # Extract learning for cross-condition pool (skip crashed)
                    if not failure_mode:
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
