"""V2: Stochastic eval stability validation.

Runs N repeated baseline evaluations on the same artifact with different
seeds. Measures score variance to determine if the stochastic eval
framework produces stable rankings.

Usage:
    uv run python experiments/eval-stability-v2/run.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from anneal.engine.eval import StochasticEvaluator
from anneal.engine.types import AgentConfig, BinaryCriterion, EvalResult, StochasticEval
from experiments._harness.base import ExperimentHarness, load_config, make_progress, update_progress
from experiments._harness.plotting import plot_variance
from experiments._harness.types import ConditionConfig, ExperimentConfig, ResultRecord

EXPERIMENT_DIR = Path(__file__).resolve().parent


def _load_eval_criteria(repo_root: Path, criteria_path: str) -> tuple[list[BinaryCriterion], list[str], dict[str, object]]:
    """Load criteria, test prompts, and generation config from eval_criteria.toml."""
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


class EvalStabilityExperiment(ExperimentHarness):
    """Runs repeated baseline evals and measures variance."""

    def __init__(self, config: ExperimentConfig, experiment_dir: Path) -> None:
        super().__init__(config, experiment_dir)
        self._evaluator = StochasticEvaluator()
        self._repo_root = PROJECT_ROOT
        self._thresholds = self._load_thresholds()

    def _load_thresholds(self) -> dict[str, float]:
        config_path = self._experiment_dir / "config.toml"
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return data.get("thresholds", {
            "max_score_variance": 0.1,
            "max_criterion_variance": 0.15,
            "max_coefficient_of_variation": 0.20,
        })

    def _build_stochastic_config(self) -> StochasticEval:
        """Build StochasticEval from the experiment's eval_criteria.toml."""
        criteria, test_prompts, gen_config = _load_eval_criteria(
            self._repo_root, self.config.eval_criteria_path,
        )

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

    async def run_condition(self, condition: ConditionConfig) -> list[ResultRecord]:
        """Run N repeated evals on the same artifact."""
        stochastic_config = self._build_stochastic_config()

        # Read artifact content
        artifact_parts: list[str] = []
        for rel_path in self.config.artifact_paths:
            full = self._repo_root / rel_path
            if full.exists():
                artifact_parts.append(full.read_text(encoding="utf-8"))
        artifact_content = "\n\n".join(artifact_parts)

        records: list[ResultRecord] = []
        n_runs = self.config.max_experiments_per_condition
        total_cost = 0.0
        best_score = 0.0

        progress, task_id = make_progress(f"{condition.name} (eval stability)", total=n_runs)

        with progress:
            for run_idx in range(n_runs):
                seed = self.config.seed + run_idx

                start = time.monotonic()
                result: EvalResult = await self._evaluator.evaluate(
                    worktree_path=self._repo_root,
                    config=stochastic_config,
                    artifact_content=artifact_content,
                )
                elapsed = time.monotonic() - start

                per_criterion: dict[str, float] = {}
                if result.raw_scores:
                    for i, criterion in enumerate(stochastic_config.criteria):
                        per_criterion[criterion.name] = result.raw_scores[i] if i < len(result.raw_scores) else 0.0

                record = ResultRecord(
                    condition=condition.name,
                    experiment_idx=run_idx + 1,
                    hypothesis=f"Baseline eval run {run_idx + 1}",
                    score=result.score,
                    ci_lower=result.ci_lower,
                    ci_upper=result.ci_upper,
                    baseline_score=result.score,
                    kept=False,
                    cost_usd=result.cost_usd,
                    duration_seconds=elapsed,
                    seed=seed,
                    raw_scores=list(result.raw_scores) if result.raw_scores else [],
                    per_criterion=per_criterion,
                )
                records.append(record)
                total_cost += result.cost_usd
                best_score = max(best_score, result.score)
                update_progress(progress, task_id, record, best_score, total_cost)

        return records

    async def run_all(self) -> dict[str, list[ResultRecord]]:
        """Override to add stability analysis after standard run."""
        results = await super().run_all()

        # Stability analysis
        self._analyze_stability(results)

        return results

    def _analyze_stability(self, results: dict[str, list[ResultRecord]]) -> None:
        """Compute and report variance metrics. Write additional outputs."""
        all_records = [r for recs in results.values() for r in recs]
        if not all_records:
            return

        scores = [r.score for r in all_records]
        mean_score = float(np.mean(scores))
        score_var = float(np.var(scores))
        score_std = float(np.std(scores))
        cv = score_std / mean_score if mean_score > 0 else float("inf")

        # Per-criterion variance from raw_scores across runs
        raw_scores_per_run = [r.raw_scores for r in all_records if r.raw_scores]
        criterion_variances: dict[str, float] = {}
        if raw_scores_per_run:
            max_len = max(len(rs) for rs in raw_scores_per_run)
            for i in range(max_len):
                values = [rs[i] for rs in raw_scores_per_run if i < len(rs)]
                if len(values) >= 2:
                    criterion_variances[f"sample_{i}"] = float(np.var(values))

        # CI widths
        ci_widths = [
            (r.ci_upper - r.ci_lower) for r in all_records
            if r.ci_upper is not None and r.ci_lower is not None
        ]
        mean_ci_width = float(np.mean(ci_widths)) if ci_widths else 0.0

        # Pass/fail assessment
        t = self._thresholds
        var_pass = score_var <= t.get("max_score_variance", 0.1)
        cv_pass = cv <= t.get("max_coefficient_of_variation", 0.20)
        crit_pass = all(
            v <= t.get("max_criterion_variance", 0.15)
            for v in criterion_variances.values()
        )
        overall_pass = var_pass and cv_pass and crit_pass

        # Write stability report
        report_lines = [
            "=" * 60,
            "EVAL STABILITY ANALYSIS",
            "=" * 60,
            "",
            f"  Runs:                  {len(all_records)}",
            f"  Mean score:            {mean_score:.4f}",
            f"  Score variance:        {score_var:.6f}  {'PASS' if var_pass else 'FAIL'} (threshold: {t.get('max_score_variance', 0.1)})",
            f"  Score std dev:         {score_std:.4f}",
            f"  Coefficient of var:    {cv:.4f}  {'PASS' if cv_pass else 'FAIL'} (threshold: {t.get('max_coefficient_of_variation', 0.20)})",
            f"  Mean CI width:         {mean_ci_width:.4f}",
            "",
            "  Per-sample variances:",
        ]

        crit_threshold = t.get("max_criterion_variance", 0.15)
        for name, var in sorted(criterion_variances.items()):
            status = "PASS" if var <= crit_threshold else "FAIL"
            report_lines.append(f"    {name:20s}  {var:.6f}  {status}")

        report_lines.extend([
            "",
            "-" * 60,
            f"  VERDICT: {'PASS' if overall_pass else 'FAIL'}",
            "",
            f"  {'Stochastic eval is stable. Proceed to V3.' if overall_pass else 'Eval variance too high. Investigate sample count, criteria, or evaluator model.'}",
            "=" * 60,
        ])

        stability_text = "\n".join(report_lines) + "\n"
        stability_path = self.results_dir / "stability.txt"
        stability_path.write_text(stability_text, encoding="utf-8")
        print(stability_text)

        # Variance box plot
        if raw_scores_per_run:
            scores_per_sample: dict[str, list[float]] = {}
            max_len = max(len(rs) for rs in raw_scores_per_run)
            for i in range(max_len):
                key = f"sample_{i}"
                scores_per_sample[key] = [
                    rs[i] for rs in raw_scores_per_run if i < len(rs)
                ]
            plot_variance(
                scores_per_sample,
                self.results_dir / "stability.png",
                title="Per-Sample Score Variance Across Runs",
            )


async def main() -> None:
    config = load_config(EXPERIMENT_DIR / "config.toml")
    harness = EvalStabilityExperiment(config, EXPERIMENT_DIR)
    await harness.run_all()


if __name__ == "__main__":
    asyncio.run(main())
