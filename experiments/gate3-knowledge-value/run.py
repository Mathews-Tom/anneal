"""Gate 3: Knowledge store value validation.

A/B comparison: agent with experiment history vs memoryless agent.
Runs 50 experiments per condition on the same artifact and compares
score-at-N convergence curves.

Uses a local temp git repo (no GitHub, no mocks) so the production
ExperimentRunner operates exactly as it would in a real optimization.

Usage:
    PYTHONPATH=. uv run python experiments/gate3-knowledge-value/run.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from anneal.engine.agent import AgentInvoker
from anneal.engine.environment import GitEnvironment
from anneal.engine.eval import EvalEngine
from anneal.engine.knowledge import KnowledgeStore
from anneal.engine.registry import Registry
from anneal.engine.runner import ExperimentRunner
from anneal.engine.search import GreedySearch
from anneal.engine.types import (
    AgentConfig,
    BinaryCriterion,
    BudgetCap,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    OptimizationTarget,
    StochasticEval,
)
from experiments._harness.base import ExperimentHarness, load_config, make_progress, update_progress
from experiments._harness.git_setup import ExperimentRepo, create_experiment_repo
from experiments._harness.plotting import plot_trajectory
from experiments._harness.types import ConditionConfig, ExperimentConfig, ResultRecord

EXPERIMENT_DIR = Path(__file__).resolve().parent


def _load_eval_criteria(repo_root: Path, criteria_path: str) -> tuple[list[BinaryCriterion], list[str], dict[str, object]]:
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


class KnowledgeValueExperiment(ExperimentHarness):
    """A/B comparison: history-informed vs memoryless agent."""

    def __init__(self, config: ExperimentConfig, experiment_dir: Path) -> None:
        super().__init__(config, experiment_dir)
        self._repo_root = PROJECT_ROOT
        self._thresholds = self._load_thresholds()

    def _load_thresholds(self) -> dict[str, float]:
        config_path = self._experiment_dir / "config.toml"
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return data.get("thresholds", {
            "convergence_experiment": 30,
            "min_score_advantage": 0.05,
        })

    def _build_target(
        self, condition: ConditionConfig, repo: ExperimentRepo,
    ) -> OptimizationTarget:
        """Build an OptimizationTarget using the temp git repo."""
        criteria, test_prompts, gen_config = _load_eval_criteria(
            repo.repo_root, self.config.eval_criteria_path,
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

        stochastic_eval = StochasticEval(
            sample_count=len(test_prompts),
            criteria=criteria,
            test_prompts=test_prompts,
            generation_prompt_template=str(gen_config.get("prompt_template", "")),
            output_format=str(gen_config.get("output_format", "text")),
            confidence_level=0.95,
            generation_agent_config=gen_agent,
        )

        eval_config = EvalConfig(
            metric_name="binary_criteria_score",
            direction=Direction.HIGHER_IS_BETTER,
            stochastic=stochastic_eval,
        )

        agent_config = AgentConfig(
            mode="claude_code",
            model=condition.agent_model,
            evaluator_model=condition.evaluator_model,
            max_budget_usd=1.00,
        )

        knowledge_path = str(repo.worktree_path / ".anneal" / "knowledge")

        return OptimizationTarget(
            id=f"gate3-{condition.name}",
            domain_tier=DomainTier.SANDBOX,
            artifact_paths=list(self.config.artifact_paths),
            scope_path=repo.scope_rel_path,
            scope_hash=repo.scope_hash,
            eval_mode=EvalMode.STOCHASTIC,
            eval_config=eval_config,
            agent_config=agent_config,
            time_budget_seconds=300,
            loop_interval_seconds=300,
            knowledge_path=knowledge_path,
            worktree_path=str(repo.worktree_path),
            git_branch=f"anneal/gate3-{condition.name}",
            baseline_score=0.0,
            budget_cap=BudgetCap(max_usd_per_day=5.0),
        )

    async def run_condition(self, condition: ConditionConfig) -> list[ResultRecord]:
        """Run N experiments with or without knowledge store in a temp git repo."""
        import gc

        async with create_experiment_repo(
            source_root=self._repo_root,
            artifact_paths=list(self.config.artifact_paths),
            scope_path="examples/skill-diagram/scope.yaml",
            eval_criteria_path=self.config.eval_criteria_path,
            target_id=f"gate3-{condition.name}",
        ) as repo:
            target = self._build_target(condition, repo)

            # Run baseline eval to set the starting score
            try:
                artifact_content = ""
                for ap in target.artifact_paths:
                    fp = repo.worktree_path / ap
                    if fp.exists():
                        artifact_content += fp.read_text(encoding="utf-8")
                baseline_result = await EvalEngine().evaluate(
                    repo.worktree_path, target.eval_config, artifact_content,
                )
                target.baseline_score = baseline_result.score
            except Exception:
                pass

            # Pre-register target so runner.update_target() works on KEPT outcomes
            repo.pre_register_target(target)

            knowledge: KnowledgeStore | None = None
            if condition.knowledge_enabled:
                knowledge = KnowledgeStore(Path(target.knowledge_path))

            git = GitEnvironment()
            runner = ExperimentRunner(
                git=git,
                agent_invoker=AgentInvoker(),
                eval_engine=EvalEngine(),
                search=GreedySearch(),
                registry=Registry(repo.repo_root),
                repo_root=repo.repo_root,
                knowledge=knowledge,
            )

            records: list[ResultRecord] = []
            n = self.config.max_experiments_per_condition
            total_cost = 0.0
            best_score = 0.0

            progress, task_id = make_progress(condition.name, total=n)

            with progress:
                for exp_idx in range(n):
                    start = time.monotonic()

                    try:
                        engine_record = await runner.run_one(target)
                        elapsed = time.monotonic() - start

                        record = ResultRecord(
                            condition=condition.name,
                            experiment_idx=exp_idx + 1,
                            hypothesis=engine_record.hypothesis,
                            score=engine_record.score,
                            ci_lower=engine_record.score_ci_lower,
                            ci_upper=engine_record.score_ci_upper,
                            baseline_score=engine_record.baseline_score,
                            kept=engine_record.outcome.value == "KEPT",
                            cost_usd=engine_record.cost_usd,
                            duration_seconds=elapsed,
                            seed=self.config.seed + exp_idx,
                            raw_scores=list(engine_record.raw_scores) if engine_record.raw_scores else [],
                            tags=list(engine_record.tags),
                            failure_mode=engine_record.failure_mode or "",
                        )
                    except Exception as exc:
                        elapsed = time.monotonic() - start
                        record = ResultRecord(
                            condition=condition.name,
                            experiment_idx=exp_idx + 1,
                            hypothesis=f"{condition.name} mutation (crashed)",
                            score=0.0,
                            ci_lower=None,
                            ci_upper=None,
                            baseline_score=best_score,
                            kept=False,
                            cost_usd=0.0,
                            duration_seconds=elapsed,
                            seed=self.config.seed + exp_idx,
                            failure_mode=f"{type(exc).__name__}: {str(exc)[:100]}",
                        )

                    records.append(record)
                    total_cost += record.cost_usd
                    best_score = max(best_score, record.score)
                    update_progress(progress, task_id, record, best_score, total_cost)

            # Release subprocess transports before context exit triggers cleanup.
            # gc.collect() alone is insufficient — async transports have ref cycles.
            # Yielding to the event loop lets it process pending transport closures.
            gc.collect()
            await asyncio.sleep(0.1)
            gc.collect()

        return records

    async def run_all(self) -> dict[str, list[ResultRecord]]:
        results = await super().run_all()
        self._analyze_convergence(results)
        return results

    def _analyze_convergence(self, results: dict[str, list[ResultRecord]]) -> None:
        """Compare convergence curves between conditions."""
        t = self._thresholds
        convergence_n = int(t.get("convergence_experiment", 30))
        min_advantage = t.get("min_score_advantage", 0.05)

        report_lines = [
            "=" * 60,
            "KNOWLEDGE VALUE ANALYSIS",
            "=" * 60,
        ]

        for cond_name, records in sorted(results.items()):
            scores = [r.score for r in records]
            running_best: list[float] = []
            best_so_far = float("-inf")
            for s in scores:
                best_so_far = max(best_so_far, s)
                running_best.append(best_so_far)

            report_lines.append(f"\n  {cond_name}:")
            report_lines.append(f"    Final best:           {running_best[-1]:.4f}" if running_best else "    No data")
            report_lines.append(f"    Score at exp {convergence_n}:     {running_best[convergence_n - 1]:.4f}" if len(running_best) >= convergence_n else f"    Score at exp {convergence_n}:     N/A (only {len(running_best)} experiments)")
            report_lines.append(f"    Kept rate:            {sum(1 for r in records if r.kept) / max(len(records), 1):.2%}")
            report_lines.append(f"    Total cost:           ${sum(r.cost_usd for r in records):.4f}")

        informed_records = results.get("history-informed", [])
        memoryless_records = results.get("memoryless", [])

        if informed_records and memoryless_records:
            def best_at_n(records: list[ResultRecord], n: int) -> float:
                best = float("-inf")
                for r in records[:n]:
                    best = max(best, r.score)
                return best

            informed_at_n = best_at_n(informed_records, convergence_n)
            memoryless_at_n = best_at_n(memoryless_records, convergence_n)
            advantage = informed_at_n - memoryless_at_n
            passes = advantage >= min_advantage

            report_lines.extend([
                "",
                "-" * 60,
                f"  Score advantage at exp {convergence_n}: {advantage:+.4f}",
                f"  Threshold: {min_advantage}",
                f"  VERDICT: {'PASS' if passes else 'FAIL'}",
                "",
                f"  {'Knowledge store provides measurable convergence advantage.' if passes else 'Knowledge store does not improve convergence. Consider simplifying to structured logging.'}",
                "=" * 60,
            ])
        else:
            report_lines.extend([
                "",
                "  Cannot compare — missing condition data.",
                "=" * 60,
            ])

        report_text = "\n".join(report_lines) + "\n"
        report_path = self.results_dir / "convergence.txt"
        report_path.write_text(report_text, encoding="utf-8")
        print(report_text)

        # Convergence plot: running best score per condition
        convergence_results: dict[str, list[ResultRecord]] = {}
        for cond_name, records in results.items():
            running_best_records: list[ResultRecord] = []
            best_so_far = float("-inf")
            for r in records:
                best_so_far = max(best_so_far, r.score)
                running_best_records.append(ResultRecord(
                    condition=r.condition,
                    experiment_idx=r.experiment_idx,
                    hypothesis="",
                    score=best_so_far,
                    ci_lower=None,
                    ci_upper=None,
                    baseline_score=r.baseline_score,
                    kept=r.kept,
                    cost_usd=r.cost_usd,
                    duration_seconds=r.duration_seconds,
                    seed=r.seed,
                ))
            convergence_results[cond_name] = running_best_records

        plot_trajectory(
            convergence_results,
            self.results_dir / "convergence.png",
            title="Knowledge Value: Best Score at N",
        )


async def main() -> None:
    config = load_config(EXPERIMENT_DIR / "config.toml")
    harness = KnowledgeValueExperiment(config, EXPERIMENT_DIR)
    await harness.run_all()


if __name__ == "__main__":
    asyncio.run(main())
