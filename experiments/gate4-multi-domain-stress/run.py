"""V4: Multi-domain stress test.

Runs the optimization loop across multiple domains with different feedback
loop speeds. Documents failure modes and convergence characteristics.

Usage:
    uv run python experiments/multi-domain-stress-v4/run.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import tomllib
from collections import Counter
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
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    OptimizationTarget,
    StochasticEval,
)
from experiments._harness.base import ExperimentHarness, load_config, make_progress, update_progress
from experiments._harness.plotting import plot_domain_comparison, plot_trajectory
from experiments._harness.types import ConditionConfig, ExperimentConfig, ResultRecord

EXPERIMENT_DIR = Path(__file__).resolve().parent


class MultiDomainExperiment(ExperimentHarness):
    """Runs optimization across multiple domains."""

    def __init__(self, config: ExperimentConfig, experiment_dir: Path) -> None:
        super().__init__(config, experiment_dir)
        self._repo_root = PROJECT_ROOT
        self._thresholds = self._load_thresholds()

    def _load_thresholds(self) -> dict[str, float]:
        config_path = self._experiment_dir / "config.toml"
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return data.get("thresholds", {
            "min_improving_domains": 2,
            "max_crash_rate": 0.30,
        })

    def _build_target(self, condition: ConditionConfig, work_dir: Path) -> OptimizationTarget:
        """Build domain-specific OptimizationTarget from condition config."""
        extra = condition.extra
        domain = str(extra.get("domain", "deterministic"))

        eval_config: EvalConfig
        eval_mode: EvalMode

        if domain == "stochastic":
            criteria_path = self._repo_root / str(extra.get("eval_criteria_path", ""))
            crit_data = tomllib.loads(criteria_path.read_text(encoding="utf-8"))

            criteria = [
                BinaryCriterion(name=c["name"], question=c["question"])
                for c in crit_data.get("criteria", [])
            ]
            test_prompts = [tp["prompt"] for tp in crit_data.get("test_prompts", [])]
            gen = crit_data.get("generation", {})

            gen_agent = AgentConfig(
                mode="api",
                model=gen.get("agent", {}).get("model", "gemini-2.5-flash"),
                evaluator_model="gpt-4.1",
                max_budget_usd=0.02,
                temperature=gen.get("agent", {}).get("temperature", 0.7),
            )

            stochastic_eval = StochasticEval(
                sample_count=len(test_prompts),
                criteria=criteria,
                test_prompts=test_prompts,
                generation_prompt_template=gen.get("prompt_template", ""),
                output_format=gen.get("output_format", "text"),
                confidence_level=0.95,
                generation_agent_config=gen_agent,
            )

            eval_config = EvalConfig(
                metric_name="binary_criteria_score",
                direction=Direction.HIGHER_IS_BETTER,
                stochastic=stochastic_eval,
            )
            eval_mode = EvalMode.STOCHASTIC
        else:
            deterministic_eval = DeterministicEval(
                run_command=str(extra.get("run_cmd", "")),
                parse_command=str(extra.get("parse_cmd", "")),
                timeout_seconds=300,
            )
            eval_config = EvalConfig(
                metric_name="deterministic_score",
                direction=Direction.LOWER_IS_BETTER,
                deterministic=deterministic_eval,
            )
            eval_mode = EvalMode.DETERMINISTIC

        artifact_paths = extra.get("artifact_paths", [])
        if isinstance(artifact_paths, str):
            artifact_paths = [artifact_paths]

        agent_config = AgentConfig(
            mode="claude_code",
            model=condition.agent_model,
            evaluator_model=condition.evaluator_model,
            max_budget_usd=1.00,
        )

        return OptimizationTarget(
            id=f"gate4-{condition.name}",
            domain_tier=DomainTier.SANDBOX,
            artifact_paths=list(artifact_paths),
            scope_path=str(extra.get("scope_path", "")),
            scope_hash="experiment",
            eval_mode=eval_mode,
            eval_config=eval_config,
            agent_config=agent_config,
            time_budget_seconds=300,
            loop_interval_seconds=300,
            knowledge_path=str(work_dir / "knowledge"),
            worktree_path=str(work_dir / "worktree"),
            git_branch=f"anneal/gate4-{condition.name}",
            baseline_score=0.0,
            budget_cap=BudgetCap(max_usd_per_day=5.0),
        )

    async def run_condition(self, condition: ConditionConfig) -> list[ResultRecord]:
        """Run N experiments for a single domain."""
        with tempfile.TemporaryDirectory(prefix=f"anneal-gate4-{condition.name}-") as tmpdir:
            work_dir = Path(tmpdir)
            target = self._build_target(condition, work_dir)
            knowledge = KnowledgeStore(work_dir / "knowledge")

            git = GitEnvironment()
            runner = ExperimentRunner(
                git=git,
                agent_invoker=AgentInvoker(),
                eval_engine=EvalEngine(),
                search=GreedySearch(),
                registry=Registry(self._repo_root),
                repo_root=self._repo_root,
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
                        tags=list(engine_record.tags),
                        failure_mode=engine_record.failure_mode or "",
                    )
                    records.append(record)
                    total_cost += record.cost_usd
                    best_score = max(best_score, record.score)
                    update_progress(progress, task_id, record, best_score, total_cost)

            return records

    async def run_all(self) -> dict[str, list[ResultRecord]]:
        results = await super().run_all()
        self._analyze_domains(results)
        return results

    def _analyze_domains(self, results: dict[str, list[ResultRecord]]) -> None:
        """Cross-domain failure analysis."""
        t = self._thresholds
        max_crash_rate = t.get("max_crash_rate", 0.30)
        min_improving = int(t.get("min_improving_domains", 2))

        report_lines = [
            "=" * 60,
            "MULTI-DOMAIN STRESS TEST",
            "=" * 60,
        ]

        domain_metrics: dict[str, dict[str, float]] = {}
        improving_domains = 0

        for domain, records in sorted(results.items()):
            n = len(records)
            kept = sum(1 for r in records if r.kept)
            crashes = sum(1 for r in records if r.failure_mode and "crash" in r.failure_mode.lower())
            blocked = sum(1 for r in records if r.failure_mode and "blocked" in r.failure_mode.lower())
            total_cost = sum(r.cost_usd for r in records)
            total_time = sum(r.duration_seconds for r in records)

            scores = [r.score for r in records]
            best_score = max(scores) if scores else 0.0
            first_score = scores[0] if scores else 0.0
            improved = best_score > first_score

            crash_rate = crashes / max(n, 1)
            crash_ok = crash_rate <= max_crash_rate

            if improved:
                improving_domains += 1

            # Failure mode taxonomy
            failure_modes: Counter[str] = Counter()
            for r in records:
                if r.failure_mode:
                    failure_modes[r.failure_mode] += 1

            report_lines.extend([
                f"\n  {domain}:",
                f"    Experiments:    {n}",
                f"    Kept:           {kept} ({kept / max(n, 1):.0%})",
                f"    Crashes:        {crashes} ({crash_rate:.0%})  {'PASS' if crash_ok else 'FAIL'}",
                f"    Blocked:        {blocked}",
                f"    Score range:    {first_score:.4f} -> {best_score:.4f}  {'improved' if improved else 'stalled'}",
                f"    Total cost:     ${total_cost:.4f}",
                f"    Avg duration:   {total_time / max(n, 1):.1f}s",
            ])

            if failure_modes:
                report_lines.append("    Failure modes:")
                for mode, count in failure_modes.most_common(5):
                    report_lines.append(f"      {mode}: {count}")

            domain_metrics[domain] = {
                "kept_rate": kept / max(n, 1),
                "crash_rate": crash_rate,
                "improvement": best_score - first_score,
                "cost_per_kept": total_cost / max(kept, 1),
            }

        passes = improving_domains >= min_improving

        report_lines.extend([
            "",
            "-" * 60,
            f"  Improving domains: {improving_domains}/{len(results)} (threshold: {min_improving})",
            f"  VERDICT: {'PASS' if passes else 'FAIL'}",
            "",
            f"  {'Pattern works across multiple domains.' if passes else 'Pattern fails in too many domains. Document failure modes above.'}",
            "=" * 60,
        ])

        report_text = "\n".join(report_lines) + "\n"
        report_path = self.results_dir / "domains.txt"
        report_path.write_text(report_text, encoding="utf-8")
        print(report_text)

        # Cross-domain comparison chart
        plot_domain_comparison(
            domain_metrics,
            self.results_dir / "domain_comparison.png",
            title="Cross-Domain Performance Comparison",
        )


async def main() -> None:
    config = load_config(EXPERIMENT_DIR / "config.toml")
    harness = MultiDomainExperiment(config, EXPERIMENT_DIR)
    await harness.run_all()


if __name__ == "__main__":
    asyncio.run(main())
