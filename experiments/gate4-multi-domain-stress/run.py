"""Gate 4: Multi-domain stress test.

Runs the optimization loop across multiple domains with different feedback
loop speeds. Documents failure modes and convergence characteristics.

Uses a local temp git repo per domain so the production ExperimentRunner
operates exactly as it would in a real optimization.

Usage:
    PYTHONPATH=. uv run python experiments/gate4-multi-domain-stress/run.py
"""

from __future__ import annotations

import asyncio
import sys
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
from experiments._harness.git_setup import ExperimentRepo, create_experiment_repo
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

    def _build_target(
        self, condition: ConditionConfig, repo: ExperimentRepo,
    ) -> OptimizationTarget:
        """Build domain-specific OptimizationTarget from condition config."""
        extra = condition.extra
        domain = str(extra.get("domain", "deterministic"))

        eval_config: EvalConfig
        eval_mode: EvalMode

        if domain == "stochastic":
            criteria_path = repo.repo_root / str(extra.get("eval_criteria_path", ""))
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

        # knowledge_path determines where program.md is read from.
        # Use the artifact's parent directory so the runner finds program.md
        # next to the artifact (e.g., examples/code-minify/program.md).
        if artifact_paths:
            knowledge_path = str(Path(str(artifact_paths[0])).parent)
        else:
            knowledge_path = str(repo.worktree_path / ".anneal" / "knowledge")

        return OptimizationTarget(
            id=f"gate4-{condition.name}",
            domain_tier=DomainTier.SANDBOX,
            artifact_paths=list(artifact_paths),
            scope_path=repo.scope_rel_path,
            scope_hash=repo.scope_hash,
            eval_mode=eval_mode,
            eval_config=eval_config,
            agent_config=agent_config,
            time_budget_seconds=300,
            loop_interval_seconds=300,
            knowledge_path=knowledge_path,
            worktree_path=str(repo.worktree_path),
            git_branch=f"anneal/gate4-{condition.name}",
            baseline_score=0.0,
            budget_cap=BudgetCap(max_usd_per_day=5.0),
        )

    def _get_repo_files(self, condition: ConditionConfig) -> tuple[list[str], str, str, list[str]]:
        """Extract file paths needed for the temp repo from condition config."""
        extra = condition.extra
        artifact_paths = extra.get("artifact_paths", [])
        if isinstance(artifact_paths, str):
            artifact_paths = [artifact_paths]
        scope_path = str(extra.get("scope_path", ""))
        eval_criteria_path = str(extra.get("eval_criteria_path", ""))

        # Collect extra files: eval scripts, their siblings, and program.md
        extra_files: list[str] = []
        run_cmd = str(extra.get("run_cmd", ""))
        parse_cmd = str(extra.get("parse_cmd", ""))

        # Files referenced directly in commands
        for cmd in [run_cmd, parse_cmd]:
            for token in cmd.split():
                if "/" in token and (self._repo_root / token).exists():
                    path = self._repo_root / token
                    extra_files.append(token)
                    # Include all sibling files in the same directory
                    # (eval scripts often reference siblings via $SCRIPT_DIR)
                    if path.is_file():
                        for sibling in path.parent.iterdir():
                            if sibling.is_file():
                                rel = str(sibling.relative_to(self._repo_root))
                                if rel not in extra_files and rel not in artifact_paths and rel != scope_path:
                                    extra_files.append(rel)

        # Include program.md if it exists alongside the artifact
        if artifact_paths:
            artifact_dir = str(Path(str(artifact_paths[0])).parent)
            program_md = f"{artifact_dir}/program.md"
            if (self._repo_root / program_md).exists() and program_md not in extra_files:
                extra_files.append(program_md)

        return list(artifact_paths), scope_path, eval_criteria_path, extra_files

    async def run_condition(self, condition: ConditionConfig) -> list[ResultRecord]:
        """Run N experiments for a single domain in a temp git repo."""
        import gc

        artifact_paths, scope_path, eval_criteria_path, extra_files = self._get_repo_files(condition)

        async with create_experiment_repo(
            source_root=self._repo_root,
            artifact_paths=artifact_paths,
            scope_path=scope_path,
            eval_criteria_path=eval_criteria_path,
            target_id=f"gate4-{condition.name}",
            extra_files=extra_files,
        ) as repo:
            target = self._build_target(condition, repo)

            # Run baseline eval to set the starting score.
            # Critical for LOWER_IS_BETTER: baseline of 0.0 means nothing
            # can ever be kept (score < 0 is impossible).
            eval_engine = EvalEngine()
            try:
                artifact_content = ""
                for ap in target.artifact_paths:
                    fp = repo.worktree_path / ap
                    if fp.exists():
                        artifact_content += fp.read_text(encoding="utf-8")
                baseline_result = await eval_engine.evaluate(
                    repo.worktree_path, target.eval_config, artifact_content,
                )
                target.baseline_score = baseline_result.score
            except Exception:
                pass  # Keep 0.0 baseline if eval fails — experiments will still run

            # Pre-register target so runner.update_target() works on KEPT outcomes
            repo.pre_register_target(target)

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
                            tags=list(engine_record.tags),
                            failure_mode=engine_record.failure_mode or "",
                        )
                    except Exception as exc:
                        elapsed = time.monotonic() - start
                        record = ResultRecord(
                            condition=condition.name,
                            experiment_idx=exp_idx + 1,
                            hypothesis=f"{condition.name} (crashed)",
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
            # Use baseline_score from first record (set by baseline eval before loop)
            baseline = records[0].baseline_score if records else 0.0
            # "Improved" = at least one experiment was kept by the runner.
            # This is direction-agnostic — the runner already handles
            # HIGHER_IS_BETTER vs LOWER_IS_BETTER in the keep decision.
            improved = kept > 0
            best_score = max(scores) if scores else 0.0

            crash_rate = crashes / max(n, 1)
            crash_ok = crash_rate <= max_crash_rate

            if improved:
                improving_domains += 1

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
                f"    Score:          baseline={baseline:.4f}  best={best_score:.4f}  {'improved' if improved else 'stalled'}",
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
                "improvement": abs(best_score - baseline) if improved else 0.0,
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
