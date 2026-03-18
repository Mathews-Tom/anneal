"""Base experiment harness for validation checkpoints."""

from __future__ import annotations

import shutil
import time
import tomllib
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from experiments._harness.plotting import plot_trajectory
from experiments._harness.report import write_csv, write_summary
from experiments._harness.types import ConditionConfig, ExperimentConfig, ResultRecord

console = Console(stderr=True)


def load_config(config_path: Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a TOML file."""
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    exp = data["experiment"]

    conditions = []
    for name, cdata in data.get("conditions", {}).items():
        conditions.append(ConditionConfig(
            name=name,
            description=cdata.get("description", ""),
            knowledge_enabled=cdata.get("knowledge_enabled", False),
            search_strategy=cdata.get("search_strategy", "greedy"),
            agent_model=cdata.get("agent_model", "sonnet"),
            evaluator_model=cdata.get("evaluator_model", "gpt-4.1"),
            extra={k: v for k, v in cdata.items() if k not in {
                "description", "knowledge_enabled", "search_strategy",
                "agent_model", "evaluator_model",
            }},
        ))

    return ExperimentConfig(
        name=exp["name"],
        checkpoint=exp["checkpoint"],
        description=exp["description"],
        conditions=conditions,
        max_experiments_per_condition=exp.get("max_experiments_per_condition", 50),
        seed=exp.get("seed", 42),
        output_dir=Path(exp.get("output_dir", "results")),
        artifact_paths=exp.get("artifact_paths", []),
        eval_criteria_path=exp.get("eval_criteria_path", ""),
        extra={k: v for k, v in exp.items() if k not in {
            "name", "checkpoint", "description", "max_experiments_per_condition",
            "seed", "output_dir", "artifact_paths", "eval_criteria_path",
        }},
    )


def make_progress(condition_name: str, total: int | None = None) -> tuple[Progress, object]:
    """Create a rich Progress bar for a condition run.

    Returns (progress, task_id). Caller uses `with progress:` context.
    """
    progress = Progress(
        SpinnerColumn("dots"),
        TextColumn(f"[bold cyan]{condition_name}[/]"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task(condition_name, total=total, status="starting...")
    return progress, task_id


def update_progress(
    progress: Progress,
    task_id: object,
    record: ResultRecord,
    best_score: float,
    total_cost: float,
) -> None:
    """Advance progress bar with experiment result."""
    if record.kept:
        tag = "[green]KEPT[/]"
    elif record.failure_mode:
        tag = "[red]FAIL[/]"
    else:
        tag = "[dim]DISC[/]"

    progress.update(
        task_id,  # type: ignore[arg-type]
        advance=1,
        status=f"{tag} {record.score:.3f}  best={best_score:.3f}  ${total_cost:.3f}",
    )


class ExperimentHarness:
    """Base class for validation experiments.

    Subclasses override `run_condition` to implement experiment-specific logic.
    The harness handles config loading, result persistence, and report generation.
    """

    def __init__(self, config: ExperimentConfig, experiment_dir: Path) -> None:
        self._config = config
        self._experiment_dir = experiment_dir
        self._results_dir = experiment_dir / config.output_dir

    @property
    def config(self) -> ExperimentConfig:
        return self._config

    @property
    def results_dir(self) -> Path:
        return self._results_dir

    async def run_condition(
        self, condition: ConditionConfig
    ) -> list[ResultRecord]:
        """Run all experiments for a single condition.

        Subclasses must override this method.
        """
        raise NotImplementedError

    async def run_all(self) -> dict[str, list[ResultRecord]]:
        """Run all conditions and return results keyed by condition name."""
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # Freeze config into results for reproducibility
        config_src = self._experiment_dir / "config.toml"
        if config_src.exists():
            shutil.copy2(config_src, self._results_dir / "config.toml")

        console.print()
        console.print(
            Panel(
                f"[bold]{self._config.name}[/bold]\n"
                f"  {self._config.description}\n"
                f"  Conditions: {', '.join(c.name for c in self._config.conditions)}\n"
                f"  Max experiments: {self._config.max_experiments_per_condition} per condition",
                title=f"Validation Gate: {self._config.checkpoint}",
                style="blue",
            )
        )

        all_results: dict[str, list[ResultRecord]] = {}

        for condition in self._config.conditions:
            console.print()
            start = time.monotonic()
            records = await self.run_condition(condition)
            elapsed = time.monotonic() - start
            all_results[condition.name] = records

            kept = sum(1 for r in records if r.kept)
            cost = sum(r.cost_usd for r in records)
            best = max((r.score for r in records), default=0.0)
            console.print(
                f"  [dim]{condition.name}: {len(records)} experiments, "
                f"{kept} kept, best={best:.3f}, ${cost:.4f}, {elapsed:.1f}s[/dim]"
            )

        self.save_results(all_results)
        return all_results

    def save_results(self, results: dict[str, list[ResultRecord]]) -> None:
        """Write CSV, summary, and trajectory plot."""
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # Flatten all records for CSV
        all_records: list[ResultRecord] = []
        for records in results.values():
            all_records.extend(records)

        write_csv(all_records, self._results_dir / "experiments.csv")
        write_summary(
            results,
            self._results_dir / "summary.txt",
            title=f"{self._config.name.upper()} — {self._config.description}",
        )
        plot_trajectory(
            results,
            self._results_dir / "trajectory.png",
            title=f"{self._config.name}: Score Trajectory",
        )

        console.print(
            Panel(
                f"Results saved to [bold]{self._results_dir}/[/bold]",
                style="green",
            )
        )
