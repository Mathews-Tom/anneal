"""Benchmark suite orchestration engine.

Responsible for:
  1. Building the full matrix of BenchmarkRun objects from targets and configs.
  2. Generating anneal register and anneal run shell commands for each run.
  3. Executing commands via subprocess (or previewing them in dry-run mode).
  4. Collecting result records from anneal output into JSONL files.
  5. Supporting parallel execution with configurable concurrency.

Usage: uv run python benchmarks/suite/run_suite.py --dry-run
"""
from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchmarks.suite.config import BenchmarkConfig, BenchmarkRun, BenchmarkTarget

console = Console()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_SUITE_DIR = Path(__file__).parent
_REPO_ROOT = _SUITE_DIR.parent.parent

# The four experimental configurations applied to every target.
BENCHMARK_CONFIGS: list[BenchmarkConfig] = [
    BenchmarkConfig(
        name="raw",
        description="Raw LLM generation, no optimization",
        search_strategy="none",
        enhancements={
            "strategy_manifest": False,
            "dual_agent": False,
            "two_phase": False,
            "lineage_context": False,
            "episodic_memory": False,
        },
    ),
    BenchmarkConfig(
        name="greedy",
        description="Anneal with greedy search, no knowledge injection",
        search_strategy="greedy",
        enhancements={
            "strategy_manifest": False,
            "dual_agent": False,
            "two_phase": False,
            "lineage_context": False,
            "episodic_memory": False,
        },
    ),
    BenchmarkConfig(
        name="control",
        description="Anneal with default hybrid search and monolithic program.md",
        search_strategy="hybrid",
        enhancements={
            "strategy_manifest": False,
            "dual_agent": False,
            "two_phase": False,
            "lineage_context": False,
            "episodic_memory": False,
        },
    ),
    BenchmarkConfig(
        name="treatment",
        description="Anneal with all enhancements enabled",
        search_strategy="hybrid",
        enhancements={
            "strategy_manifest": True,
            "dual_agent": True,
            "two_phase": True,
            "lineage_context": True,
            "episodic_memory": True,
        },
    ),
]

CONFIG_BY_NAME: dict[str, BenchmarkConfig] = {c.name: c for c in BENCHMARK_CONFIGS}


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def build_register_command(run: BenchmarkRun) -> list[str]:
    """Build the anneal register command for a BenchmarkRun.

    Returns a list of string tokens suitable for subprocess.run.
    """
    target = run.target
    target_name = run.target_name

    cmd: list[str] = [
        "uv", "run", "anneal", "register",
        "--name", target_name,
        "--artifact", target.artifact_path,
        "--scope", target.scope_path,
        "--eval-mode", target.eval_mode,
        "--direction", target.direction,
    ]

    if target.eval_mode == "stochastic" and target.criteria_path:
        cmd += ["--criteria", target.criteria_path]

    if target.eval_mode == "deterministic":
        if target.run_cmd:
            cmd += ["--run-cmd", target.run_cmd]
        if target.parse_cmd:
            cmd += ["--parse-cmd", target.parse_cmd]

    return cmd


def build_run_command(run: BenchmarkRun) -> list[str]:
    """Build the anneal run command for a BenchmarkRun.

    Returns a list of string tokens suitable for subprocess.run.
    Raw config does not use anneal run; returns an empty list in that case.
    The caller is responsible for handling the empty-list case.
    """
    if run.config.search_strategy == "none":
        return []

    cmd: list[str] = [
        "uv", "run", "anneal", "run",
        "--target", run.target_name,
        "--experiments", str(run.target.experiment_budget),
        "--yes",  # non-interactive: skip cost confirmation prompt
    ]

    # Search strategy is a run-time flag, not a registration flag
    if run.config.search_strategy not in ("none", "hybrid"):
        cmd += ["--search", run.config.search_strategy]

    return cmd


def format_command(tokens: list[str]) -> str:
    """Format a command token list as a shell-quoted string for display."""
    import shlex
    return " ".join(shlex.quote(t) for t in tokens)


# ---------------------------------------------------------------------------
# Run matrix construction
# ---------------------------------------------------------------------------


def build_run_matrix(
    targets: list[BenchmarkTarget],
    configs: list[BenchmarkConfig],
    seeds: list[int],
    output_dir: Path,
) -> list[BenchmarkRun]:
    """Build all BenchmarkRun objects for the given targets, configs, and seeds.

    Returns runs ordered by seed first, then target, then config — matching
    the execution plan in the implementation document.
    """
    runs: list[BenchmarkRun] = []
    for seed in seeds:
        for target in targets:
            for config in configs:
                runs.append(
                    BenchmarkRun(
                        target=target,
                        config=config,
                        seed=seed,
                        output_dir=output_dir,
                    )
                )
    return runs


# ---------------------------------------------------------------------------
# Dry-run preview
# ---------------------------------------------------------------------------


def print_dry_run(runs: list[BenchmarkRun]) -> None:
    """Print all register and run commands without executing anything."""
    table = Table(
        title=f"Benchmark suite — {len(runs)} runs",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Run ID", style="dim", no_wrap=True)
    table.add_column("Register command")
    table.add_column("Run command")

    for run in runs:
        reg_cmd = build_register_command(run)
        run_cmd = build_run_command(run)
        table.add_row(
            run.run_id,
            format_command(reg_cmd),
            format_command(run_cmd) if run_cmd else "[dim](raw — no run)[/dim]",
        )

    console.print(table)
    console.print(
        f"\nTotal: {len(runs)} runs "
        f"({len({r.target_name for r in runs})} unique targets "
        f"x {len({r.seed for r in runs})} seeds)"
    )


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------


def _collect_result(run: BenchmarkRun, stdout: str, elapsed_seconds: float) -> dict[str, object]:
    """Parse anneal run stdout into a result record.

    Stores raw stdout alongside parsed metrics so that post-processing
    can re-derive any field without re-running the experiment.
    """
    record: dict[str, object] = {
        "run_id": run.run_id,
        "target_id": run.target.id,
        "config_name": run.config.name,
        "seed": run.seed,
        "elapsed_seconds": elapsed_seconds,
        "raw_stdout": stdout,
        "final_score": None,
        "total_cost_usd": None,
        "mutation_acceptance_rate": None,
        "convergence_speed": None,
        "experiment_records": [],
    }

    # Parse final score from anneal run output.
    # anneal run prints "best_score=<float>" on the final summary line.
    for line in stdout.splitlines():
        if "best_score=" in line:
            try:
                record["final_score"] = float(line.split("best_score=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        if "total_cost=" in line or "cost_usd=" in line:
            key = "total_cost=" if "total_cost=" in line else "cost_usd="
            try:
                record["total_cost_usd"] = float(line.split(key)[1].split()[0].rstrip("$"))
            except (IndexError, ValueError):
                pass
        if "acceptance_rate=" in line:
            try:
                record["mutation_acceptance_rate"] = float(
                    line.split("acceptance_rate=")[1].split()[0].rstrip("%")
                ) / 100.0
            except (IndexError, ValueError):
                pass

    return record


def write_result(run: BenchmarkRun, record: dict[str, object]) -> None:
    """Append a result record as a JSONL line to the run's result file."""
    run.result_path.parent.mkdir(parents=True, exist_ok=True)
    with run.result_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------


def _execute_run(run: BenchmarkRun, dry_run: bool = False) -> dict[str, object]:
    """Register and execute a single BenchmarkRun.

    For dry_run=True, prints commands and returns a placeholder record.
    Returns the result record dict.
    """
    reg_cmd = build_register_command(run)
    run_cmd = build_run_command(run)

    if dry_run:
        console.print(f"[dim]DRY-RUN {run.run_id}:[/dim]")
        console.print(f"  {format_command(reg_cmd)}")
        if run_cmd:
            console.print(f"  {format_command(run_cmd)}")
        return {"run_id": run.run_id, "dry_run": True}

    # Step 1: Register target (idempotent — re-registration is safe)
    console.print(f"  [cyan]register[/cyan] {run.target_name}")
    reg_result = subprocess.run(
        reg_cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if reg_result.returncode != 0:
        console.print(f"  [red]register failed for {run.run_id}:[/red]\n{reg_result.stderr}")
        return {
            "run_id": run.run_id,
            "error": f"register failed (rc={reg_result.returncode})",
            "stderr": reg_result.stderr,
        }

    # Step 2: Raw baseline — no anneal run, just collect the baseline score
    if not run_cmd:
        console.print(f"  [yellow]raw baseline[/yellow] {run.run_id} — skipping anneal run")
        record = _collect_result(run, reg_result.stdout, elapsed_seconds=0.0)
        write_result(run, record)
        return record

    # Step 3: Run anneal
    console.print(f"  [green]run[/green] {run.run_id}")
    start = time.monotonic()
    anneal_result = subprocess.run(
        run_cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start

    if anneal_result.returncode != 0:
        console.print(f"  [red]run failed for {run.run_id}:[/red]\n{anneal_result.stderr}")
        record = {
            "run_id": run.run_id,
            "error": f"anneal run failed (rc={anneal_result.returncode})",
            "stderr": anneal_result.stderr,
            "elapsed_seconds": elapsed,
        }
    else:
        record = _collect_result(run, anneal_result.stdout, elapsed_seconds=elapsed)

    write_result(run, record)
    return record


def execute_runs(
    runs: list[BenchmarkRun],
    dry_run: bool = False,
    parallel: int = 1,
) -> list[dict[str, object]]:
    """Execute all benchmark runs, optionally in parallel.

    Args:
        runs: Ordered list of BenchmarkRun objects to execute.
        dry_run: When True, print commands without executing.
        parallel: Maximum number of concurrent workers. 1 = sequential.

    Returns:
        List of result records in completion order (not submission order
        when parallel > 1).
    """
    results: list[dict[str, object]] = []

    if parallel <= 1:
        for run in runs:
            results.append(_execute_run(run, dry_run=dry_run))
        return results

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_run = {
            executor.submit(_execute_run, run, dry_run): run for run in runs
        }
        for future in as_completed(future_to_run):
            run = future_to_run[future]
            try:
                record = future.result()
                results.append(record)
            except Exception as exc:
                console.print(f"  [red]Exception in {run.run_id}:[/red] {exc}")
                results.append({"run_id": run.run_id, "error": str(exc)})

    return results
