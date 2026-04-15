"""Benchmark suite orchestration engine.

Responsible for:
  1. Building the full matrix of BenchmarkRun objects from targets and configs.
  2. Generating anneal register and anneal run shell commands for each run.
  3. Executing commands via subprocess (or previewing them in dry-run mode).
  4. Collecting experiment records from anneal's internal state into JSONL files.
  5. Supporting parallel execution with configurable concurrency.

Result files contain raw ExperimentRecord objects (one JSON line per experiment),
matching the format produced by the anneal engine's runner. The analysis pipeline
(benchmarks/analysis/loader.py) reads these records directly.

Usage: uv run python benchmarks/suite/run_suite.py --dry-run
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.table import Table

from anneal.engine.display import LiveProgressMonitor, OutputMode, build_run_summary
from benchmarks.suite.config import BenchmarkConfig, BenchmarkRun, BenchmarkTarget

console = Console()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_SUITE_DIR = Path(__file__).parent
_REPO_ROOT = _SUITE_DIR.parent.parent
_ANNEAL_DIR = _REPO_ROOT / ".anneal"

# Three-model split used for all benchmark runs.
#
#   _MUTATION_MODEL:    primary mutation agent (AgentConfig.model)
#   _DIAGNOSIS_MODEL:   two-phase diagnosis, dual-agent exploration arm,
#                       research operator, and policy rewriter
#   _JUDGE_MODEL:       stochastic-eval LLM judge (and deterministic
#                       evaluator_model slot, since the CLI routes both
#                       through --evaluator-model)
#
# Pricing for all three models must be defined in anneal/engine/client.py
# (_load_pricing) or ~/.anneal/pricing.toml before cost tracking is accurate.
_MUTATION_MODEL = "gpt-5.4"
_DIAGNOSIS_MODEL = "gpt-5.4-mini"
_JUDGE_MODEL = "gemini-3-flash-preview"

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


def _resolve_repo_root_placeholder(value: str) -> str:
    """Replace {repo_root} with the absolute repo root path.

    This allows eval commands to reference the main repo's harness files
    via absolute path, preventing the optimization agent from reading
    harness source in the worktree.
    """
    return value.replace("{repo_root}", str(_REPO_ROOT))


def build_register_command(run: BenchmarkRun) -> list[str]:
    """Build the anneal register command for a BenchmarkRun.

    Returns a list of string tokens suitable for subprocess.run.
    """
    target = run.target
    target_name = run.target_name

    cmd: list[str] = [
        "uv",
        "run",
        "anneal",
        "register",
        "--name",
        target_name,
        "--artifact",
        target.artifact_path,
        "--scope",
        target.scope_path,
        "--eval-mode",
        target.eval_mode,
        "--direction",
        target.direction,
    ]

    if target.eval_mode == "stochastic" and target.criteria_path:
        cmd += ["--criteria", target.criteria_path]

    if target.eval_mode == "deterministic":
        if target.run_cmd:
            cmd += ["--run-cmd", _resolve_repo_root_placeholder(target.run_cmd)]
        if target.parse_cmd:
            cmd += ["--parse-cmd", _resolve_repo_root_placeholder(target.parse_cmd)]

    # Three-model split (see module constants). The CLI only exposes three of
    # the six model slots; the remaining three (exploration_model,
    # diagnosis_model, research_config.model) are patched into config.toml
    # after registration by _patch_model_config().
    cmd += [
        "--agent-model", _MUTATION_MODEL,
        "--agent-mode", "api",
        "--evaluator-model", _JUDGE_MODEL,
        "--policy-model", _DIAGNOSIS_MODEL,
    ]
    if target.eval_mode == "stochastic":
        cmd += ["--judgment-model", _JUDGE_MODEL]

    # Set budget high enough for the full experiment budget to complete
    # without pausing. Default $5/day would stall after ~5 experiments.
    cmd += ["--max-budget-usd", "50"]

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
        "uv",
        "run",
        "anneal",
        "run",
        "--target",
        run.target_name,
        "--experiments",
        str(run.target.experiment_budget),
        "--yes",  # non-interactive: skip cost confirmation prompt
    ]

    # Search strategy is a run-time flag, not a registration flag
    if run.config.search_strategy not in ("none", "hybrid"):
        cmd += ["--search", run.config.search_strategy]

    # Early-stop on perfect score for deterministic targets with a known
    # ceiling (e.g., B4 composite score 0.0–1.0).  Stochastic targets use
    # raw criterion pass counts (e.g., 5 criteria × 10 samples = max 50),
    # so --until 1.0 would trigger after the first improvement and must be
    # skipped — the experiment budget is the only stopping condition.
    if run.target.direction == "maximize" and run.target.eval_mode == "deterministic":
        cmd += ["--until", "1.0"]

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


def _read_experiment_records(target_name: str) -> list[dict[str, object]]:
    """Read experiment records from anneal's internal state.

    Returns the parsed ExperimentRecord dicts from
    ``.anneal/targets/<target_name>/experiments.jsonl``.
    """
    experiments_path = _ANNEAL_DIR / "targets" / target_name / "experiments.jsonl"
    if not experiments_path.exists():
        return []

    records: list[dict[str, object]] = []
    for line in experiments_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _patch_model_config(target_name: str) -> None:
    """Fill in the three non-CLI model slots in ``.anneal/config.toml``.

    After ``anneal register`` writes the target section, three ``agent_config``
    fields are still empty strings because the CLI does not expose them:

      - ``exploration_model``  (dual-agent exploration arm)
      - ``diagnosis_model``    (two-phase diagnosis)

    This function rewrites those two lines inside the
    ``[targets.<name>.agent_config]`` section, scoping the replacement to
    that section only so sibling targets are not touched. If the target's
    research_config is present, its ``model`` field is also filled in.
    """
    config_path = _ANNEAL_DIR / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"anneal config not found at {config_path}")

    lines = config_path.read_text(encoding="utf-8").splitlines()
    agent_section_header = f"[targets.{target_name}.agent_config]"
    research_section_header = f"[targets.{target_name}.research_config]"

    current_section: str | None = None
    patched_agent = {"exploration_model": False, "diagnosis_model": False}
    patched_research = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped
            continue

        if current_section == agent_section_header:
            if stripped.startswith("exploration_model"):
                lines[i] = f'exploration_model = "{_DIAGNOSIS_MODEL}"'
                patched_agent["exploration_model"] = True
            elif stripped.startswith("diagnosis_model"):
                lines[i] = f'diagnosis_model = "{_DIAGNOSIS_MODEL}"'
                patched_agent["diagnosis_model"] = True
        elif current_section == research_section_header:
            if stripped.startswith("model"):
                lines[i] = f'model = "{_DIAGNOSIS_MODEL}"'
                patched_research = True

    if not all(patched_agent.values()):
        missing = [k for k, v in patched_agent.items() if not v]
        raise RuntimeError(
            f"Failed to patch agent_config fields {missing} for target "
            f"{target_name}: section {agent_section_header} not found or "
            f"fields missing"
        )

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    research_note = " + research_config.model" if patched_research else ""
    console.print(
        f"    [dim]patched agent_config.exploration_model, "
        f"diagnosis_model{research_note} → {_DIAGNOSIS_MODEL}[/dim]"
    )


def _read_baseline_score(target_name: str) -> float | None:
    """Read the baseline score from anneal's target config.

    Parses ``.anneal/config.toml`` for the target's ``baseline_score`` field.
    """
    config_path = _ANNEAL_DIR / "config.toml"
    if not config_path.exists():
        return None

    # Simple TOML value extraction — avoids adding tomllib dependency
    # for a single field read. Targets are under [targets.<name>].
    in_target_section = False
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == f"[targets.{target_name}]":
            in_target_section = True
            continue
        if in_target_section:
            if stripped.startswith("["):
                break
            if stripped.startswith("baseline_score"):
                parts = stripped.split("=", 1)
                if len(parts) == 2:
                    try:
                        return float(parts[1].strip())
                    except ValueError:
                        return None
    return None


def write_result(run: BenchmarkRun, records: list[dict[str, object]]) -> None:
    """Write experiment records as JSONL to the run's result file.

    Each record is one ExperimentRecord dict, written as a separate JSON line.
    The analysis loader expects this format.
    """
    run.result_path.parent.mkdir(parents=True, exist_ok=True)
    with run.result_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------


def _execute_run(run: BenchmarkRun, dry_run: bool = False) -> dict[str, object]:
    """Register and execute a single BenchmarkRun.

    For dry_run=True, prints commands and returns a placeholder record.

    Returns a summary dict with run_id and status. Experiment-level data is
    written to the result JSONL via write_result.
    """
    reg_cmd = build_register_command(run)
    run_cmd = build_run_command(run)

    if dry_run:
        console.print(f"[dim]DRY-RUN {run.run_id}:[/dim]")
        console.print(f"  {format_command(reg_cmd)}")
        if run_cmd:
            console.print(f"  {format_command(run_cmd)}")
        return {"run_id": run.run_id, "dry_run": True}

    # Step 1: Clean slate — deregister, delete branch, clear state, register fresh.
    console.print(f"  [cyan]register[/cyan] {run.target_name}")
    subprocess.run(
        ["uv", "run", "anneal", "deregister", "--target", run.target_name],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )  # ignore errors — target may not exist yet
    subprocess.run(
        ["git", "branch", "-D", f"anneal/{run.target_name}"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )  # ignore errors — branch may not exist
    # Clear stale experiment/hypothesis records from prior runs.
    # Deregister removes config but leaves knowledge files behind.
    target_state_dir = _ANNEAL_DIR / "targets" / run.target_name
    for stale_file in ("experiments.jsonl", "hypotheses.jsonl", ".loop-state.json"):
        stale_path = target_state_dir / stale_file
        if stale_path.exists():
            stale_path.unlink()

    reg_result = subprocess.run(
        reg_cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if reg_result.returncode != 0:
        console.print(
            f"  [red]register failed for {run.run_id}:[/red]\n{reg_result.stderr}"
        )
        return {
            "run_id": run.run_id,
            "error": f"register failed (rc={reg_result.returncode})",
            "stderr": reg_result.stderr,
        }

    # Step 1a: Patch the three model slots the CLI does not expose
    # (exploration_model, diagnosis_model, research_config.model).
    _patch_model_config(run.target_name)

    # Step 1b: Scrub eval harness files from the worktree so the optimization
    # agent cannot read the hidden test suite.  The eval commands reference
    # the main repo's harness via absolute path (see {repo_root} placeholder).
    worktree_dir = _ANNEAL_DIR / "worktrees" / run.target_name
    if worktree_dir.is_dir():
        harness_dir = worktree_dir / "benchmarks" / "suite" / "harness"
        if harness_dir.is_dir():
            shutil.rmtree(harness_dir)
            console.print("    [dim]scrubbed harness from worktree[/dim]")

    # Step 2: Raw baseline — no optimization, just record the baseline score.
    if not run_cmd:
        console.print(f"  [yellow]raw baseline[/yellow] {run.run_id}")
        baseline = _read_baseline_score(run.target_name)
        if baseline is not None:
            # Synthesize a single experiment record representing the unoptimized baseline.
            baseline_record: dict[str, object] = {
                "id": f"{run.run_id}-baseline",
                "target_id": run.target_name,
                "score": baseline,
                "baseline_score": baseline,
                "outcome": "BASELINE",
                "cost_usd": 0.0,
                "duration_seconds": 0.0,
            }
            write_result(run, [baseline_record])
            console.print(f"    baseline_score={baseline}")
            return {"run_id": run.run_id, "baseline_score": baseline}

        console.print(f"  [red]no baseline score found for {run.target_name}[/red]")
        return {"run_id": run.run_id, "error": "no baseline score after registration"}

    # Step 3: Run anneal optimization with live progress monitoring.
    console.print(f"  [green]run[/green] {run.run_id}")
    experiments_path = _ANNEAL_DIR / "targets" / run.target_name / "experiments.jsonl"
    monitor = LiveProgressMonitor(
        experiments_path,
        run_label=run.run_id,
        console=console,
        max_experiments=run.target.experiment_budget,
    )
    monitor.start()
    start = time.monotonic()
    anneal_result = subprocess.run(
        run_cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start
    records = monitor.stop()

    if anneal_result.returncode != 0:
        console.print(
            f"  [red]run failed for {run.run_id}:[/red]\n{anneal_result.stderr}"
        )
        # Still save any records collected before failure.
        if records:
            write_result(run, records)
        return {
            "run_id": run.run_id,
            "error": f"anneal run failed (rc={anneal_result.returncode})",
            "stderr": anneal_result.stderr,
            "elapsed_seconds": elapsed,
        }

    # Step 4: Write collected experiment records.
    if not records:
        # Fallback: monitor may have missed records if JSONL was written after
        # the subprocess exited but before stop() drained. Read directly.
        records = _read_experiment_records(run.target_name)

    if not records:
        console.print(
            f"  [yellow]no experiment records found for {run.run_id}[/yellow]"
        )
        return {
            "run_id": run.run_id,
            "error": "no experiment records",
            "elapsed_seconds": elapsed,
        }

    write_result(run, records)
    summary = build_run_summary(records, mode=OutputMode.JSON)
    assert isinstance(summary, dict)
    console.print(
        f"    {summary['experiment_count']} experiments, {summary['kept_count']} kept, "
        f"best={summary['best_score']:.4f}, elapsed={elapsed:.0f}s"
    )
    return {
        "run_id": run.run_id,
        **summary,
        "elapsed_seconds": elapsed,
    }


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
