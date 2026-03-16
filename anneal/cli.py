"""CLI entry point for the ``anneal`` command.

Implements subcommands via argparse: init, register (fully wired),
and stubs for run, stop, resume, status, history, list.
"""

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from anneal.engine.environment import GitEnvironment, GitError
from anneal.engine.registry import Registry, RegistryError, init_project
from anneal.engine.scope import ScopeError, compute_scope_hash, load_scope, validate_scope
from anneal.engine.types import (
    AgentConfig,
    BudgetCap,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    ExperimentRecord,
    OptimizationTarget,
)

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Repo root discovery
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """Walk up from cwd to find the git repository root (.git directory).

    Raises SystemExit if not in a git repo.
    """
    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        if (directory / ".git").exists():
            return directory
    console.print("[red]Not inside a git repository.[/red]")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _handle_init(_args: argparse.Namespace) -> None:
    """Handle ``anneal init``."""
    repo_root = _find_repo_root()
    try:
        asyncio.run(init_project(repo_root))
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    console.print(
        Panel(
            f"Initialized anneal project at [bold]{repo_root}[/bold]\n"
            f"  Created: {repo_root / 'anneal'}/\n"
            f"  Added worktrees/ to .gitignore",
            title="anneal init",
            style="green",
        )
    )


def _handle_register(args: argparse.Namespace) -> None:
    """Handle ``anneal register``."""
    repo_root = _find_repo_root()

    # Validate eval-mode specific arguments
    eval_mode = EvalMode(args.eval_mode)

    if eval_mode is EvalMode.DETERMINISTIC:
        if not args.run_cmd or not args.parse_cmd:
            console.print(
                "[red]--run-cmd and --parse-cmd are required for deterministic eval mode.[/red]"
            )
            sys.exit(1)
    if eval_mode is EvalMode.STOCHASTIC:
        if not args.criteria:
            console.print(
                "[red]--criteria is required for stochastic eval mode.[/red]"
            )
            sys.exit(1)

    # Resolve scope path
    scope_path = Path(args.scope)
    if not scope_path.is_absolute():
        scope_path = repo_root / scope_path

    # Load and validate scope
    try:
        scope = load_scope(scope_path)
    except ScopeError as exc:
        console.print(f"[red]Scope error: {exc}[/red]")
        sys.exit(1)

    scope_errors = validate_scope(scope, eval_mode)
    if scope_errors:
        console.print("[red]Scope validation failed:[/red]")
        for err in scope_errors:
            console.print(f"  [red]- {err}[/red]")
        sys.exit(1)

    scope_hash = compute_scope_hash(scope_path)

    # Resolve direction
    direction_map = {
        "maximize": Direction.HIGHER_IS_BETTER,
        "minimize": Direction.LOWER_IS_BETTER,
    }
    direction = direction_map[args.direction]

    # Build eval config
    deterministic_eval = None
    if eval_mode is EvalMode.DETERMINISTIC:
        deterministic_eval = DeterministicEval(
            run_command=args.run_cmd,
            parse_command=args.parse_cmd,
            timeout_seconds=args.time_budget,
        )

    eval_config = EvalConfig(
        metric_name="binary_criteria_score" if eval_mode is EvalMode.STOCHASTIC else "deterministic_score",
        direction=direction,
        deterministic=deterministic_eval,
    )

    # Build agent config
    agent_config = AgentConfig(
        mode="claude_code",
        model=args.agent_model,
        evaluator_model=args.evaluator_model,
        max_budget_usd=args.max_budget_usd,
    )

    # Build budget cap
    budget_cap = BudgetCap(max_usd_per_day=args.max_budget_usd)

    # Resolve interval
    interval = args.interval if args.interval is not None else args.time_budget

    # Relative paths for storage
    scope_rel = str(scope_path.relative_to(repo_root))
    target_id = args.name
    knowledge_path = f"targets/{target_id}"
    worktree_path = f"worktrees/{target_id}"
    git_branch = f"anneal/{target_id}"

    target = OptimizationTarget(
        id=target_id,
        domain_tier=DomainTier.SANDBOX,
        artifact_paths=args.artifact,
        scope_path=scope_rel,
        scope_hash=scope_hash,
        eval_mode=eval_mode,
        eval_config=eval_config,
        agent_config=agent_config,
        time_budget_seconds=args.time_budget,
        loop_interval_seconds=interval,
        knowledge_path=knowledge_path,
        worktree_path=worktree_path,
        git_branch=git_branch,
        baseline_score=0.0,
        budget_cap=budget_cap,
    )

    if args.dry_run:
        console.print(
            Panel(
                f"[bold]Dry run[/bold] — target validated, nothing written.\n\n"
                f"  ID:           {target.id}\n"
                f"  Artifacts:    {target.artifact_paths}\n"
                f"  Eval mode:    {target.eval_mode.value}\n"
                f"  Direction:    {target.eval_config.direction.value}\n"
                f"  Scope:        {target.scope_path}\n"
                f"  Worktree:     {target.worktree_path}\n"
                f"  Branch:       {target.git_branch}\n"
                f"  Time budget:  {target.time_budget_seconds}s\n"
                f"  Budget cap:   ${target.budget_cap.max_usd_per_day:.2f}/day",
                title="anneal register --dry-run",
                style="yellow",
            )
        )
        return

    try:
        asyncio.run(Registry(repo_root).register_target(target))
    except (ScopeError, GitError) as exc:
        console.print(f"[red]Registration failed: {exc}[/red]")
        sys.exit(1)

    console.print(
        Panel(
            f"Registered target [bold]{target.id}[/bold]\n\n"
            f"  Worktree:     {repo_root / target.worktree_path}\n"
            f"  Branch:       {target.git_branch}\n"
            f"  Eval mode:    {target.eval_mode.value}\n"
            f"  Baseline:     {target.baseline_score}",
            title="anneal register",
            style="green",
        )
    )


def _handle_run(args: argparse.Namespace) -> None:
    """Handle ``anneal run``."""
    from anneal.engine.agent import AgentInvoker  # noqa: F811
    from anneal.engine.eval import EvalEngine  # noqa: F811
    from anneal.engine.runner import ExperimentRunner  # noqa: F811
    from anneal.engine.search import GreedySearch  # noqa: F811

    repo_root = _find_repo_root()
    registry = Registry(repo_root)
    git = GitEnvironment()

    try:
        target = registry.get_target(args.target)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    runner = ExperimentRunner(
        git=git,
        agent_invoker=AgentInvoker(),
        eval_engine=EvalEngine(),
        search=GreedySearch(),
        registry=registry,
    )

    def on_experiment(record: "ExperimentRecord") -> None:
        style = "green" if record.outcome.value == "KEPT" else "red"
        console.print(
            f"  [{style}]{record.outcome.value}[/{style}] "
            f"score={record.score:.3f} (baseline={record.baseline_score:.3f}) "
            f"${record.cost_usd:.4f} {record.duration_seconds:.1f}s "
            f"— {record.hypothesis[:80] if record.hypothesis else 'no hypothesis'}"
        )

    console.print(
        Panel(
            f"Running target [bold]{target.id}[/bold]\n"
            f"  Eval mode:   {target.eval_mode.value}\n"
            f"  Worktree:    {target.worktree_path}\n"
            f"  Branch:      {target.git_branch}",
            title="anneal run",
            style="blue",
        )
    )

    try:
        records = asyncio.run(
            runner.run_loop(
                target=target,
                max_experiments=args.experiments,
                stop_score=args.until,
                on_experiment=on_experiment,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Current experiment will complete.[/yellow]")
        runner.request_stop(target.id)
        records = []

    kept = sum(1 for r in records if r.outcome.value == "KEPT")
    console.print(
        Panel(
            f"Target [bold]{target.id}[/bold] — {len(records)} experiments, {kept} kept",
            title="anneal run — complete",
            style="green",
        )
    )


def _handle_stop(args: argparse.Namespace) -> None:
    """Handle ``anneal stop``."""
    console.print(f"[yellow]anneal stop[/yellow] is not yet implemented.")
    sys.exit(0)


def _handle_status(args: argparse.Namespace) -> None:
    """Handle ``anneal status``."""
    import json as json_mod

    repo_root = _find_repo_root()
    registry = Registry(repo_root)

    target_id = args.target
    if target_id:
        try:
            target = registry.get_target(target_id)
        except RegistryError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
        targets = [target]
    else:
        targets = registry.all_targets()

    if not targets:
        console.print("[yellow]No targets registered.[/yellow]")
        return

    for target in targets:
        status_path = Path(target.worktree_path) / ".anneal-status"
        if status_path.exists():
            status_data = json_mod.loads(status_path.read_text())
        else:
            status_data = {"state": "UNKNOWN", "last_score": target.baseline_score, "experiment_count": 0}

        if getattr(args, "json", False):
            console.print(json_mod.dumps(status_data, indent=2))
        else:
            console.print(
                f"  [bold]{target.id}[/bold]  "
                f"score={status_data.get('last_score', '?')}  "
                f"experiments={status_data.get('experiment_count', 0)}  "
                f"state={status_data.get('state', 'UNKNOWN')}"
            )


def _handle_list(_args: argparse.Namespace) -> None:
    """Handle ``anneal list``."""
    repo_root = _find_repo_root()
    registry = Registry(repo_root)
    targets = registry.all_targets()

    if not targets:
        console.print("[yellow]No targets registered.[/yellow]")
        return

    for target in targets:
        console.print(
            f"  [bold]{target.id}[/bold]  "
            f"eval={target.eval_mode.value}  "
            f"score={target.baseline_score:.3f}  "
            f"branch={target.git_branch}"
        )


def _handle_resume(args: argparse.Namespace) -> None:
    """Handle ``anneal resume``."""
    repo_root = _find_repo_root()
    registry = Registry(repo_root)

    try:
        target = registry.get_target(args.target)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    if args.increase_budget and target.budget_cap:
        target.budget_cap.max_usd_per_day += args.increase_budget
        console.print(f"  Budget increased to ${target.budget_cap.max_usd_per_day:.2f}/day")

    if args.reset_failures:
        console.print("  Failure counter will reset on next run")

    registry.update_target(target)
    console.print(
        Panel(
            f"Target [bold]{target.id}[/bold] resumed.\n"
            f"  Run [bold]anneal run --target {target.id}[/bold] to continue.",
            title="anneal resume",
            style="green",
        )
    )


def _handle_reregister(args: argparse.Namespace) -> None:
    """Handle ``anneal re-register``."""
    from anneal.engine.scope import compute_scope_hash as _compute_hash

    repo_root = _find_repo_root()
    registry = Registry(repo_root)

    try:
        target = registry.get_target(args.target)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    scope_path = repo_root / target.scope_path
    new_hash = _compute_hash(scope_path)
    target.scope_hash = new_hash
    registry.update_target(target)

    console.print(
        Panel(
            f"Target [bold]{target.id}[/bold] re-registered.\n"
            f"  Scope hash updated: {new_hash[:16]}...",
            title="anneal re-register",
            style="green",
        )
    )


def _handle_history(args: argparse.Namespace) -> None:
    """Handle ``anneal history``."""
    import json as json_mod
    from anneal.engine.knowledge import KnowledgeStore

    repo_root = _find_repo_root()
    registry = Registry(repo_root)

    try:
        target = registry.get_target(args.target)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    store = KnowledgeStore(repo_root / target.knowledge_path)
    records = store.load_records(limit=args.limit)

    if args.outcome:
        records = [r for r in records if r.outcome.value.lower() == args.outcome.lower()]

    if not records:
        console.print("[yellow]No experiment records found.[/yellow]")
        return

    if getattr(args, "json", False):
        for r in records:
            console.print(json_mod.dumps({
                "id": r.id, "hypothesis": r.hypothesis, "score": r.score,
                "outcome": r.outcome.value, "cost_usd": r.cost_usd,
                "duration_seconds": r.duration_seconds,
            }))
    else:
        for r in records:
            style = "green" if r.outcome.value == "KEPT" else "dim"
            console.print(
                f"  [{style}]{r.outcome.value:9s}[/{style}] "
                f"score={r.score:.3f}  ${r.cost_usd:.4f}  "
                f"{r.hypothesis[:70] if r.hypothesis else '-'}"
            )


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anneal",
        description="Domain-agnostic autonomous optimization framework.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- init --
    subparsers.add_parser("init", help="Initialize anneal project in current git repo")

    # -- register --
    reg = subparsers.add_parser("register", help="Register an optimization target")
    reg.add_argument("--name", required=True, help="Target identifier")
    reg.add_argument("--artifact", required=True, nargs="+", help="Artifact paths relative to repo root")
    reg.add_argument("--eval-mode", required=True, choices=["deterministic", "stochastic"], help="Evaluation strategy")
    reg.add_argument("--criteria", help="Path to eval_criteria.toml (stochastic only)")
    reg.add_argument("--run-cmd", help="Eval run command (deterministic only)")
    reg.add_argument("--parse-cmd", help="Eval parse command (deterministic only)")
    reg.add_argument("--direction", choices=["maximize", "minimize"], default="maximize", help="Optimization direction")
    reg.add_argument("--time-budget", type=int, default=300, help="Time budget in seconds")
    reg.add_argument("--interval", type=int, default=None, help="Loop interval in seconds (default: same as --time-budget)")
    reg.add_argument("--max-budget-usd", type=float, default=5.00, help="Max budget per day in USD")
    reg.add_argument("--agent-model", default="sonnet", help="Agent model identifier")
    reg.add_argument("--evaluator-model", default="gpt-4.1", help="Evaluator model identifier")
    reg.add_argument("--scope", required=True, help="Path to scope.yaml")
    reg.add_argument("--dry-run", action="store_true", help="Validate without writing")

    # -- run (stub) --
    run = subparsers.add_parser("run", help="Run optimization loop")
    run.add_argument("--target", required=True, help="Target identifier")
    run.add_argument("--experiments", type=int, help="Stop after N experiments")
    run.add_argument("--until", type=float, help="Stop when score reaches threshold")
    run.add_argument("--foreground", action="store_true", help="Block terminal")
    run.add_argument("--dry-run", action="store_true", help="One experiment, print output, do not commit")

    # -- stop (stub) --
    stop = subparsers.add_parser("stop", help="Stop optimization loop")
    stop.add_argument("--target", help="Target identifier")
    stop.add_argument("--now", action="store_true", help="Kill immediately instead of graceful stop")

    # -- resume (stub) --
    resume = subparsers.add_parser("resume", help="Resume a paused or halted target")
    resume.add_argument("--target", required=True, help="Target identifier")
    resume.add_argument("--increase-budget", type=float, help="Add dollars to daily cap")
    resume.add_argument("--reset-failures", action="store_true", help="Clear consecutive failure counter")

    # -- status (stub) --
    status = subparsers.add_parser("status", help="Show target status")
    status.add_argument("--target", help="Target identifier")
    status.add_argument("--json", action="store_true", help="Output as JSON")
    status.add_argument("--watch", action="store_true", help="Continuously refresh")

    # -- history (stub) --
    history = subparsers.add_parser("history", help="Show experiment history")
    history.add_argument("--target", required=True, help="Target identifier")
    history.add_argument("--limit", type=int, default=20, help="Number of records to show")
    history.add_argument("--outcome", help="Filter by outcome (kept, discarded, crashed, blocked)")
    history.add_argument("--diff", metavar="EXP_ID", help="Show diff for a specific experiment")
    history.add_argument("--json", action="store_true", help="Output as JSON")

    # -- re-register --
    rereg = subparsers.add_parser("re-register", help="Re-hash scope after manual edits")
    rereg.add_argument("--target", required=True, help="Target identifier")

    # -- list --
    subparsers.add_parser("list", help="List all registered targets")

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point for the ``anneal`` command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers: dict[str, object] = {
        "init": lambda: _handle_init(args),
        "register": lambda: _handle_register(args),
        "run": lambda: _handle_run(args),
        "stop": lambda: _handle_stop(args),
        "resume": lambda: _handle_resume(args),
        "status": lambda: _handle_status(args),
        "history": lambda: _handle_history(args),
        "re-register": lambda: _handle_reregister(args),
        "list": lambda: _handle_list(args),
    }

    try:
        handlers[args.command]()  # type: ignore[operator]
    except (ScopeError, GitError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
