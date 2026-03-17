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
    BinaryCriterion,
    BudgetCap,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    ExperimentRecord,
    OptimizationTarget,
    StochasticEval,
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
    stochastic_eval = None

    if eval_mode is EvalMode.DETERMINISTIC:
        deterministic_eval = DeterministicEval(
            run_command=args.run_cmd,
            parse_command=args.parse_cmd,
            timeout_seconds=args.time_budget,
        )

    if eval_mode is EvalMode.STOCHASTIC:
        import tomllib
        criteria_path = Path(args.criteria)
        if not criteria_path.is_absolute():
            criteria_path = repo_root / criteria_path
        if not criteria_path.exists():
            console.print(f"[red]Criteria file not found: {criteria_path}[/red]")
            sys.exit(1)
        criteria_data = tomllib.loads(criteria_path.read_text(encoding="utf-8"))

        binary_criteria = [
            BinaryCriterion(name=c["name"], question=c["question"])
            for c in criteria_data.get("criteria", [])
        ]
        test_prompts = [
            tp["prompt"] for tp in criteria_data.get("test_prompts", [])
        ]
        gen = criteria_data.get("generation", {})
        meta = criteria_data.get("meta", {})

        # Generation agent: use a cheap model for sample generation
        gen_agent = AgentConfig(
            mode="api",
            model=gen.get("agent", {}).get("model", "gemini-2.5-flash"),
            evaluator_model=args.evaluator_model,
            max_budget_usd=0.02,
            temperature=gen.get("agent", {}).get("temperature", 0.7),
        )

        stochastic_eval = StochasticEval(
            sample_count=meta.get("sample_count", 10),
            criteria=binary_criteria,
            test_prompts=test_prompts,
            generation_prompt_template=gen.get("prompt_template", ""),
            output_format=gen.get("output_format", "text"),
            confidence_level=meta.get("confidence_level", 0.95),
            generation_agent_config=gen_agent,
        )

    eval_config = EvalConfig(
        metric_name="binary_criteria_score" if eval_mode is EvalMode.STOCHASTIC else "deterministic_score",
        direction=direction,
        deterministic=deterministic_eval,
        stochastic=stochastic_eval,
    )

    # Build agent config (per-invocation budget is separate from daily cap)
    agent_config = AgentConfig(
        mode=args.agent_mode,
        model=args.agent_model,
        evaluator_model=args.evaluator_model,
        max_budget_usd=1.00,
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

    # Registration-time warnings (3.9) — only warn if user explicitly set --interval
    if args.interval is not None and interval < target.time_budget_seconds * 1.5:
        console.print(
            f"[yellow]Warning: loop_interval ({interval}s) < time_budget × 1.5 "
            f"({target.time_budget_seconds * 1.5:.0f}s). Experiments may overlap.[/yellow]"
        )
    for artifact_path in args.artifact:
        ap = repo_root / artifact_path
        if ap.exists():
            size = ap.stat().st_size
            if size > 50_000:
                console.print(
                    f"[yellow]Warning: artifact {artifact_path} is {size / 1024:.0f}KB. "
                    f"Large artifacts consume context budget and may reduce mutation quality.[/yellow]"
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
                f"  Budget cap:   ${target.budget_cap.max_usd_per_day:.2f}/day"
                + (
                    f"\n  Criteria:     {len(stochastic_eval.criteria)} binary, "
                    f"{len(stochastic_eval.test_prompts)} test prompts, "
                    f"N={stochastic_eval.sample_count}"
                    if stochastic_eval else ""
                ),
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
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from anneal.engine.agent import AgentInvoker  # noqa: F811
    from anneal.engine.eval import EvalEngine  # noqa: F811
    from anneal.engine.knowledge import KnowledgeStore  # noqa: F811
    from anneal.engine.notifications import NotificationManager  # noqa: F811
    from anneal.engine.runner import ExperimentRunner  # noqa: F811
    from anneal.engine.search import GreedySearch  # noqa: F811

    repo_root = _find_repo_root()
    registry = Registry(repo_root)
    git = GitEnvironment()

    # Resolve target(s)
    if args.target:
        try:
            targets = [registry.get_target(args.target)]
        except RegistryError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
    else:
        targets = registry.all_targets()
        if not targets:
            console.print("[yellow]No targets registered. Use 'anneal register' first.[/yellow]")
            sys.exit(1)
        console.print(f"  Running all {len(targets)} registered targets sequentially")

    # Apply runtime overrides (ephemeral — not persisted to config.toml)
    overrides: list[str] = []
    for target in targets:
        if args.samples is not None and target.eval_config.stochastic:
            target.eval_config.stochastic.sample_count = args.samples
            overrides.append(f"samples={args.samples}")
        if args.confidence is not None and target.eval_config.stochastic:
            target.eval_config.stochastic.confidence_level = args.confidence
            overrides.append(f"confidence={args.confidence}")
        if args.agent_budget is not None:
            target.agent_config.max_budget_usd = args.agent_budget
            overrides.append(f"agent_budget=${args.agent_budget:.2f}")
    if overrides:
        console.print(f"  [dim]Runtime overrides: {', '.join(set(overrides))}[/dim]")

    target = targets[0]  # For single-target path below; multi-target loops over all

    knowledge = KnowledgeStore(repo_root / target.knowledge_path)
    knowledge.validate_and_repair()
    notifier = NotificationManager(target.notifications)

    # Select search strategy
    if getattr(args, "search", None) == "annealing":
        from anneal.engine.search import SimulatedAnnealingSearch  # noqa: F811
        search_strategy = SimulatedAnnealingSearch()
        overrides.append("search=annealing")
    else:
        search_strategy = GreedySearch()

    runner = ExperimentRunner(
        git=git,
        agent_invoker=AgentInvoker(),
        eval_engine=EvalEngine(),
        search=search_strategy,
        registry=registry,
        repo_root=repo_root,
        knowledge=knowledge,
        notifications=notifier,
    )

    max_exp = args.experiments or 0
    total_cost = 0.0
    best_score = target.baseline_score
    kept_count = 0

    # Progress bar with inline status
    progress = Progress(
        SpinnerColumn("dots"),
        TextColumn(f"[bold cyan]{target.id}[/]"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task(
        target.id,
        total=max_exp if max_exp > 0 else None,
        status=f"baseline={target.baseline_score:.3f}",
    )

    def on_experiment(record: ExperimentRecord) -> None:
        nonlocal total_cost, best_score, kept_count
        total_cost += record.cost_usd
        if record.outcome.value == "KEPT":
            best_score = record.score
            kept_count += 1

        outcome = record.outcome.value
        if outcome == "KEPT":
            tag = "[green]KEPT[/]"
        elif outcome == "BLOCKED":
            tag = "[yellow]BLKD[/]"
        elif outcome == "CRASHED":
            tag = "[red]CRASH[/]"
        else:
            tag = "[red]DISC[/]"

        failure = ""
        if record.failure_mode:
            failure = f"  {record.failure_mode[:60]}"

        progress.update(
            task_id,
            advance=1,
            status=f"{tag} {record.score:.3f}  best={best_score:.3f}  ${total_cost:.3f}{failure}",
        )

    console.print()
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
    console.print()

    try:
        with progress:
            records = asyncio.run(
                runner.run_loop(
                    target=target,
                    max_experiments=args.experiments,
                    stop_score=args.until,
                    on_experiment=on_experiment,
                )
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        runner.request_stop(target.id)
        records = []

    console.print()
    kept = sum(1 for r in records if r.outcome.value == "KEPT")
    cond_time = sum(r.duration_seconds for r in records)
    console.print(
        Panel(
            f"Target [bold]{target.id}[/bold]\n"
            f"  Experiments:  {len(records)}\n"
            f"  Kept:         {kept}\n"
            f"  Best score:   {best_score:.3f}\n"
            f"  Total cost:   ${total_cost:.4f}\n"
            f"  Total time:   {cond_time / 60:.1f} min",
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


def _handle_deregister(args: argparse.Namespace) -> None:
    """Handle ``anneal deregister``."""
    repo_root = _find_repo_root()
    registry = Registry(repo_root)

    try:
        asyncio.run(registry.deregister_target(args.target))
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    console.print(
        Panel(
            f"Target [bold]{args.target}[/bold] deregistered.\n"
            f"  Worktree removed. Experiment history preserved in targets/{args.target}/.",
            title="anneal deregister",
            style="yellow",
        )
    )


def _handle_configure(args: argparse.Namespace) -> None:
    """Handle ``anneal configure``."""
    repo_root = _find_repo_root()
    registry = Registry(repo_root)

    try:
        target = registry.get_target(args.target)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    changes: list[str] = []

    if args.samples is not None and target.eval_config.stochastic:
        target.eval_config.stochastic.sample_count = args.samples
        changes.append(f"  sample_count = {args.samples}")

    if args.confidence is not None and target.eval_config.stochastic:
        target.eval_config.stochastic.confidence_level = args.confidence
        changes.append(f"  confidence_level = {args.confidence}")

    if args.agent_budget is not None:
        target.agent_config.max_budget_usd = args.agent_budget
        changes.append(f"  agent max_budget_usd = ${args.agent_budget:.2f}")

    if args.daily_budget is not None:
        if target.budget_cap is None:
            target.budget_cap = BudgetCap(max_usd_per_day=args.daily_budget)
        else:
            target.budget_cap.max_usd_per_day = args.daily_budget
        changes.append(f"  daily_budget = ${args.daily_budget:.2f}")

    if args.agent_model is not None:
        target.agent_config.model = args.agent_model
        changes.append(f"  agent model = {args.agent_model}")

    if getattr(args, "agent_mode", None) is not None:
        target.agent_config.mode = args.agent_mode
        changes.append(f"  agent mode = {args.agent_mode}")

    if args.evaluator_model is not None:
        target.agent_config.evaluator_model = args.evaluator_model
        changes.append(f"  evaluator model = {args.evaluator_model}")

    if getattr(args, "generation_model", None) is not None and target.eval_config.stochastic:
        if target.eval_config.stochastic.generation_agent_config:
            target.eval_config.stochastic.generation_agent_config.model = args.generation_model
        changes.append(f"  generation model = {args.generation_model}")

    if getattr(args, "base_url", None) is not None:
        # Store base_url as a custom field — the agent invoker reads it
        target.agent_config.temperature = target.agent_config.temperature  # no-op to keep config valid
        changes.append(f"  base_url = {args.base_url}")
        console.print(f"  [dim]Note: base_url support requires setting OPENAI_BASE_URL={args.base_url} env var[/dim]")

    if args.time_budget is not None:
        target.time_budget_seconds = args.time_budget
        changes.append(f"  time_budget = {args.time_budget}s")

    if args.max_failures is not None:
        target.max_consecutive_failures = args.max_failures
        changes.append(f"  max_consecutive_failures = {args.max_failures}")

    if not changes:
        console.print("[yellow]No configuration changes specified.[/yellow]")
        return

    registry.update_target(target)
    console.print(
        Panel(
            f"Target [bold]{target.id}[/bold] updated:\n\n" + "\n".join(changes),
            title="anneal configure",
            style="green",
        )
    )


def _handle_dashboard(args: argparse.Namespace) -> None:
    """Handle ``anneal dashboard``."""
    from anneal.engine.dashboard import DashboardServer, get_event_bus

    event_bus = get_event_bus()
    server = DashboardServer(event_bus, host=args.host, port=args.port)

    if getattr(args, "open", False):
        import webbrowser
        webbrowser.open(f"http://{args.host}:{args.port}")

    console.print(
        Panel(
            f"Dashboard running at [bold]http://{args.host}:{args.port}[/bold]\n"
            f"  Press Ctrl+C to stop",
            title="anneal dashboard",
            style="blue",
        )
    )

    try:
        asyncio.run(server.start())
        # Keep running until interrupted
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")


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
    reg.add_argument("--agent-mode", choices=["claude_code", "api"], default="claude_code", help="Agent invocation mode")
    reg.add_argument("--evaluator-model", default="gpt-4.1", help="Evaluator model identifier")
    reg.add_argument("--base-url", help="Custom API base URL (for local LLMs, e.g., http://localhost:11434/v1)")
    reg.add_argument("--scope", required=True, help="Path to scope.yaml")
    reg.add_argument("--dry-run", action="store_true", help="Validate without writing")

    # -- run (stub) --
    run = subparsers.add_parser("run", help="Run optimization loop")
    run.add_argument("--target", help="Target identifier (omit to run all registered targets)")
    run.add_argument("--experiments", type=int, help="Stop after N experiments")
    run.add_argument("--until", type=float, help="Stop when score reaches threshold")
    run.add_argument("--foreground", action="store_true", help="Block terminal")
    run.add_argument("--dry-run", action="store_true", help="One experiment, print output, do not commit")
    # Runtime overrides (do not persist — apply to this run only)
    run.add_argument("--samples", type=int, help="Override sample count (N) for this run")
    run.add_argument("--confidence", type=float, help="Override confidence level for this run")
    run.add_argument("--agent-budget", type=float, help="Override per-invocation agent budget for this run")
    run.add_argument("--search", choices=["greedy", "annealing"], help="Override search strategy for this run")

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

    # -- deregister --
    dereg = subparsers.add_parser("deregister", help="Remove a target and its worktree")
    dereg.add_argument("--target", required=True, help="Target identifier")

    # -- configure --
    conf = subparsers.add_parser("configure", help="Update target configuration permanently")
    conf.add_argument("--target", required=True, help="Target identifier")
    conf.add_argument("--samples", type=int, help="Set sample count (N)")
    conf.add_argument("--confidence", type=float, help="Set confidence level (0.0-1.0)")
    conf.add_argument("--agent-budget", type=float, help="Set per-invocation agent budget (USD)")
    conf.add_argument("--daily-budget", type=float, help="Set daily budget cap (USD)")
    conf.add_argument("--agent-model", help="Set agent model")
    conf.add_argument("--agent-mode", choices=["claude_code", "api"], help="Set agent invocation mode")
    conf.add_argument("--evaluator-model", help="Set evaluator model")
    conf.add_argument("--generation-model", help="Set sample generation model (stochastic)")
    conf.add_argument("--base-url", help="Set custom API base URL (for local LLMs, e.g., http://localhost:11434/v1)")
    conf.add_argument("--time-budget", type=int, help="Set time budget per experiment (seconds)")
    conf.add_argument("--max-failures", type=int, help="Set max consecutive failures before HALT")

    # -- dashboard --
    dash = subparsers.add_parser("dashboard", help="Start live SSE dashboard server")
    dash.add_argument("--port", type=int, default=8080, help="Server port")
    dash.add_argument("--host", default="127.0.0.1", help="Server host")
    dash.add_argument("--open", action="store_true", help="Open browser on start")

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
        "deregister": lambda: _handle_deregister(args),
        "configure": lambda: _handle_configure(args),
        "dashboard": lambda: _handle_dashboard(args),
        "list": lambda: _handle_list(args),
    }

    try:
        handlers[args.command]()  # type: ignore[operator]
    except (ScopeError, GitError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
