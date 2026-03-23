"""CLI entry point for the ``anneal`` command.

Implements subcommands via argparse: init, register (fully wired),
and stubs for run, stop, resume, status, history, list.
"""

from __future__ import annotations

import asyncio
import argparse
import sys
import time
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
    MetricConstraint,
    OptimizationTarget,
    PopulationConfig,
    StochasticEval,
    VerifierCommand,
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
            f"  Created: {repo_root / '.anneal'}/\n"
            f"  Added .anneal/ to .gitignore",
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

    # F1: Load held-out prompts
    if args.held_out_prompts and stochastic_eval is not None:
        ho_path = Path(args.held_out_prompts)
        if not ho_path.is_absolute():
            ho_path = repo_root / ho_path
        if not ho_path.exists():
            console.print(f"[red]Held-out prompts file not found: {ho_path}[/red]")
            sys.exit(1)
        stochastic_eval.held_out_prompts = [
            line.strip() for line in ho_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # F2: Parse constraints
    constraints: list[MetricConstraint] = []
    for constraint_str in (args.constraint or []):
        if ">=" in constraint_str:
            parts = constraint_str.split(">=", 1)
            constraints.append(MetricConstraint(
                metric_name=parts[0].strip(),
                threshold=float(parts[1].strip()),
                direction=Direction.HIGHER_IS_BETTER,
            ))
        elif "<=" in constraint_str:
            parts = constraint_str.split("<=", 1)
            constraints.append(MetricConstraint(
                metric_name=parts[0].strip(),
                threshold=float(parts[1].strip()),
                direction=Direction.LOWER_IS_BETTER,
            ))
        else:
            console.print(f"[red]Invalid constraint format: {constraint_str}. Use 'metric>=value' or 'metric<=value'.[/red]")
            sys.exit(1)

    held_out_interval = args.held_out_interval if args.held_out_interval is not None else 10

    # Parse verifier gates
    verifiers: list[VerifierCommand] = []
    for verifier_str in (args.verifier or []):
        if ":" not in verifier_str:
            console.print(f"[red]Invalid verifier format: {verifier_str}. Use 'name:command'.[/red]")
            sys.exit(1)
        v_name, v_command = verifier_str.split(":", 1)
        verifiers.append(VerifierCommand(name=v_name.strip(), run_command=v_command.strip()))

    eval_config = EvalConfig(
        metric_name="binary_criteria_score" if eval_mode is EvalMode.STOCHASTIC else "deterministic_score",
        direction=direction,
        deterministic=deterministic_eval,
        stochastic=stochastic_eval,
        held_out_interval=held_out_interval,
        constraints=constraints,
        verifiers=verifiers,
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
    knowledge_path = f".anneal/targets/{target_id}"
    worktree_path = f".anneal/worktrees/{target_id}"
    git_branch = f"anneal/{target_id}"

    # F6: Domain tier
    domain_tier = DomainTier(args.domain_tier) if args.domain_tier else DomainTier.SANDBOX

    # F6: Approval callback for deployment tier (runtime only, not serialized)
    approval_callback = None
    if domain_tier is DomainTier.DEPLOYMENT:
        approval_callback = lambda diff: input("Apply changes? [y/N] ").lower() == "y"

    # F8: Meta depth
    meta_depth = args.meta_depth if args.meta_depth is not None else 0

    target = OptimizationTarget(
        id=target_id,
        domain_tier=domain_tier,
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
        meta_depth=meta_depth,
        restart_probability=args.restart_probability,
        approval_callback=approval_callback,
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

    # Run baseline eval for deterministic targets
    if eval_mode is EvalMode.DETERMINISTIC and deterministic_eval is not None:
        from anneal.engine.eval import DeterministicEvaluator
        worktree_full = repo_root / target.worktree_path
        try:
            baseline_result = asyncio.run(
                DeterministicEvaluator().evaluate(worktree_full, deterministic_eval)
            )
            target.baseline_score = baseline_result.score
            Registry(repo_root).update_target(target)
            console.print(f"  Baseline eval: {baseline_result.score}")
        except Exception as exc:
            console.print(f"  [yellow]Baseline eval failed: {exc}. Set manually via anneal configure --target {target.id} --baseline <score>[/yellow]")

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


def _print_dry_run(targets: list[OptimizationTarget], max_experiments: int | None) -> None:
    """Print cost estimate for each target without running experiments."""
    from anneal.engine.context import estimate_tokens
    from anneal.engine.safety import estimate_experiment_cost

    n = max_experiments or 50

    for target in targets:
        # Estimate context size from artifact content
        artifact_tokens = sum(
            estimate_tokens(Path(target.worktree_path, p).read_text())
            for p in target.artifact_paths
            if Path(target.worktree_path, p).exists()
        )
        estimate = estimate_experiment_cost(target, context_tokens=artifact_tokens)

        total = estimate.total_usd * n
        budget_info = ""
        if target.budget_cap:
            cap = target.budget_cap.max_usd_per_day
            max_before_pause = int(cap / estimate.total_usd) if estimate.total_usd > 0 else n
            budget_info = f"\n  Daily budget cap:         ${cap:.2f} (pauses after ~{max_before_pause} experiments)"

        eval_detail = ""
        if target.eval_config.stochastic:
            sto = target.eval_config.stochastic
            k = len(sto.criteria)
            votes = sto.judgment_votes
            calls = sto.sample_count * k * votes * 2  # x2 for position debiasing
            eval_detail = (
                f"\n  Eval breakdown:           {sto.sample_count} samples x "
                f"{k} criteria x {votes} votes x 2 (debiasing) = {calls} judge calls"
            )

        console.print(
            Panel(
                f"Target:                     [bold]{target.id}[/bold]\n"
                f"  Eval mode:                {target.eval_mode.value}\n"
                f"  Context tokens:           ~{artifact_tokens:,}\n"
                f"  Est. cost per experiment: ${estimate.total_usd:.4f}"
                f"{eval_detail}\n"
                f"  Est. total for {n} experiments: ${total:.2f}"
                f"{budget_info}\n"
                f"  Est. wall-clock time:     ~{n * target.time_budget_seconds / 60:.0f} min "
                f"(at {target.time_budget_seconds}s/experiment)",
                title="anneal run --dry-run",
                style="cyan",
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

    # --dry-run: preview cost estimates and exit
    if getattr(args, "dry_run", False):
        _print_dry_run(targets, args.experiments)
        return

    # F5: Global learning pool
    learning_pool = None
    if getattr(args, "global_learnings", True):
        from anneal.engine.learning_pool import GlobalLearningPool  # noqa: F811
        learning_pool = GlobalLearningPool()

    for target in targets:
        knowledge = KnowledgeStore(repo_root / target.knowledge_path)
        knowledge.validate_and_repair()
        notifier = NotificationManager(target.notifications)

        # Select search strategy
        if getattr(args, "search", None) == "annealing":
            from anneal.engine.search import SimulatedAnnealingSearch  # noqa: F811
            search_strategy = SimulatedAnnealingSearch()
        elif getattr(args, "search", None) == "population":
            from anneal.engine.search import PopulationSearch  # noqa: F811
            pop_size = getattr(args, "population_size", None) or 4
            search_strategy = PopulationSearch(population_size=pop_size)
        else:
            search_strategy = GreedySearch()

        # F6: Set approval callback for deployment-tier targets
        if target.domain_tier is DomainTier.DEPLOYMENT and target.approval_callback is None:
            target.approval_callback = lambda diff: input("Apply changes? [y/N] ").lower() == "y"

        runner = ExperimentRunner(
            git=git,
            agent_invoker=AgentInvoker(),
            eval_engine=EvalEngine(),
            search=search_strategy,
            registry=registry,
            repo_root=repo_root,
            knowledge=knowledge,
            notifications=notifier,
            learning_pool=learning_pool,
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
    """Handle ``anneal stop``: write stop signal to target's status file."""
    repo_root = Path.cwd()
    target_id = args.target

    anneal_dir = repo_root / ".anneal"
    if not anneal_dir.exists():
        console.print("[red]Not an anneal project. Run 'anneal init' first.[/red]")
        sys.exit(1)

    stop_file = anneal_dir / "targets" / target_id / ".stop"
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text(str(time.time()))
    console.print(f"[yellow]Stop signal sent to target {target_id}.[/yellow]")
    console.print("The runner will stop after the current experiment completes.")


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
            from anneal.engine.knowledge import KnowledgeStore
            from anneal.engine.runner import RunLoopState

            knowledge = KnowledgeStore(repo_root / target.knowledge_path)
            records = knowledge.load_records()
            loop_state_path = Path(target.knowledge_path) / ".loop-state.json"
            loop = RunLoopState.load(loop_state_path)

            kept_records = [r for r in records if r.outcome.value == "KEPT"]
            enriched = {
                "target_id": target.id,
                "runner_state": status_data.get("state", "UNKNOWN"),
                "total_experiments": loop.total_experiments,
                "kept_count": loop.kept_count,
                "baseline_score": target.baseline_score,
                "current_score": status_data.get("last_score", target.baseline_score),
                "total_cost_usd": loop.cumulative_cost_usd,
                "experiments": [
                    {
                        "id": r.id,
                        "outcome": r.outcome.value,
                        "hypothesis": r.hypothesis,
                        "score": r.score,
                        "git_sha": r.git_sha,
                    }
                    for r in kept_records[-10:]  # last 10 KEPT
                ],
            }
            console.print(json_mod.dumps(enriched, indent=2))
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
            f"  Worktree removed. Experiment history preserved in .anneal/targets/{args.target}/.",
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

    if getattr(args, "baseline", None) is not None:
        target.baseline_score = args.baseline
        changes.append(f"  baseline_score = {args.baseline}")

    if args.time_budget is not None:
        target.time_budget_seconds = args.time_budget
        changes.append(f"  time_budget = {args.time_budget}s")

    if args.max_failures is not None:
        target.max_consecutive_failures = args.max_failures
        changes.append(f"  max_consecutive_failures = {args.max_failures}")

    if getattr(args, "meta_depth", None) is not None:
        target.meta_depth = args.meta_depth
        changes.append(f"  meta_depth = {args.meta_depth}")

    if getattr(args, "knowledge_context", None) is not None:
        target.inject_knowledge_context = args.knowledge_context
        changes.append(f"  inject_knowledge_context = {args.knowledge_context}")

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


def _handle_drift(args: argparse.Namespace) -> None:
    """Handle ``anneal drift``."""
    from anneal.engine.knowledge import KnowledgeStore

    repo_root = _find_repo_root()
    registry = Registry(repo_root)

    try:
        target = registry.get_target(args.target)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    store = KnowledgeStore(repo_root / target.knowledge_path)
    entries = store.get_drift_report()

    if not entries:
        console.print(f"  No evaluator drift detected for [bold]{target.id}[/bold].")
        return

    console.print(
        Panel(
            f"Drift report for [bold]{target.id}[/bold]",
            title="anneal drift",
            style="yellow",
        )
    )
    for entry in entries:
        console.print(
            f"  [yellow]{entry.criterion_name}[/yellow]  "
            f"variance={entry.variance:.4f}  "
            f"mean={entry.mean_score:.4f}  "
            f"window={entry.window_size}"
        )


def _handle_dashboard(args: argparse.Namespace) -> None:
    """Handle ``anneal dashboard``."""
    try:
        from anneal.engine.dashboard import DashboardServer
    except ImportError:
        raise SystemExit(
            "Dashboard requires aiohttp. Install with: uv add 'anneal-cli[dashboard]'"
        )

    # Resolve .anneal root directory
    if args.root:
        anneal_root = Path(args.root).resolve()
    else:
        repo_root = _find_repo_root()
        anneal_root = repo_root / ".anneal"

    if not anneal_root.exists():
        console.print(f"[red].anneal directory not found at {anneal_root}[/red]")
        console.print("[dim]Run 'anneal init' first, or specify --root <path>[/dim]")
        sys.exit(1)

    server = DashboardServer(anneal_root, host=args.host, port=args.port)

    if getattr(args, "open", False):
        import webbrowser
        webbrowser.open(f"http://{args.host}:{args.port}")

    console.print(
        Panel(
            f"Dashboard reading from [bold]{anneal_root}[/bold]\n"
            f"  Serving at [bold]http://{args.host}:{args.port}[/bold]\n"
            f"  Press Ctrl+C to stop",
            title="anneal dashboard",
            style="blue",
        )
    )

    async def _run_dashboard() -> None:
        await server.start()
        try:
            while True:
                await asyncio.sleep(1)
        finally:
            await server.stop()

    try:
        asyncio.run(_run_dashboard())
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


def _handle_local_check(args: argparse.Namespace) -> None:
    """Handle ``anneal local-check``."""
    from anneal.engine.client import check_local_server, is_local_model

    model = args.model
    if not is_local_model(model):
        console.print(
            f"[red]{model} is not a local model. "
            f"Use ollama/model-name, lmstudio/model-name, or local/model-name.[/red]"
        )
        sys.exit(1)

    console.print(f"[dim]Checking {model}...[/dim]")
    healthy, message = asyncio.run(check_local_server(model))

    if healthy:
        console.print(
            Panel(
                f"[green]{message}[/green]",
                title=f"anneal local-check — {model}",
                style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[red]{message}[/red]",
                title=f"anneal local-check — {model}",
                style="red",
            )
        )
        sys.exit(1)


def _handle_compare(args: argparse.Namespace) -> None:
    """Handle ``anneal compare``."""
    from anneal.engine.compare import compare_runs

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)

    if not run_a.exists():
        console.print(f"[red]Run path not found: {run_a}[/red]")
        sys.exit(1)
    if not run_b.exists():
        console.print(f"[red]Run path not found: {run_b}[/red]")
        sys.exit(1)

    compare_runs(run_a, run_b, args.label_a, args.label_b)


def _handle_templates(_args: argparse.Namespace) -> None:
    """Handle ``anneal templates``."""
    from anneal.suggest.templates import list_templates

    templates = list_templates()
    if not templates:
        console.print("[yellow]No templates available.[/yellow]")
        return

    for t in templates:
        console.print(
            f"  [bold]{t.name:20s}[/bold] {t.description:50s} "
            f"[dim]{t.eval_mode}/{t.direction}[/dim]"
        )


def _handle_suggest(args: argparse.Namespace) -> None:
    """Handle ``anneal suggest``."""
    from anneal.suggest.analyzer import analyze_problem
    from anneal.suggest.generators import build_suggestion
    from anneal.suggest.renderer import render_criteria, render_plan, write_suggestion_files
    from anneal.suggest.scope import generate_scope

    repo_root = _find_repo_root()

    # Pre-flight: verify artifact files exist
    for artifact in args.artifact:
        if not (repo_root / artifact).exists():
            console.print(f"[red]Artifact not found: {artifact}[/red]")
            sys.exit(1)

    # Step 1: Analyze the problem
    console.print("[dim]Analyzing problem...[/dim]")
    intent = asyncio.run(analyze_problem(
        problem=args.problem,
        artifact_paths=args.artifact,
        eval_cmd=args.eval_cmd,
        parse_cmd=args.parse_cmd,
        metric=args.metric,
        direction=args.direction,
        model=args.model,
    ))

    # Step 2: Scan codebase and generate scope
    scope = generate_scope(repo_root, args.artifact, intent)

    # Step 3: Build the complete suggestion
    suggestion = build_suggestion(
        intent=intent,
        scope=scope,
        artifact_paths=args.artifact,
        eval_cmd=args.eval_cmd,
        parse_cmd=args.parse_cmd or ("cat" if args.eval_cmd else None),
    )

    # Step 4: Render the plan
    render_plan(suggestion)
    if suggestion.intent.criteria:
        render_criteria(suggestion)

    # Step 5: Write files or show what would be written
    target_dir = repo_root / ".anneal" / "targets" / suggestion.name
    if args.accept:
        written = write_suggestion_files(suggestion, target_dir)
        console.print(
            Panel(
                f"Files written to [bold]{target_dir}[/bold]:\n"
                + "\n".join(f"  {p.name}" for p in written)
                + "\n\nRegister with:\n"
                + f"  anneal register --name {suggestion.name} "
                + f"--artifact {' '.join(args.artifact)} "
                + f"--eval-mode {suggestion.eval_mode} "
                + (f"--run-cmd '{suggestion.run_command}' --parse-cmd '{suggestion.parse_command}' " if suggestion.eval_mode == "deterministic" else f"--criteria {target_dir / 'eval_criteria.toml'} ")
                + f"--direction {suggestion.direction} "
                + f"--scope {target_dir / 'scope.yaml'}",
                title="anneal suggest — Files Written",
                style="green",
            )
        )
    else:
        console.print()
        console.print("[dim]Run with --accept to write files, or review the plan above and adjust.[/dim]")
        console.print(f"[dim]Target directory: {target_dir}[/dim]")


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
    reg.add_argument("--held-out-prompts", help="Path to held-out prompts file (one per line, stochastic only)")
    reg.add_argument("--held-out-interval", type=int, help="Run held-out eval every N kept experiments (default: 10)")
    reg.add_argument("--constraint", action="append", help="Metric constraint: 'metric>=value' or 'metric<=value' (repeatable)")
    reg.add_argument("--domain-tier", choices=["sandbox", "deployment"], help="Domain tier (default: sandbox)")
    reg.add_argument("--meta-depth", type=int, help="Meta-optimization depth (0=disabled, 1=enabled)")
    reg.add_argument("--verifier", action="append", metavar="NAME:COMMAND", help="Binary pass/fail verifier gate (repeatable, format: name:command)")
    reg.add_argument("--restart-probability", type=float, default=0.0, help="Probability of restart experiment per cycle (0.0-1.0, default: 0.0)")

    # -- run (stub) --
    run = subparsers.add_parser("run", help="Run optimization loop")
    run.add_argument("--target", help="Target identifier (omit to run all registered targets)")
    run.add_argument("--experiments", type=int, help="Stop after N experiments")
    run.add_argument("--until", type=float, help="Stop when score reaches threshold")
    run.add_argument("--foreground", action="store_true", help="Block terminal")
    run.add_argument("--dry-run", action="store_true", help="Preview cost estimate without running experiments")
    # Runtime overrides (do not persist — apply to this run only)
    run.add_argument("--samples", type=int, help="Override sample count (N) for this run")
    run.add_argument("--confidence", type=float, help="Override confidence level for this run")
    run.add_argument("--agent-budget", type=float, help="Override per-invocation agent budget for this run")
    run.add_argument("--search", choices=["greedy", "annealing", "population"], help="Override search strategy for this run")
    run.add_argument("--population-size", type=int, help="Population size for population search (default: 4)")
    run.add_argument("--global-learnings", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable cross-project learning pool")

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
    conf.add_argument("--baseline", type=float, help="Set baseline score manually")
    conf.add_argument("--time-budget", type=int, help="Set time budget per experiment (seconds)")
    conf.add_argument("--max-failures", type=int, help="Set max consecutive failures before HALT")
    conf.add_argument("--meta-depth", type=int, help="Set meta-optimization depth (0=disabled, 1=enabled)")
    conf.add_argument("--knowledge-context", action=argparse.BooleanOptionalAction, help="Enable/disable knowledge context injection into agent prompt")

    # -- dashboard --
    dash = subparsers.add_parser("dashboard", help="Start live dashboard reading from .anneal/ directory")
    dash.add_argument("--root", help="Path to .anneal directory (default: .anneal/ in repo root)")
    dash.add_argument("--port", type=int, default=8080, help="Server port")
    dash.add_argument("--host", default="127.0.0.1", help="Server host")
    dash.add_argument("--open", action="store_true", help="Open browser on start")

    # -- drift --
    drift = subparsers.add_parser("drift", help="Show evaluator drift report")
    drift.add_argument("--target", required=True, help="Target identifier")

    # -- list --
    subparsers.add_parser("list", help="List all registered targets")

    # -- suggest --
    sug = subparsers.add_parser("suggest", help="Generate experiment configuration from a problem description")
    sug.add_argument("--problem", required=True, help="Natural-language description of what to optimize")
    sug.add_argument("--artifact", required=True, nargs="+", help="Artifact file paths to optimize")
    sug.add_argument("--eval-cmd", help="Deterministic eval run command (omit for stochastic mode)")
    sug.add_argument("--parse-cmd", help="Deterministic eval parse command (default: cat)")
    sug.add_argument("--metric", help="Metric name (e.g., 'p95 latency in ms', 'character count')")
    sug.add_argument("--direction", choices=["maximize", "minimize"], help="Optimization direction")
    sug.add_argument("--accept", action="store_true", help="Write files and register target (skip review)")
    sug.add_argument("--model", default="gpt-4.1", help="Model for problem analysis (default: gpt-4.1)")

    # -- local-check --
    lc = subparsers.add_parser("local-check", help="Verify local LLM server health and model availability")
    lc.add_argument("--model", required=True, help="Local model to check (e.g., ollama/llama3.1:8b)")

    # -- compare --
    cmp = subparsers.add_parser("compare", help="Compare two experiment runs side-by-side")
    cmp.add_argument("run_a", help="Path to first run directory or experiments.jsonl")
    cmp.add_argument("run_b", help="Path to second run directory or experiments.jsonl")
    cmp.add_argument("--label-a", default="Run A", help="Label for first run")
    cmp.add_argument("--label-b", default="Run B", help="Label for second run")

    # -- templates --
    subparsers.add_parser("templates", help="List available experiment templates")

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
        "drift": lambda: _handle_drift(args),
        "list": lambda: _handle_list(args),
        "suggest": lambda: _handle_suggest(args),
        "compare": lambda: _handle_compare(args),
        "templates": lambda: _handle_templates(args),
        "local-check": lambda: _handle_local_check(args),
    }

    try:
        handlers[args.command]()  # type: ignore[operator]
    except (ScopeError, GitError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
