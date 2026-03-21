"""Target registry: manages config.toml persistence, target lifecycle,
worktree creation, and project initialization.

The registry is the authoritative store for optimization target configuration.
It owns config.toml and mediates all target registration and deregistration.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

from anneal.engine.environment import GitEnvironment
from anneal.engine.scope import (
    ScopeError,
    compute_scope_hash,
    load_scope,
    validate_scope,
)
from anneal.engine.types import (
    AgentConfig,
    BinaryCriterion,
    BudgetCap,
    ColabConfig,
    ConstraintCommand,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    EvalMode,
    MetricConstraint,
    NotificationConfig,
    OptimizationTarget,
    PopulationConfig,
    StochasticEval,
)

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Raised on registry operations failure."""


# ---------------------------------------------------------------------------
# TOML serialization helpers
# ---------------------------------------------------------------------------


def _toml_value(value: object) -> str:
    """Serialize a single Python value to a TOML-compatible string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        # Escape backslashes and quotes
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        items = [_toml_value(v) for v in value]
        return f"[{', '.join(items)}]"
    if value is None:
        # TOML has no null; skip these at the call site
        return ""
    raise RegistryError(f"Cannot serialize {type(value).__name__} to TOML: {value!r}")


def _serialize_target_toml(target: OptimizationTarget) -> str:
    """Serialize an OptimizationTarget to TOML table sections."""
    tid = target.id
    lines: list[str] = []
    lines.append(f"[targets.{tid}]")

    # Flat scalar/list fields
    flat_fields = [
        ("id", target.id),
        ("domain_tier", target.domain_tier.value),
        ("artifact_paths", target.artifact_paths),
        ("scope_path", target.scope_path),
        ("scope_hash", target.scope_hash),
        ("eval_mode", target.eval_mode.value),
        ("time_budget_seconds", target.time_budget_seconds),
        ("loop_interval_seconds", target.loop_interval_seconds),
        ("knowledge_path", target.knowledge_path),
        ("worktree_path", target.worktree_path),
        ("git_branch", target.git_branch),
        ("baseline_score", target.baseline_score),
        ("baseline_raw_scores", target.baseline_raw_scores),
        ("max_consecutive_failures", target.max_consecutive_failures),
        ("meta_depth", target.meta_depth),
        ("inject_knowledge_context", target.inject_knowledge_context),
    ]

    for key, val in flat_fields:
        lines.append(f"{key} = {_toml_value(val)}")

    # eval_config sub-table
    ec = target.eval_config
    lines.append("")
    lines.append(f"[targets.{tid}.eval_config]")
    lines.append(f"metric_name = {_toml_value(ec.metric_name)}")
    lines.append(f"direction = {_toml_value(ec.direction.value)}")
    lines.append(f"min_improvement_threshold = {_toml_value(ec.min_improvement_threshold)}")
    lines.append(f"held_out_interval = {_toml_value(ec.held_out_interval)}")

    # F2: Serialize constraints
    if ec.constraints:
        constraint_items: list[str] = []
        for c in ec.constraints:
            constraint_items.append(
                f"{{metric_name = {_toml_value(c.metric_name)}, "
                f"threshold = {_toml_value(c.threshold)}, "
                f"direction = {_toml_value(c.direction.value)}}}"
            )
        lines.append(f"constraints = [{', '.join(constraint_items)}]")

    if ec.constraint_commands:
        for cc in ec.constraint_commands:
            lines.append("")
            lines.append(f"[[targets.{tid}.eval_config.constraint_commands]]")
            lines.append(f"name = {_toml_value(cc.name)}")
            lines.append(f"run_command = {_toml_value(cc.run_command)}")
            lines.append(f"parse_command = {_toml_value(cc.parse_command)}")
            lines.append(f"timeout_seconds = {_toml_value(cc.timeout_seconds)}")
            lines.append(f"threshold = {_toml_value(cc.threshold)}")
            lines.append(f"direction = {_toml_value(cc.direction.value)}")

    if ec.deterministic is not None:
        det = ec.deterministic
        lines.append("")
        lines.append(f"[targets.{tid}.eval_config.deterministic]")
        lines.append(f"run_command = {_toml_value(det.run_command)}")
        lines.append(f"parse_command = {_toml_value(det.parse_command)}")
        lines.append(f"timeout_seconds = {_toml_value(det.timeout_seconds)}")

    if ec.stochastic is not None:
        sto = ec.stochastic
        lines.append("")
        lines.append(f"[targets.{tid}.eval_config.stochastic]")
        lines.append(f"sample_count = {_toml_value(sto.sample_count)}")
        lines.append(f"test_prompts = {_toml_value(sto.test_prompts)}")
        lines.append(f"generation_prompt_template = {_toml_value(sto.generation_prompt_template)}")
        lines.append(f"output_format = {_toml_value(sto.output_format)}")
        lines.append(f"confidence_level = {_toml_value(sto.confidence_level)}")
        lines.append(f"judgment_votes = {_toml_value(sto.judgment_votes)}")

        # F1: held_out_prompts
        if sto.held_out_prompts:
            lines.append(f"held_out_prompts = {_toml_value(sto.held_out_prompts)}")

        # F2: min_criterion_scores as inline table
        if sto.min_criterion_scores:
            mcs_parts = [f"{_toml_value(k)} = {_toml_value(v)}" for k, v in sto.min_criterion_scores.items()]
            lines.append(f"min_criterion_scores = {{{', '.join(mcs_parts)}}}")

        # criteria as array of inline tables
        criteria_items: list[str] = []
        for c in sto.criteria:
            name = _toml_value(c.name)
            question = _toml_value(c.question)
            criteria_items.append(f"{{name = {name}, question = {question}}}")
        lines.append(f"criteria = [{', '.join(criteria_items)}]")

        if sto.generation_agent_config is not None:
            lines.append("")
            lines.append(f"[targets.{tid}.eval_config.stochastic.generation_agent_config]")
            _append_agent_config_lines(lines, sto.generation_agent_config)

    # colab sub-table
    if ec.colab is not None and ec.colab.enabled:
        cb = ec.colab
        lines.append("")
        lines.append(f"[targets.{tid}.eval_config.colab]")
        lines.append(f"enabled = {_toml_value(cb.enabled)}")
        lines.append(f"accelerator = {_toml_value(cb.accelerator)}")
        lines.append(f"setup_script = {_toml_value(cb.setup_script)}")
        lines.append(f"credentials_path = {_toml_value(cb.credentials_path)}")
        lines.append(f"max_ccu_per_day = {_toml_value(cb.max_ccu_per_day)}")
        lines.append(f"timeout_seconds = {_toml_value(cb.timeout_seconds)}")

    # agent_config sub-table
    lines.append("")
    lines.append(f"[targets.{tid}.agent_config]")
    _append_agent_config_lines(lines, target.agent_config)

    # budget_cap sub-table
    if target.budget_cap is not None:
        bc = target.budget_cap
        lines.append("")
        lines.append(f"[targets.{tid}.budget_cap]")
        lines.append(f"max_usd_per_day = {_toml_value(bc.max_usd_per_day)}")
        lines.append(f"cumulative_usd_spent = {_toml_value(bc.cumulative_usd_spent)}")

    # population_config sub-table
    if target.population_config is not None:
        pc = target.population_config
        lines.append("")
        lines.append(f"[targets.{tid}.population_config]")
        lines.append(f"population_size = {_toml_value(pc.population_size)}")
        lines.append(f"tournament_size = {_toml_value(pc.tournament_size)}")

    # notifications sub-table
    notif = target.notifications
    lines.append("")
    lines.append(f"[targets.{tid}.notifications]")
    if notif.webhook_url is not None:
        lines.append(f"webhook_url = {_toml_value(notif.webhook_url)}")
    if notif.fallback_webhook_url is not None:
        lines.append(f"fallback_webhook_url = {_toml_value(notif.fallback_webhook_url)}")
    lines.append(f"status_file = {_toml_value(notif.status_file)}")
    lines.append(f"notify_on = {_toml_value(notif.notify_on)}")
    lines.append(f"milestone_interval = {_toml_value(notif.milestone_interval)}")
    lines.append(f"webhook_retry_count = {_toml_value(notif.webhook_retry_count)}")
    lines.append(f"webhook_retry_delay_seconds = {_toml_value(notif.webhook_retry_delay_seconds)}")

    return "\n".join(lines)


def _append_agent_config_lines(lines: list[str], ac: AgentConfig) -> None:
    """Append agent config key-value pairs to lines list."""
    lines.append(f"mode = {_toml_value(ac.mode)}")
    lines.append(f"model = {_toml_value(ac.model)}")
    lines.append(f"evaluator_model = {_toml_value(ac.evaluator_model)}")
    lines.append(f"max_budget_usd = {_toml_value(ac.max_budget_usd)}")
    lines.append(f"max_context_tokens = {_toml_value(ac.max_context_tokens)}")
    lines.append(f"temperature = {_toml_value(ac.temperature)}")
    lines.append(f"sandbox = {_toml_value(ac.sandbox)}")


# ---------------------------------------------------------------------------
# TOML deserialization helpers
# ---------------------------------------------------------------------------


def _parse_agent_config(data: dict[str, object]) -> AgentConfig:
    """Parse an AgentConfig from a TOML dict."""
    return AgentConfig(
        mode=data["mode"],  # type: ignore[arg-type]
        model=str(data["model"]),
        evaluator_model=str(data["evaluator_model"]),
        max_budget_usd=float(data.get("max_budget_usd", 0.10)),  # type: ignore[arg-type]
        max_context_tokens=int(data.get("max_context_tokens", 80_000)),  # type: ignore[arg-type]
        temperature=float(data.get("temperature", 0.7)),  # type: ignore[arg-type]
        sandbox=bool(data.get("sandbox", False)),
    )


def _parse_eval_config(data: dict[str, object]) -> EvalConfig:
    """Parse an EvalConfig from a TOML dict."""
    det: DeterministicEval | None = None
    sto: StochasticEval | None = None

    det_data = data.get("deterministic")
    if isinstance(det_data, dict):
        det = DeterministicEval(
            run_command=str(det_data["run_command"]),
            parse_command=str(det_data["parse_command"]),
            timeout_seconds=int(det_data["timeout_seconds"]),
        )

    sto_data = data.get("stochastic")
    if isinstance(sto_data, dict):
        criteria = [
            BinaryCriterion(name=str(c["name"]), question=str(c["question"]))
            for c in sto_data.get("criteria", [])  # type: ignore[union-attr]
        ]
        gen_ac: AgentConfig | None = None
        gen_ac_data = sto_data.get("generation_agent_config")
        if isinstance(gen_ac_data, dict):
            gen_ac = _parse_agent_config(gen_ac_data)

        # F1: held_out_prompts
        held_out_prompts = [str(p) for p in sto_data.get("held_out_prompts", [])]  # type: ignore[union-attr]
        # F2: min_criterion_scores
        min_criterion_scores = {
            str(k): float(v) for k, v in (sto_data.get("min_criterion_scores") or {}).items()  # type: ignore[union-attr]
        }

        sto = StochasticEval(
            sample_count=int(sto_data["sample_count"]),
            criteria=criteria,
            test_prompts=list(sto_data.get("test_prompts", [])),  # type: ignore[arg-type]
            generation_prompt_template=str(sto_data["generation_prompt_template"]),
            output_format=str(sto_data["output_format"]),
            confidence_level=float(sto_data.get("confidence_level", 0.95)),  # type: ignore[arg-type]
            generation_agent_config=gen_ac,
            held_out_prompts=held_out_prompts,
            min_criterion_scores=min_criterion_scores,
            judgment_votes=int(sto_data.get("judgment_votes", 3)),  # type: ignore[arg-type]
        )

    # F2: Parse constraints
    constraints: list[MetricConstraint] = []
    for c in data.get("constraints", []):  # type: ignore[union-attr]
        constraints.append(MetricConstraint(
            metric_name=str(c["metric_name"]),  # type: ignore[index]
            threshold=float(c["threshold"]),  # type: ignore[index]
            direction=Direction(str(c["direction"])),  # type: ignore[index]
        ))

    constraint_commands: list[ConstraintCommand] = []
    for cc in data.get("constraint_commands", []):  # type: ignore[union-attr]
        constraint_commands.append(ConstraintCommand(
            name=str(cc["name"]),  # type: ignore[index]
            run_command=str(cc["run_command"]),  # type: ignore[index]
            parse_command=str(cc["parse_command"]),  # type: ignore[index]
            timeout_seconds=int(cc["timeout_seconds"]),  # type: ignore[index]
            threshold=float(cc["threshold"]),  # type: ignore[index]
            direction=Direction(str(cc["direction"])),  # type: ignore[index]
        ))

    # Parse colab config
    colab_config: ColabConfig | None = None
    colab_data = data.get("colab")
    if colab_data is not None and isinstance(colab_data, dict):
        colab_config = ColabConfig(
            enabled=bool(colab_data.get("enabled", False)),
            accelerator=str(colab_data.get("accelerator", "T4")),
            setup_script=str(colab_data.get("setup_script", "")),
            credentials_path=str(colab_data.get("credentials_path", ".anneal/colab-credentials.json")),
            max_ccu_per_day=float(colab_data.get("max_ccu_per_day", 10.0)),  # type: ignore[arg-type]
            timeout_seconds=int(colab_data.get("timeout_seconds", 600)),  # type: ignore[arg-type]
        )

    return EvalConfig(
        metric_name=str(data["metric_name"]),
        direction=Direction(str(data["direction"])),
        min_improvement_threshold=float(data.get("min_improvement_threshold", 0.0)),  # type: ignore[arg-type]
        deterministic=det,
        stochastic=sto,
        held_out_interval=int(data.get("held_out_interval", 10)),  # type: ignore[arg-type]
        constraints=constraints,
        constraint_commands=constraint_commands,
        colab=colab_config,
    )


def _parse_notification_config(data: dict[str, object]) -> NotificationConfig:
    """Parse a NotificationConfig from a TOML dict."""
    return NotificationConfig(
        webhook_url=data.get("webhook_url"),  # type: ignore[arg-type]
        fallback_webhook_url=data.get("fallback_webhook_url"),  # type: ignore[arg-type]
        status_file=str(data.get("status_file", ".anneal-status")),
        notify_on=list(data.get("notify_on", ["PAUSED", "HALTED"])),  # type: ignore[arg-type]
        milestone_interval=int(data.get("milestone_interval", 10)),  # type: ignore[arg-type]
        webhook_retry_count=int(data.get("webhook_retry_count", 3)),  # type: ignore[arg-type]
        webhook_retry_delay_seconds=float(data.get("webhook_retry_delay_seconds", 5.0)),  # type: ignore[arg-type]
    )


def _parse_target(data: dict[str, object]) -> OptimizationTarget:
    """Parse an OptimizationTarget from a TOML dict."""
    bc_data = data.get("budget_cap")
    budget_cap: BudgetCap | None = None
    if isinstance(bc_data, dict):
        budget_cap = BudgetCap(
            max_usd_per_day=float(bc_data["max_usd_per_day"]),
            cumulative_usd_spent=float(bc_data.get("cumulative_usd_spent", 0.0)),  # type: ignore[arg-type]
        )

    notif_data = data.get("notifications")
    notifications = (
        _parse_notification_config(notif_data)  # type: ignore[arg-type]
        if isinstance(notif_data, dict)
        else NotificationConfig()
    )

    # Population config
    pc_data = data.get("population_config")
    population_config: PopulationConfig | None = None
    if isinstance(pc_data, dict):
        population_config = PopulationConfig(
            population_size=int(pc_data.get("population_size", 4)),  # type: ignore[arg-type]
            tournament_size=int(pc_data.get("tournament_size", 2)),  # type: ignore[arg-type]
        )

    return OptimizationTarget(
        id=str(data["id"]),
        domain_tier=DomainTier(str(data["domain_tier"])),
        artifact_paths=list(data.get("artifact_paths", [])),  # type: ignore[arg-type]
        scope_path=str(data["scope_path"]),
        scope_hash=str(data["scope_hash"]),
        eval_mode=EvalMode(str(data["eval_mode"])),
        eval_config=_parse_eval_config(data["eval_config"]),  # type: ignore[arg-type]
        agent_config=_parse_agent_config(data["agent_config"]),  # type: ignore[arg-type]
        time_budget_seconds=int(data["time_budget_seconds"]),  # type: ignore[arg-type]
        loop_interval_seconds=int(data["loop_interval_seconds"]),  # type: ignore[arg-type]
        knowledge_path=str(data["knowledge_path"]),
        worktree_path=str(data["worktree_path"]),
        git_branch=str(data["git_branch"]),
        baseline_score=float(data["baseline_score"]),  # type: ignore[arg-type]
        baseline_raw_scores=[float(s) for s in data.get("baseline_raw_scores", [])],  # type: ignore[union-attr]
        max_consecutive_failures=int(data.get("max_consecutive_failures", 5)),  # type: ignore[arg-type]
        budget_cap=budget_cap,
        meta_depth=int(data.get("meta_depth", 0)),  # type: ignore[arg-type]
        inject_knowledge_context=bool(data.get("inject_knowledge_context", False)),
        notifications=notifications,
        population_config=population_config,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CONFIG_VERSION = "0.1.0"


class Registry:
    """Target registry backed by config.toml."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve()
        self._targets: dict[str, OptimizationTarget] = {}
        self._git = GitEnvironment()
        self.load()

    @property
    def config_path(self) -> Path:
        return self._repo_root / ".anneal" / "config.toml"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load config.toml from disk. No-op if file doesn't exist."""
        if not self.config_path.exists():
            return

        raw = self.config_path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))

        targets_data = data.get("targets", {})
        self._targets = {}
        for tid, tdata in targets_data.items():
            self._targets[tid] = _parse_target(tdata)

    def save(self) -> None:
        """Persist current state to config.toml."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        sections: list[str] = []
        sections.append("[anneal]")
        sections.append(f'version = "{_CONFIG_VERSION}"')

        for target in self._targets.values():
            sections.append("")
            sections.append(_serialize_target_toml(target))

        content = "\n".join(sections) + "\n"
        self.config_path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Target lifecycle
    # ------------------------------------------------------------------

    async def register_target(self, target: OptimizationTarget) -> None:
        """Register a new target: validate scope, create worktree, store config.

        Raises RegistryError if validation fails or target ID already exists.
        """
        if target.id in self._targets:
            raise RegistryError(f"Target already registered: {target.id}")

        # Resolve scope path relative to repo root
        scope_path = self._repo_root / target.scope_path
        try:
            scope = load_scope(scope_path)
        except ScopeError as exc:
            raise RegistryError(f"Scope load failed for {target.id}: {exc}") from exc

        # Validate scope against invariants
        sibling_targets = list(self._targets.values())
        errors = validate_scope(scope, target.eval_mode, sibling_targets)
        if errors:
            raise RegistryError(
                f"Scope validation failed for {target.id}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # Compute and store scope hash
        target.scope_hash = compute_scope_hash(scope_path)

        # Create git worktree
        worktree_info = await self._git.create_worktree(self._repo_root, target.id)
        target.worktree_path = str(worktree_info.path.relative_to(self._repo_root))
        target.git_branch = worktree_info.branch

        # Configure gc to preserve experiment history
        await self._git.configure_gc(self._repo_root)

        # Store and persist
        self._targets[target.id] = target
        self.save()

        logger.info("Registered target %s (worktree: %s)", target.id, target.worktree_path)

    async def deregister_target(self, target_id: str) -> None:
        """Remove target: remove worktree, remove from config, preserve experiment history.

        Raises RegistryError if target doesn't exist.
        """
        if target_id not in self._targets:
            raise RegistryError(f"Target not found: {target_id}")

        # Remove worktree (experiment history in .anneal/targets/<id>/ is preserved)
        await self._git.remove_worktree(self._repo_root, target_id)

        del self._targets[target_id]
        self.save()

        logger.info("Deregistered target %s", target_id)

    def get_target(self, target_id: str) -> OptimizationTarget:
        """Get a target by ID. Raises RegistryError if not found."""
        if target_id not in self._targets:
            raise RegistryError(f"Target not found: {target_id}")
        return self._targets[target_id]

    def all_targets(self) -> list[OptimizationTarget]:
        """Return all registered targets."""
        return list(self._targets.values())

    def update_target(self, target: OptimizationTarget) -> None:
        """Update a target's config (e.g., after baseline score change). Persists to disk."""
        if target.id not in self._targets:
            raise RegistryError(f"Target not found: {target.id}")
        self._targets[target.id] = target
        self.save()


# ---------------------------------------------------------------------------
# Project initialization
# ---------------------------------------------------------------------------


async def init_project(repo_root: Path) -> None:
    """One-time project setup.

    1. Verify repo_root is a git repository (has .git)
    2. Create .anneal/ directory structure (config, targets, templates, worktrees)
    3. Add '.anneal/' to .gitignore if not already present
    4. Raise RegistryError if not a git repo or already initialized
    """
    repo_root = repo_root.resolve()

    # Verify git repository
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        raise RegistryError(f"Not a git repository: {repo_root}")

    # Check not already initialized
    anneal_dir = repo_root / ".anneal"
    config_path = anneal_dir / "config.toml"
    if config_path.exists():
        raise RegistryError(f"Project already initialized: {config_path} exists")

    # Create directory structure under .anneal/
    (anneal_dir / "targets").mkdir(parents=True, exist_ok=True)
    (anneal_dir / "templates").mkdir(parents=True, exist_ok=True)
    (anneal_dir / "worktrees").mkdir(parents=True, exist_ok=True)

    # Write default config.toml
    config_path.write_text(
        f'[anneal]\nversion = "{_CONFIG_VERSION}"\n',
        encoding="utf-8",
    )

    # Ensure .anneal/ is in .gitignore
    gitignore_path = repo_root / ".gitignore"
    ignore_entry = ".anneal/"

    if gitignore_path.exists():
        existing = gitignore_path.read_text(encoding="utf-8")
        if ignore_entry not in existing.splitlines():
            with gitignore_path.open("a", encoding="utf-8") as f:
                if not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{ignore_entry}\n")
    else:
        gitignore_path.write_text(f"{ignore_entry}\n", encoding="utf-8")

    logger.info("Initialized anneal project at %s", repo_root)
