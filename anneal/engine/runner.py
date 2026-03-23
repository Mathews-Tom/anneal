"""Experiment runner — state machine orchestrating single experiment cycles.

Wires together GitEnvironment, AgentInvoker, EvalEngine, GreedySearch,
Registry, and scope enforcement into the per-experiment cycle described
in the system design.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from anneal.engine.agent import AgentInvocationError, AgentInvoker, AgentTimeoutError
from anneal.engine.context import build_target_context
from anneal.engine.environment import GitEnvironment
from anneal.engine.eval import EvalEngine, EvalError
from anneal.engine.knowledge import KnowledgeStore
from anneal.engine.learning_pool import LearningPool, extract_learning
from anneal.engine.notifications import NotificationManager
from anneal.engine.registry import Registry
from anneal.engine.safety import pre_experiment_check
from anneal.engine.scope import enforce_scope, load_scope, verify_scope_hash
from anneal.engine.search import GreedySearch, SearchStrategy
from anneal.engine.types import (
    AgentInvocationResult,
    DeterministicEval,
    Direction,
    DomainTier,
    EvalConfig,
    ExperimentRecord,
    OptimizationTarget,
    Outcome,
    RunnerState,
    ScopeConfig,
)

logger = logging.getLogger(__name__)

DIVERGENCE_WARNING = 0.10   # 10%
DIVERGENCE_CRITICAL = 0.25  # 25%


class ScopeIntegrityError(Exception):
    """Raised when scope.yaml hash has drifted since registration."""


class RunLoopState:
    """Persisted loop counters restored across process restarts."""

    def __init__(
        self,
        consecutive_failures: int = 0,
        kept_count: int = 0,
        consecutive_no_kept: int = 0,
        total_experiments: int = 0,
        cumulative_cost_usd: float = 0.0,
    ) -> None:
        self.consecutive_failures = consecutive_failures
        self.kept_count = kept_count
        self.consecutive_no_kept = consecutive_no_kept
        self.total_experiments = total_experiments
        self.cumulative_cost_usd = cumulative_cost_usd

    def save(self, path: Path) -> None:
        """Persist loop state to JSON file."""
        path.write_text(
            json.dumps({
                "consecutive_failures": self.consecutive_failures,
                "kept_count": self.kept_count,
                "consecutive_no_kept": self.consecutive_no_kept,
                "total_experiments": self.total_experiments,
                "cumulative_cost_usd": self.cumulative_cost_usd,
            }),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> RunLoopState:
        """Load loop state from JSON file, or return defaults if absent."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            kept_count=int(data.get("kept_count", 0)),
            consecutive_no_kept=int(data.get("consecutive_no_kept", 0)),
            total_experiments=int(data.get("total_experiments", 0)),
            cumulative_cost_usd=float(data.get("cumulative_cost_usd", 0.0)),
        )


def _make_early_record(
    experiment_id: str,
    target: OptimizationTarget,
    pre_experiment_sha: str,
    start_time: float,
    outcome: Outcome,
    *,
    hypothesis: str = "",
    hypothesis_source: Literal["agent", "synthesized"] = "synthesized",
    tags: list[str] | None = None,
    failure_mode: str | None = None,
    cost_usd: float = 0.0,
    score: float | None = None,
) -> ExperimentRecord:
    """Build an ExperimentRecord for early-exit paths (timeout, blocked, etc.)."""
    return ExperimentRecord(
        id=experiment_id,
        target_id=target.id,
        git_sha=pre_experiment_sha,
        pre_experiment_sha=pre_experiment_sha,
        timestamp=datetime.now(tz=timezone.utc),
        hypothesis=hypothesis,
        hypothesis_source=hypothesis_source,
        mutation_diff_summary="",
        score=score if score is not None else target.baseline_score,
        score_ci_lower=None,
        score_ci_upper=None,
        raw_scores=None,
        baseline_score=target.baseline_score,
        outcome=outcome,
        failure_mode=failure_mode,
        duration_seconds=time.monotonic() - start_time,
        tags=tags or [],
        learnings="",
        cost_usd=cost_usd,
        bootstrap_seed=0,
        agent_model=target.agent_config.model,
    )


class ExperimentRunner:
    """Orchestrates single experiment cycles and experiment loops for a target."""

    def __init__(
        self,
        git: GitEnvironment,
        agent_invoker: AgentInvoker,
        eval_engine: EvalEngine,
        search: GreedySearch | SearchStrategy,
        registry: Registry,
        repo_root: Path | None = None,
        knowledge: KnowledgeStore | None = None,
        notifications: NotificationManager | None = None,
        learning_pool: LearningPool | None = None,
    ) -> None:
        self._git = git
        self._agent = agent_invoker
        self._eval = eval_engine
        self._search = search
        self._registry = registry
        self._repo_root = repo_root
        self._knowledge = knowledge
        self._notifications = notifications
        self._learning_pool = learning_pool
        self._stop_flags: set[str] = set()
        self._stop_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Stop flag (thread-safe)
    # ------------------------------------------------------------------

    def request_stop(self, target_id: str) -> None:
        """Signal the runner to stop after the current experiment."""
        with self._stop_lock:
            self._stop_flags.add(target_id)

    def is_stop_requested(self, target_id: str) -> bool:
        with self._stop_lock:
            return target_id in self._stop_flags

    def _clear_stop(self, target_id: str) -> None:
        with self._stop_lock:
            self._stop_flags.discard(target_id)

    # ------------------------------------------------------------------
    # Single experiment cycle
    # ------------------------------------------------------------------

    async def run_one(self, target: OptimizationTarget) -> ExperimentRecord:
        """Run a single experiment: context -> mutate -> validate -> eval -> decide -> log."""
        worktree = Path(target.worktree_path)
        start_time = time.monotonic()
        experiment_id = str(uuid.uuid4())

        # 1. Record pre-experiment state and verify scope integrity
        pre_sha = await self._git.rev_parse(worktree, "HEAD")
        base = self._repo_root if self._repo_root else worktree
        scope_path = base / target.scope_path
        if not verify_scope_hash(scope_path, target.scope_hash):
            raise ScopeIntegrityError(
                f"scope.yaml hash mismatch for target {target.id}. "
                f"Re-register with: anneal re-register --target {target.id}"
            )
        scope = load_scope(scope_path)
        artifact_content = self._read_artifacts(worktree, target.artifact_paths)
        pre_agent_status = set(
            path for _, path in await self._git.status_porcelain(worktree)
        )

        # 2. Build prompt with optional knowledge context
        # Auto-enable knowledge injection after sufficient KEPT experiments
        KNOWLEDGE_ACTIVATION_THRESHOLD = 20
        inject_knowledge = target.inject_knowledge_context
        if not inject_knowledge and self._knowledge:
            kept_total = sum(
                1 for r in self._knowledge.load_records()
                if r.outcome is Outcome.KEPT
            )
            if kept_total >= KNOWLEDGE_ACTIVATION_THRESHOLD:
                inject_knowledge = True
                logger.info(
                    "Auto-enabling knowledge context for %s (%d KEPT experiments)",
                    target.id, kept_total,
                )

        history: list[ExperimentRecord] = []
        knowledge_context = ""
        if self._knowledge and inject_knowledge:
            history = self._knowledge.load_records(limit=10)
            knowledge_context = self._knowledge.get_context()

        prompt, _context_tokens = build_target_context(
            target=target,
            worktree_path=worktree,
            repo_root=self._repo_root or worktree,
            history=history,
            knowledge_context=knowledge_context,
        )

        # 3. Invoke mutation agent
        result = await self._invoke_agent(
            target, worktree, prompt, pre_sha, start_time, experiment_id,
        )
        if isinstance(result, ExperimentRecord):
            return result
        agent_result = result
        cost_usd = agent_result.cost_usd
        hypothesis = agent_result.hypothesis or "No hypothesis provided"
        hypothesis_source = agent_result.hypothesis_source
        tags = agent_result.tags

        # Ablation logging: track whether knowledge context influenced the hypothesis
        if knowledge_context and hypothesis != "No hypothesis provided":
            # Check if any retrieved hypothesis keywords appear in the agent's hypothesis
            retrieved_hypotheses = [r.hypothesis for r in history if r.hypothesis]
            referenced = any(
                keyword in hypothesis.lower()
                for rh in retrieved_hypotheses
                for keyword in rh.lower().split()
                if len(keyword) > 5  # skip short common words
            )
            logger.info(
                "Knowledge ablation for %s: injected=%d records, hypothesis_references_knowledge=%s",
                target.id, len(history), referenced,
            )

        # 4. Enforce scope and commit valid edits
        commit_or_record = await self._enforce_and_commit(
            target, worktree, scope, pre_agent_status, pre_sha,
            start_time, experiment_id, cost_usd,
            hypothesis, hypothesis_source, tags,
        )
        if isinstance(commit_or_record, ExperimentRecord):
            return commit_or_record
        commit_sha = commit_or_record

        # Re-read artifact content after mutation
        artifact_content = self._read_artifacts(worktree, target.artifact_paths)

        # 5. Run pre-checks (constraints + fidelity stages)
        pre_check = await self._run_pre_checks(
            target, worktree, artifact_content, pre_sha,
            start_time, experiment_id, cost_usd,
            hypothesis, hypothesis_source, tags,
        )
        if pre_check is not None:
            return pre_check

        # 6. Evaluate and decide
        return await self._evaluate_and_decide(
            target, worktree, artifact_content, pre_sha, commit_sha,
            start_time, experiment_id, cost_usd,
            hypothesis, hypothesis_source, tags,
        )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    async def _invoke_agent(
        self,
        target: OptimizationTarget,
        worktree: Path,
        prompt: str,
        pre_sha: str,
        start_time: float,
        experiment_id: str,
    ) -> AgentInvocationResult | ExperimentRecord:
        """Invoke the mutation agent. Returns AgentInvocationResult on success
        or ExperimentRecord on early exit (timeout, crash, approval rejection)."""
        try:
            if target.domain_tier is DomainTier.DEPLOYMENT:
                agent_result = await self._agent.invoke_deployment(
                    target.agent_config, prompt, worktree, target.time_budget_seconds,
                )
                if target.approval_callback is None:
                    raise ValueError(
                        f"DEPLOYMENT target {target.id} requires approval_callback. "
                        f"Set approval_callback during registration."
                    )
                if not target.approval_callback(agent_result.raw_output):
                    return _make_early_record(
                        experiment_id, target, pre_sha, start_time,
                        Outcome.DISCARDED,
                        hypothesis=agent_result.hypothesis or "Deployment change rejected",
                        hypothesis_source=agent_result.hypothesis_source,
                        tags=agent_result.tags,
                        failure_mode="approval_rejected",
                        cost_usd=agent_result.cost_usd,
                    )
            else:
                agent_result = await self._agent.invoke(
                    target.agent_config, prompt, worktree, target.time_budget_seconds,
                )
        except AgentTimeoutError as exc:
            await self._handle_killed(worktree, pre_sha)
            return _make_early_record(
                experiment_id, target, pre_sha, start_time,
                Outcome.KILLED, hypothesis="Agent timed out", failure_mode=str(exc),
            )
        except AgentInvocationError as exc:
            await self._safe_restore(worktree, pre_sha)
            return _make_early_record(
                experiment_id, target, pre_sha, start_time,
                Outcome.CRASHED, hypothesis="Agent invocation failed", failure_mode=str(exc),
            )
        return agent_result

    async def _enforce_and_commit(
        self,
        target: OptimizationTarget,
        worktree: Path,
        scope: ScopeConfig,
        pre_agent_status: set[str],
        pre_sha: str,
        start_time: float,
        experiment_id: str,
        cost_usd: float,
        hypothesis: str,
        hypothesis_source: Literal["agent", "synthesized"],
        tags: list[str],
    ) -> str | ExperimentRecord:
        """Enforce scope, reset violations, commit valid edits.
        Returns commit SHA on success or ExperimentRecord on early exit."""
        _INTERNAL_FILES = {".anneal-status", ".anneal.lock"}
        git_status = [
            (code, path) for code, path in await self._git.status_porcelain(worktree)
            if path not in _INTERNAL_FILES and path not in pre_agent_status
        ]
        scope_result = await enforce_scope(worktree, scope, git_status)

        logger.info(
            "Scope check for %s: status=%s valid=%s violated=%s all_blocked=%s",
            target.id, git_status, scope_result.valid_paths,
            scope_result.violated_paths, scope_result.all_blocked,
        )

        if scope_result.all_blocked:
            await self._reset_violated(worktree, scope_result.violated_paths, git_status)
            await self._git.clean_untracked(worktree)
            return _make_early_record(
                experiment_id, target, pre_sha, start_time,
                Outcome.BLOCKED, hypothesis=hypothesis, hypothesis_source=hypothesis_source,
                tags=tags, failure_mode=f"All changes violated scope: {scope_result.violated_paths}",
                cost_usd=cost_usd,
            )

        if scope_result.has_violations:
            await self._reset_violated(worktree, scope_result.violated_paths, git_status)
            logger.warning("Scope violations reset for target %s: %s", target.id, scope_result.violated_paths)

        if not scope_result.valid_paths:
            return _make_early_record(
                experiment_id, target, pre_sha, start_time,
                Outcome.BLOCKED, hypothesis=hypothesis, hypothesis_source=hypothesis_source,
                tags=tags, failure_mode="Agent made no file changes", cost_usd=cost_usd,
            )

        return await self._git.commit(worktree, f"hypothesis: {hypothesis}", scope_result.valid_paths)

    async def _run_pre_checks(
        self,
        target: OptimizationTarget,
        worktree: Path,
        artifact_content: str,
        pre_sha: str,
        start_time: float,
        experiment_id: str,
        cost_usd: float,
        hypothesis: str,
        hypothesis_source: Literal["agent", "synthesized"],
        tags: list[str],
    ) -> ExperimentRecord | None:
        """Run constraint commands and fidelity stages.
        Returns ExperimentRecord if a check fails, None if all pass."""
        if target.eval_config.constraint_commands:
            fast_results = await self._eval.check_constraints(
                worktree, target.eval_config, artifact_content,
            )
            for name, passed, actual in fast_results:
                if not passed:
                    await self._git.reset_hard(worktree, pre_sha)
                    logger.warning("Fast constraint %s failed for target %s: actual=%.4f", name, target.id, actual)
                    early_record = _make_early_record(
                        experiment_id, target, pre_sha, start_time,
                        Outcome.DISCARDED, hypothesis=hypothesis, hypothesis_source=hypothesis_source,
                        tags=tags, failure_mode=f"constraint_violated:{name}", cost_usd=cost_usd,
                    )
                    if self._learning_pool is not None:
                        self._learning_pool.add(extract_learning(early_record, source_target=target.id))
                    return early_record

        if target.eval_config.fidelity_stages:
            for stage in target.eval_config.fidelity_stages:
                stage_config = EvalConfig(
                    metric_name=f"fidelity_{stage.name}",
                    direction=target.eval_config.direction,
                    deterministic=DeterministicEval(
                        run_command=stage.run_command,
                        parse_command=stage.parse_command,
                        timeout_seconds=stage.timeout_seconds,
                    ),
                )
                try:
                    stage_result = await self._eval.evaluate(worktree, stage_config, artifact_content)
                except EvalError as exc:
                    logger.warning("Fidelity stage %s failed: %s", stage.name, exc)
                    continue

                if target.eval_config.direction is Direction.HIGHER_IS_BETTER:
                    passed = stage_result.score >= stage.min_pass_score
                else:
                    passed = stage_result.score <= stage.min_pass_score

                if not passed:
                    await self._git.reset_hard(worktree, pre_sha)
                    logger.info(
                        "Fidelity stage %s rejected mutation for %s: score=%.4f (min=%.4f)",
                        stage.name, target.id, stage_result.score, stage.min_pass_score,
                    )
                    fidelity_record = _make_early_record(
                        experiment_id, target, pre_sha, start_time,
                        Outcome.DISCARDED, hypothesis=hypothesis, hypothesis_source=hypothesis_source,
                        tags=tags, failure_mode=f"fidelity_stage:{stage.name}",
                        cost_usd=cost_usd, score=stage_result.score,
                    )
                    if self._learning_pool is not None:
                        self._learning_pool.add(extract_learning(fidelity_record, source_target=target.id))
                    return fidelity_record

        return None

    async def _evaluate_and_decide(
        self,
        target: OptimizationTarget,
        worktree: Path,
        artifact_content: str,
        pre_sha: str,
        commit_sha: str,
        start_time: float,
        experiment_id: str,
        cost_usd: float,
        hypothesis: str,
        hypothesis_source: Literal["agent", "synthesized"],
        tags: list[str],
    ) -> ExperimentRecord:
        """Evaluate mutation, decide keep/discard, persist record."""
        try:
            eval_result = await self._eval.evaluate(worktree, target.eval_config, artifact_content)
        except EvalError as exc:
            await self._safe_restore(worktree, pre_sha)
            return _make_early_record(
                experiment_id, target, pre_sha, start_time,
                Outcome.CRASHED, hypothesis=hypothesis, hypothesis_source=hypothesis_source,
                tags=tags, failure_mode=str(exc), cost_usd=cost_usd,
            )

        cost_usd += eval_result.cost_usd

        # Decide: keep or discard
        stochastic_conf = target.eval_config.stochastic
        confidence = stochastic_conf.confidence_level if stochastic_conf else 0.95

        if isinstance(self._search, GreedySearch) and self._knowledge:
            experiments_in_window = self._knowledge.record_count() % self._knowledge.CONSOLIDATION_INTERVAL
            window_size = self._knowledge.CONSOLIDATION_INTERVAL
            adjusted_alpha = GreedySearch._adjusted_alpha(
                1 - confidence, experiments_in_window, window_size,
            )
            confidence = 1 - adjusted_alpha

        if target.eval_config.stochastic is not None and not target.baseline_raw_scores:
            logger.info("Cold-start for stochastic target %s: accepting first evaluation as baseline", target.id)
            keep = True
        else:
            keep = self._search.should_keep(
                eval_result, target.baseline_score,
                target.baseline_raw_scores or None,
                target.eval_config.direction,
                target.eval_config.min_improvement_threshold,
                confidence,
            )

        # Check constraints before finalizing KEEP
        constraint_failure: str | None = None
        if keep:
            constraint_results = await self._eval.check_constraints(
                worktree, target.eval_config, artifact_content,
            )
            for name, passed, actual in constraint_results:
                if not passed:
                    constraint_failure = f"constraint_violated:{name}"
                    logger.warning("Constraint %s failed for target %s: actual=%.4f", name, target.id, actual)
                    break

        if keep and constraint_failure is None:
            outcome = Outcome.KEPT
            target.baseline_score = eval_result.score
            if eval_result.raw_scores is not None:
                target.baseline_raw_scores = list(eval_result.raw_scores)
            self._registry.update_target(target)
            git_sha = commit_sha
        else:
            outcome = Outcome.DISCARDED
            await self._git.reset_hard(worktree, pre_sha)
            git_sha = pre_sha

        record = ExperimentRecord(
            id=experiment_id,
            target_id=target.id,
            git_sha=git_sha,
            pre_experiment_sha=pre_sha,
            timestamp=datetime.now(tz=timezone.utc),
            hypothesis=hypothesis,
            hypothesis_source=hypothesis_source,
            mutation_diff_summary="",
            score=eval_result.score,
            score_ci_lower=eval_result.ci_lower,
            score_ci_upper=eval_result.ci_upper,
            raw_scores=eval_result.raw_scores,
            baseline_score=target.baseline_score,
            outcome=outcome,
            failure_mode=constraint_failure,
            duration_seconds=time.monotonic() - start_time,
            tags=tags,
            learnings="",
            cost_usd=cost_usd,
            bootstrap_seed=0,
            agent_model=target.agent_config.model,
            criterion_names=eval_result.criterion_names,
            per_criterion_scores=eval_result.per_criterion_scores,
        )

        if self._knowledge:
            self._knowledge.append_record(record)
            self._knowledge.update_index(record)
            self._knowledge.consolidate_if_due()

        if self._learning_pool is not None:
            self._learning_pool.add(extract_learning(record, source_target=target.id))

        return record

    # ------------------------------------------------------------------
    # Experiment loop
    # ------------------------------------------------------------------

    async def run_loop(
        self,
        target: OptimizationTarget,
        max_experiments: int | None = None,
        stop_score: float | None = None,
        on_experiment: Callable[[ExperimentRecord], None] | None = None,
    ) -> list[ExperimentRecord]:
        """Run experiments in a loop until stopped."""
        self._clear_stop(target.id)
        records: list[ExperimentRecord] = []

        # Restore loop state from previous run (crash recovery)
        state_path = Path(target.knowledge_path) / ".loop-state.json"
        loop = RunLoopState.load(state_path)
        if loop.total_experiments > 0:
            logger.info(
                "Restored loop state for %s: %d experiments, %d kept, %d consecutive failures",
                target.id, loop.total_experiments, loop.kept_count, loop.consecutive_failures,
            )

        while True:
            # Check stop conditions
            if self.is_stop_requested(target.id):
                logger.info("Stop requested for target %s", target.id)
                break

            # Check for .stop file (from CLI `anneal stop`)
            stop_file = Path(target.knowledge_path) / ".stop"
            if stop_file.exists():
                stop_file.unlink()
                logger.info("Stop file detected for target %s", target.id)
                break

            if max_experiments is not None and len(records) >= max_experiments:
                break

            if stop_score is not None and target.baseline_score >= stop_score:
                break

            if loop.consecutive_failures >= target.max_consecutive_failures:
                logger.warning(
                    "Target %s halted: %d consecutive failures",
                    target.id,
                    loop.consecutive_failures,
                )
                await self._write_status(target, RunnerState.HALTED, records[-1] if records else None)
                if self._notifications:
                    await self._notifications.notify_state(
                        target.id, RunnerState.HALTED,
                        f"{loop.consecutive_failures} consecutive failures",
                        score=target.baseline_score,
                        experiment_count=len(records),
                    )
                break

            # Pre-experiment safety checks
            safe, reason = pre_experiment_check(
                target, Path(target.worktree_path), context_tokens=0,
            )
            if not safe:
                logger.warning("Target %s paused: %s", target.id, reason)
                await self._write_status(target, RunnerState.PAUSED, records[-1] if records else None)
                if self._notifications:
                    await self._notifications.notify_state(
                        target.id, RunnerState.PAUSED, reason,
                        score=target.baseline_score,
                        experiment_count=len(records),
                    )
                break

            # Run one experiment
            await self._write_status(target, RunnerState.RUNNING, records[-1] if records else None)
            record = await self.run_one(target)
            records.append(record)

            # Track consecutive failures
            if record.outcome in (Outcome.KILLED, Outcome.CRASHED):
                loop.consecutive_failures += 1
            else:
                loop.consecutive_failures = 0

            # Track kept count for held-out eval and plateau detection
            if record.outcome is Outcome.KEPT:
                loop.kept_count += 1
                loop.consecutive_no_kept = 0
            else:
                loop.consecutive_no_kept += 1

            loop.total_experiments += 1
            loop.cumulative_cost_usd += record.cost_usd
            loop.save(state_path)

            # F1: Held-out evaluation at regular intervals
            held_out_interval = target.eval_config.held_out_interval
            stochastic_conf = target.eval_config.stochastic
            if (
                record.outcome is Outcome.KEPT
                and stochastic_conf is not None
                and stochastic_conf.held_out_prompts
                and loop.kept_count % held_out_interval == 0
            ):
                worktree = Path(target.worktree_path)
                artifact_content = self._read_artifacts(worktree, target.artifact_paths)
                try:
                    held_out_result = await self._eval.evaluate_held_out(
                        worktree, target.eval_config, artifact_content,
                    )
                    record.held_out_score = held_out_result.score
                    logger.info(
                        "Held-out eval for %s: score=%.4f (main=%.4f)",
                        target.id, held_out_result.score, record.score,
                    )
                    if record.score > 0:
                        divergence = abs(held_out_result.score - record.score) / record.score
                        if divergence > DIVERGENCE_CRITICAL:
                            logger.error(
                                "CRITICAL: Held-out diverges %.0f%% from main for %s. "
                                "Evaluator may be compromised.",
                                divergence * 100, target.id,
                            )
                        elif divergence > DIVERGENCE_WARNING:
                            logger.warning(
                                "Held-out diverges %.0f%% from main for %s",
                                divergence * 100, target.id,
                            )
                except EvalError as exc:
                    logger.warning("Held-out eval failed for %s: %s", target.id, exc)

            # F8: Meta-optimization on plateau
            meta_m = min(target.max_consecutive_failures, 10)
            if (
                target.meta_depth > 0
                and loop.consecutive_no_kept >= meta_m
                and len(records) >= meta_m
            ):
                logger.info(
                    "Plateau detected for %s (%d consecutive non-KEPT). Triggering meta-optimization.",
                    target.id, loop.consecutive_no_kept,
                )
                recent_scores = [r.score for r in records[-meta_m:]]
                trajectory = ", ".join(f"{s:.4f}" for s in recent_scores)
                meta_prompt = (
                    f"The optimization for target '{target.id}' has plateaued. "
                    f"No improvements in the last {loop.consecutive_no_kept} experiments. "
                    f"Recent score trajectory: [{trajectory}]. "
                    f"Current baseline: {target.baseline_score:.4f}. "
                    f"Revise the optimization strategy in program.md to break through."
                )
                base = self._repo_root if self._repo_root else Path(target.worktree_path)
                program_md = base / target.knowledge_path / "program.md"
                if program_md.exists():
                    try:
                        await self._agent.invoke_meta(
                            target.agent_config,
                            meta_prompt,
                            Path(target.worktree_path),
                            target.time_budget_seconds,
                            program_md,
                        )
                        logger.info("Meta-optimization completed for %s", target.id)
                        loop.consecutive_no_kept = 0  # Reset plateau counter
                    except (AgentTimeoutError, AgentInvocationError) as exc:
                        logger.warning("Meta-optimization failed for %s: %s", target.id, exc)

            # Callback
            if on_experiment is not None:
                on_experiment(record)

            # Milestone notification
            if self._notifications and record.outcome is Outcome.KEPT:
                await self._notifications.notify_milestone(
                    target.id, loop.kept_count, record.score,
                )

            # Update status
            state = RunnerState.RUNNING
            if record.outcome is Outcome.BLOCKED:
                state = RunnerState.BLOCKED
            elif record.outcome is Outcome.KILLED:
                state = RunnerState.KILLED
            await self._write_status(target, state, record)

        return records

    # ------------------------------------------------------------------
    # Status file
    # ------------------------------------------------------------------

    async def _write_status(
        self,
        target: OptimizationTarget,
        state: RunnerState,
        last_record: ExperimentRecord | None,
    ) -> None:
        """Write .anneal-status JSON file in the worktree."""
        worktree = Path(target.worktree_path)
        status_path = worktree / target.notifications.status_file

        payload = {
            "target_id": target.id,
            "state": state.value,
            "last_score": last_record.score if last_record else target.baseline_score,
            "experiment_count": 0,  # caller tracks; status is a snapshot
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _read_artifacts(worktree: Path, artifact_paths: list[str]) -> str:
        """Read and concatenate artifact file contents from the worktree."""
        parts: list[str] = []
        for rel_path in artifact_paths:
            full_path = worktree / rel_path
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                parts.append(f"### {rel_path}\n\n{content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Recovery helpers
    # ------------------------------------------------------------------

    async def _reset_violated(
        self,
        worktree: Path,
        violated_paths: list[str],
        git_status: list[tuple[str, str]],
    ) -> None:
        """Reset violated files: checkout tracked files, delete untracked ones."""
        import shutil

        untracked_codes = {"??"}
        status_map = {path: code for code, path in git_status}

        tracked = [p for p in violated_paths if status_map.get(p) not in untracked_codes]
        untracked = [p for p in violated_paths if status_map.get(p) in untracked_codes]

        if tracked:
            await self._git.checkout_paths(worktree, tracked)
        for path in untracked:
            full = worktree / path
            if full.is_dir():
                shutil.rmtree(full)
            elif full.exists():
                full.unlink()

    async def _handle_killed(self, worktree: Path, pre_experiment_sha: str) -> None:
        """State-aware KILLED recovery with integrity verification."""
        await self._git.cleanup_index_lock(worktree)
        await self._git.reset_hard(worktree, pre_experiment_sha)
        await self._git.clean_untracked(worktree)

        # Verify git object integrity
        fsck_ok = await self._git.fsck(worktree)
        if not fsck_ok:
            logger.error(
                "Git object database corruption detected in %s after kill recovery. "
                "Manual repair required: git -C %s fsck --full",
                worktree, worktree,
            )

    async def _safe_restore(self, worktree: Path, pre_experiment_sha: str) -> None:
        """Restore worktree to pre-experiment state on error."""
        await self._git.reset_hard(worktree, pre_experiment_sha)
        await self._git.clean_untracked(worktree)
