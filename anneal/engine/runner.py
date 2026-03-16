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

from anneal.engine.agent import AgentInvocationError, AgentInvoker, AgentTimeoutError
from anneal.engine.environment import GitEnvironment
from anneal.engine.eval import EvalEngine, EvalError
from anneal.engine.knowledge import KnowledgeStore
from anneal.engine.notifications import NotificationManager
from anneal.engine.registry import Registry
from anneal.engine.safety import pre_experiment_check
from anneal.engine.scope import enforce_scope, load_scope, verify_scope_hash
from anneal.engine.search import GreedySearch
from anneal.engine.types import (
    ExperimentRecord,
    OptimizationTarget,
    Outcome,
    RunnerState,
)

logger = logging.getLogger(__name__)


class ScopeIntegrityError(Exception):
    """Raised when scope.yaml hash has drifted since registration."""


class ExperimentRunner:
    """Orchestrates single experiment cycles and experiment loops for a target."""

    def __init__(
        self,
        git: GitEnvironment,
        agent_invoker: AgentInvoker,
        eval_engine: EvalEngine,
        search: GreedySearch,
        registry: Registry,
        knowledge: KnowledgeStore | None = None,
        notifications: NotificationManager | None = None,
    ) -> None:
        self._git = git
        self._agent = agent_invoker
        self._eval = eval_engine
        self._search = search
        self._registry = registry
        self._knowledge = knowledge
        self._notifications = notifications
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
        cost_usd = 0.0

        # 1. Record pre-experiment state
        pre_experiment_sha = await self._git.rev_parse(worktree, "HEAD")

        # 2. Verify scope integrity
        scope_path = worktree / target.scope_path
        if not verify_scope_hash(scope_path, target.scope_hash):
            raise ScopeIntegrityError(
                f"scope.yaml hash mismatch for target {target.id}. "
                f"Re-register with: anneal re-register --target {target.id}"
            )
        scope = load_scope(scope_path)

        # Read artifact content for prompt building and stochastic eval
        artifact_content = self._read_artifacts(worktree, target.artifact_paths)

        # Load recent history from knowledge store
        history: list[ExperimentRecord] = []
        if self._knowledge:
            history = self._knowledge.load_records(limit=10)

        # 3. Invoke agent
        prompt = self._build_prompt(target, history)
        try:
            agent_result = await self._agent.invoke(
                target.agent_config,
                prompt,
                worktree,
                target.time_budget_seconds,
            )
        except AgentTimeoutError as exc:
            duration = time.monotonic() - start_time
            # KILLED recovery: clean up index.lock, restore worktree
            await self._handle_killed(worktree, pre_experiment_sha)
            return ExperimentRecord(
                id=experiment_id,
                target_id=target.id,
                git_sha=pre_experiment_sha,
                pre_experiment_sha=pre_experiment_sha,
                timestamp=datetime.now(tz=timezone.utc),
                hypothesis="Agent timed out",
                hypothesis_source="synthesized",
                mutation_diff_summary="",
                score=target.baseline_score,
                score_ci_lower=None,
                score_ci_upper=None,
                raw_scores=None,
                baseline_score=target.baseline_score,
                outcome=Outcome.KILLED,
                failure_mode=str(exc),
                duration_seconds=duration,
                tags=[],
                learnings="",
                cost_usd=0.0,
                bootstrap_seed=0,
            )
        except AgentInvocationError as exc:
            duration = time.monotonic() - start_time
            await self._safe_restore(worktree, pre_experiment_sha)
            return ExperimentRecord(
                id=experiment_id,
                target_id=target.id,
                git_sha=pre_experiment_sha,
                pre_experiment_sha=pre_experiment_sha,
                timestamp=datetime.now(tz=timezone.utc),
                hypothesis="Agent invocation failed",
                hypothesis_source="synthesized",
                mutation_diff_summary="",
                score=target.baseline_score,
                score_ci_lower=None,
                score_ci_upper=None,
                raw_scores=None,
                baseline_score=target.baseline_score,
                outcome=Outcome.CRASHED,
                failure_mode=str(exc),
                duration_seconds=duration,
                tags=[],
                learnings="",
                cost_usd=0.0,
                bootstrap_seed=0,
            )

        cost_usd += agent_result.cost_usd
        hypothesis = agent_result.hypothesis or "No hypothesis provided"
        hypothesis_source = agent_result.hypothesis_source
        tags = agent_result.tags

        # 4. Validate scope
        git_status = await self._git.status_porcelain(worktree)
        scope_result = await enforce_scope(worktree, scope, git_status)

        if scope_result.all_blocked:
            duration = time.monotonic() - start_time
            # Reset all violating files
            if scope_result.violated_paths:
                await self._git.checkout_paths(worktree, scope_result.violated_paths)
            await self._git.clean_untracked(worktree)
            return ExperimentRecord(
                id=experiment_id,
                target_id=target.id,
                git_sha=pre_experiment_sha,
                pre_experiment_sha=pre_experiment_sha,
                timestamp=datetime.now(tz=timezone.utc),
                hypothesis=hypothesis,
                hypothesis_source=hypothesis_source,
                mutation_diff_summary="",
                score=target.baseline_score,
                score_ci_lower=None,
                score_ci_upper=None,
                raw_scores=None,
                baseline_score=target.baseline_score,
                outcome=Outcome.BLOCKED,
                failure_mode=f"All changes violated scope: {scope_result.violated_paths}",
                duration_seconds=duration,
                tags=tags,
                learnings="",
                cost_usd=cost_usd,
                bootstrap_seed=0,
            )

        # Reset violations, keep valid edits
        if scope_result.has_violations:
            await self._git.checkout_paths(worktree, scope_result.violated_paths)
            logger.warning(
                "Scope violations reset for target %s: %s",
                target.id,
                scope_result.violated_paths,
            )

        # 5. Commit valid edits
        commit_sha = await self._git.commit(
            worktree,
            f"hypothesis: {hypothesis}",
            scope_result.valid_paths,
        )

        # Re-read artifact content after mutation for stochastic eval
        artifact_content = self._read_artifacts(worktree, target.artifact_paths)

        # 6. Evaluate
        try:
            eval_result = await self._eval.evaluate(
                worktree,
                target.eval_config,
                artifact_content,
            )
        except EvalError as exc:
            duration = time.monotonic() - start_time
            await self._safe_restore(worktree, pre_experiment_sha)
            return ExperimentRecord(
                id=experiment_id,
                target_id=target.id,
                git_sha=pre_experiment_sha,
                pre_experiment_sha=pre_experiment_sha,
                timestamp=datetime.now(tz=timezone.utc),
                hypothesis=hypothesis,
                hypothesis_source=hypothesis_source,
                mutation_diff_summary="",
                score=target.baseline_score,
                score_ci_lower=None,
                score_ci_upper=None,
                raw_scores=None,
                baseline_score=target.baseline_score,
                outcome=Outcome.CRASHED,
                failure_mode=str(exc),
                duration_seconds=duration,
                tags=tags,
                learnings="",
                cost_usd=cost_usd,
                bootstrap_seed=0,
            )

        cost_usd += eval_result.cost_usd

        # 7. Decide: keep or discard
        stochastic_conf = target.eval_config.stochastic
        confidence = stochastic_conf.confidence_level if stochastic_conf else 0.95

        keep = self._search.should_keep(
            eval_result,
            target.baseline_score,
            target.baseline_raw_scores or None,
            target.eval_config.direction,
            target.eval_config.min_improvement_threshold,
            confidence,
        )

        if keep:
            outcome = Outcome.KEPT
            # Update baseline in target and persist
            target.baseline_score = eval_result.score
            if eval_result.raw_scores is not None:
                target.baseline_raw_scores = list(eval_result.raw_scores)
            self._registry.update_target(target)
            git_sha = commit_sha
        else:
            outcome = Outcome.DISCARDED
            await self._git.reset_hard(worktree, pre_experiment_sha)
            git_sha = pre_experiment_sha

        duration = time.monotonic() - start_time

        # 8. Build ExperimentRecord
        record = ExperimentRecord(
            id=experiment_id,
            target_id=target.id,
            git_sha=git_sha,
            pre_experiment_sha=pre_experiment_sha,
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
            failure_mode=None,
            duration_seconds=duration,
            tags=tags,
            learnings="",
            cost_usd=cost_usd,
            bootstrap_seed=0,
        )

        # 9. Persist to knowledge store + check consolidation
        if self._knowledge:
            self._knowledge.append_record(record)
            self._knowledge.update_index(record)
            if self._knowledge.should_consolidate():
                self._knowledge.consolidate()
                self._knowledge.regenerate_learnings()

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
        consecutive_failures = 0

        while True:
            # Check stop conditions
            if self.is_stop_requested(target.id):
                logger.info("Stop requested for target %s", target.id)
                break

            if max_experiments is not None and len(records) >= max_experiments:
                break

            if stop_score is not None and target.baseline_score >= stop_score:
                break

            if consecutive_failures >= target.max_consecutive_failures:
                logger.warning(
                    "Target %s halted: %d consecutive failures",
                    target.id,
                    consecutive_failures,
                )
                await self._write_status(target, RunnerState.HALTED, records[-1] if records else None)
                if self._notifications:
                    await self._notifications.notify_state(
                        target.id, RunnerState.HALTED,
                        f"{consecutive_failures} consecutive failures",
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
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            # Callback
            if on_experiment is not None:
                on_experiment(record)

            # Milestone notification
            if self._notifications and record.outcome is Outcome.KEPT:
                kept_count = sum(1 for r in records if r.outcome is Outcome.KEPT)
                await self._notifications.notify_milestone(
                    target.id, kept_count, record.score,
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

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        target: OptimizationTarget,
        history: list[ExperimentRecord],
    ) -> str:
        """Build prompt from program.md, artifact content, and experiment history."""
        worktree = Path(target.worktree_path)

        # Read program.md if it exists
        program_path = worktree / "targets" / target.id / "program.md"
        if program_path.exists():
            program_content = program_path.read_text(encoding="utf-8")
        else:
            program_content = self._default_program(target)

        # Read artifact content
        artifact_content = self._read_artifacts(worktree, target.artifact_paths)

        # Format recent history (last 5)
        history_text = self._format_history(history[-5:])

        # Knowledge context (retrieved similar experiments + learnings)
        knowledge_context = ""
        if self._knowledge:
            knowledge_context = self._knowledge.get_context()

        # Assemble prompt
        parts = [
            program_content,
            "",
            "## Previous Results",
            "",
            history_text if history_text else "No previous experiments.",
        ]
        if knowledge_context:
            parts.extend(["", knowledge_context])
        parts.extend([
            "",
            "--- ARTIFACT CONTENT ---",
            artifact_content,
        ])

        return "\n".join(parts)

    def _default_program(self, target: OptimizationTarget) -> str:
        """Generate a default program.md prompt when none exists."""
        editable_list = "\n".join(f"- {p}" for p in target.artifact_paths)
        return (
            f"# {target.id} — Optimization Program\n"
            f"\n"
            f"## Your Role\n"
            f"\n"
            f"You are optimizing the artifact files listed below. Your goal is to "
            f"improve the evaluation metric: {target.eval_config.metric_name} "
            f"({target.eval_config.direction.value}).\n"
            f"\n"
            f"## Editable Files\n"
            f"\n"
            f"{editable_list}\n"
            f"\n"
            f"## Constraints\n"
            f"\n"
            f"- Only modify files listed above\n"
            f"- Produce a ## Hypothesis block before making edits\n"
            f"- Produce a ## Tags block with comma-separated mutation categories\n"
        )

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

    @staticmethod
    def _format_history(records: list[ExperimentRecord]) -> str:
        """Format experiment records for inclusion in the prompt."""
        if not records:
            return ""
        lines: list[str] = []
        for i, rec in enumerate(records, 1):
            lines.append(
                f"{i}. [{rec.outcome.value}] score={rec.score:.4f} "
                f"| hypothesis: {rec.hypothesis}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Recovery helpers
    # ------------------------------------------------------------------

    async def _handle_killed(self, worktree: Path, pre_experiment_sha: str) -> None:
        """State-aware KILLED recovery per system design spec."""
        # Clean up stale index.lock
        await self._git.cleanup_index_lock(worktree)

        current_sha = await self._git.rev_parse(worktree, "HEAD")
        if current_sha != pre_experiment_sha:
            # Commit was made before kill — reset it
            await self._git.reset_hard(worktree, pre_experiment_sha)
        else:
            # No commit — clean the working tree
            await self._git.checkout_paths(worktree, ["."])
            await self._git.clean_untracked(worktree)

    async def _safe_restore(self, worktree: Path, pre_experiment_sha: str) -> None:
        """Restore worktree to pre-experiment state on error."""
        current_sha = await self._git.rev_parse(worktree, "HEAD")
        if current_sha != pre_experiment_sha:
            await self._git.reset_hard(worktree, pre_experiment_sha)
        else:
            await self._git.checkout_paths(worktree, ["."])
            await self._git.clean_untracked(worktree)
