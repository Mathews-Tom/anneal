"""Context window budget assembly with priority-ordered truncation.

Assembles agent context within a token budget, using priority ordering
from the system design spec. Required slots are always included; optional
slots fill remaining budget in priority order.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from anneal.engine.types import ExperimentRecord, OptimizationTarget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Count tokens using tiktoken's cl100k_base encoding.

    This encoding covers GPT-4, GPT-4o, and Claude-family models
    with ~99% accuracy for context budget assembly.
    """
    if not text:
        return 0
    return len(_encoder.encode(text))


# ---------------------------------------------------------------------------
# Context slot and budget
# ---------------------------------------------------------------------------


@dataclass
class ContextSlot:
    """One slot in the context budget."""

    name: str
    content: str
    priority: int  # 1=highest
    tokens: int
    required: bool  # If True, never truncated


class ContextBudget:
    """Assembles agent context within a token budget."""

    def __init__(self, max_tokens: int = 80_000) -> None:
        self._max_tokens = max_tokens
        self._slots: list[ContextSlot] = []
        self._assembled: list[ContextSlot] = []
        self._assembled_tokens: int = 0

    def add_slot(
        self, name: str, content: str, priority: int, required: bool = False
    ) -> None:
        """Add a content slot with priority."""
        tokens = estimate_tokens(content)
        self._slots.append(
            ContextSlot(
                name=name,
                content=content,
                priority=priority,
                tokens=tokens,
                required=required,
            )
        )

    def assemble(self) -> str:
        """Assemble final context within budget.

        Priority order (from system design):
        1. System prompt (program.md + scope rules + eval description) -- required
        2. Artifact (current best version of editable files) -- required
        3. Recent history (last 5 experiment records) -- required
        4. Retrieved history (K similar experiments) -- optional
        5. Consolidated learnings (learnings.md summary) -- optional
        6. Watch files (read-only context) -- optional, truncated to fit

        Include all required slots first. Then add optional slots in
        priority order until budget is exhausted. If an optional slot
        doesn't fit, skip it. If a slot partially fits, truncate it.
        """
        self._assembled = []
        self._assembled_tokens = 0

        required = sorted(
            [s for s in self._slots if s.required], key=lambda s: s.priority
        )
        optional = sorted(
            [s for s in self._slots if not s.required], key=lambda s: s.priority
        )

        # Required slots are always included
        for slot in required:
            self._assembled.append(slot)
            self._assembled_tokens += slot.tokens

        if self._assembled_tokens > self._max_tokens:
            logger.warning(
                "Required slots (%d tokens) exceed budget (%d tokens)",
                self._assembled_tokens,
                self._max_tokens,
            )

        # Optional slots fill remaining budget in priority order
        for slot in optional:
            remaining = self._max_tokens - self._assembled_tokens
            if remaining <= 0:
                break

            if slot.tokens <= remaining:
                self._assembled.append(slot)
                self._assembled_tokens += slot.tokens
            else:
                # Truncate content to fit remaining budget
                truncated_chars = remaining * 4  # inverse of estimate_tokens
                truncated_content = slot.content[:truncated_chars]
                truncated_tokens = estimate_tokens(truncated_content)
                truncated_slot = ContextSlot(
                    name=slot.name,
                    content=truncated_content,
                    priority=slot.priority,
                    tokens=truncated_tokens,
                    required=False,
                )
                self._assembled.append(truncated_slot)
                self._assembled_tokens += truncated_tokens

        # Join assembled slots in priority order
        parts: list[str] = []
        for slot in sorted(self._assembled, key=lambda s: s.priority):
            parts.append(slot.content)

        return "\n\n".join(parts)

    @property
    def total_tokens(self) -> int:
        """Total tokens used by the assembled context."""
        return self._assembled_tokens

    @property
    def budget_remaining(self) -> int:
        """Tokens remaining in the budget."""
        return max(0, self._max_tokens - self._assembled_tokens)

    def summary(self) -> str:
        """Return a human-readable summary of slot allocation."""
        lines = [
            f"Context Budget: {self._assembled_tokens}/{self._max_tokens} tokens "
            f"({self.budget_remaining} remaining)"
        ]
        for slot in sorted(self._assembled, key=lambda s: s.priority):
            tag = "REQUIRED" if slot.required else "optional"
            lines.append(
                f"  [{slot.priority}] {slot.name}: {slot.tokens} tokens ({tag})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token cache
# ---------------------------------------------------------------------------


class TokenCache:
    """Cache token counts for static content that doesn't change between experiments."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, int]] = {}  # key -> (content_hash, tokens)

    def get(self, key: str, content: str) -> int:
        """Return cached token count, or compute and cache."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        if key in self._cache:
            cached_hash, cached_tokens = self._cache[key]
            if cached_hash == content_hash:
                return cached_tokens

        tokens = estimate_tokens(content)
        self._cache[key] = (content_hash, tokens)
        return tokens

    def invalidate(self, key: str) -> None:
        """Remove a key from the cache."""
        self._cache.pop(key, None)


# ---------------------------------------------------------------------------
# Build context for a target
# ---------------------------------------------------------------------------


def _format_experiment_record(record: ExperimentRecord) -> str:
    """Format a single experiment record for context inclusion."""
    ci = ""
    if record.score_ci_lower is not None and record.score_ci_upper is not None:
        ci = f" (CI: [{record.score_ci_lower:.4f}, {record.score_ci_upper:.4f}])"

    base = (
        f"## Experiment {record.id}\n"
        f"- Hypothesis: {record.hypothesis}\n"
        f"- Outcome: {record.outcome.value}\n"
        f"- Score: {record.score:.4f}{ci} (baseline: {record.baseline_score:.4f})\n"
        f"- Tags: {', '.join(record.tags)}\n"
        f"- Learnings: {record.learnings}\n"
        f"- Diff summary: {record.mutation_diff_summary}"
    )
    if record.per_criterion_scores:
        criterion_lines = []
        for name, score in record.per_criterion_scores.items():
            status = "PASS" if score >= 0.5 else "FAIL"
            criterion_lines.append(f"    {name}: {score:.2f} ({status})")
        base += "\n  Per-criterion:\n" + "\n".join(criterion_lines)
    return base


def _format_recent_history(history: list[ExperimentRecord]) -> str:
    """Format last 5 experiment records as structured context."""
    recent = history[-5:]
    if not recent:
        return "# Recent Experiments\nNo experiments yet."

    parts = ["# Recent Experiments"]
    for record in recent:
        parts.append(_format_experiment_record(record))
    return "\n\n".join(parts)


def _read_file_safe(path: Path) -> str | None:
    """Read a file, returning None if it doesn't exist."""
    if not path.is_file():
        return None
    return path.read_text()


def build_target_context(
    target: OptimizationTarget,
    worktree_path: Path,
    repo_root: Path,
    history: list[ExperimentRecord],
    knowledge_context: str = "",
    global_learnings: str = "",
) -> tuple[str, int]:
    """Build the complete agent context for a target.

    Returns (assembled_prompt, total_tokens).
    """
    budget = ContextBudget(max_tokens=target.agent_config.max_context_tokens)

    # Slot 1: System prompt (program.md + scope rules + eval description)
    program_path = repo_root / target.knowledge_path / "program.md"
    program_content = _read_file_safe(program_path)
    if program_content is None:
        program_content = (
            f"# Optimization Target: {target.id}\n\n"
            f"You are optimizing artifacts to improve the metric "
            f"'{target.eval_config.metric_name}' "
            f"({target.eval_config.direction.value}).\n\n"
            f"Current baseline score: {target.baseline_score}\n\n"
            f"## Scope Rules\n"
            f"- Editable files: {', '.join(target.artifact_paths)}\n"
            f"- Eval mode: {target.eval_mode.value}\n"
        )
        logger.info(
            "No program.md found at %s, using generated default", program_path
        )

    budget.add_slot("system_prompt", program_content, priority=1, required=True)

    # Slot 2: Artifact (current best version of editable files)
    artifact_parts: list[str] = []
    for artifact_rel in target.artifact_paths:
        artifact_path = worktree_path / artifact_rel
        content = _read_file_safe(artifact_path)
        if content is not None:
            artifact_parts.append(
                f"### {artifact_rel}\n```\n{content}\n```"
            )
        else:
            logger.warning("Artifact file not found: %s", artifact_path)

    artifact_content = "# Current Artifacts\n\n" + "\n\n".join(artifact_parts)
    artifact_tokens = estimate_tokens(artifact_content)

    # Warn if artifact exceeds 60% of budget
    artifact_budget_ratio = artifact_tokens / target.agent_config.max_context_tokens
    if artifact_budget_ratio > 0.60:
        logger.warning(
            "Artifact consumes %.0f%% of context budget (%d/%d tokens). "
            "Consider reducing artifact size for effective optimization.",
            artifact_budget_ratio * 100,
            artifact_tokens,
            target.agent_config.max_context_tokens,
        )

    budget.add_slot("artifact", artifact_content, priority=2, required=True)

    # Slot 3: Recent history (last 5 experiment records)
    history_content = _format_recent_history(history)
    budget.add_slot("recent_history", history_content, priority=3, required=True)

    # Slot 4: Knowledge context (retrieved history + consolidated learnings)
    if knowledge_context:
        budget.add_slot(
            "knowledge_context", knowledge_context, priority=4, required=False
        )

    # Slot 5: Global cross-project learnings (F5)
    if global_learnings:
        budget.add_slot(
            "global_learnings", global_learnings, priority=5, required=False
        )

    assembled = budget.assemble()
    logger.debug("Context assembly:\n%s", budget.summary())

    return assembled, budget.total_tokens
