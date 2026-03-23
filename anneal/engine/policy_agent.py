"""Policy agent: continuous meta-optimizer that rewrites mutation instructions."""
from __future__ import annotations

import logging

import openai

from anneal.engine.client import compute_cost, make_client, strip_provider_prefix
from anneal.engine.types import ExperimentRecord, Outcome, PolicyConfig

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """You are a meta-optimizer analyzing experiment results to improve mutation instructions.

## Current Instructions
{current_instructions}

## Target Description
{target_description}

## Recent Experiment Results (last {window_size})
{experiment_summary}

{failure_section}

## Task
Analyze the patterns above and rewrite the mutation instructions to improve success rate.
Focus on:
1. Avoiding repeated failure patterns
2. Steering toward approaches that have worked
3. Being specific about what to try and what to avoid

Return ONLY the rewritten instruction text (no commentary, no markdown fences)."""


class PolicyAgent:
    """Analyzes failure patterns and rewrites mutation instructions.

    Continuously tunes instruction quality based on recent experiment
    outcomes. Operates at a faster cadence than plateau-triggered
    meta-optimization (which rewrites program.md).
    """

    def __init__(self, config: PolicyConfig) -> None:
        self._config = config
        self._current_instructions: str = ""
        self._last_rewrite_score: float | None = None
        self._rewrite_count: int = 0

    @property
    def current_instructions(self) -> str:
        return self._current_instructions

    @property
    def rewrite_count(self) -> int:
        return self._rewrite_count

    def should_rewrite(self, experiment_count: int) -> bool:
        """Check if it's time to rewrite based on interval."""
        if experiment_count == 0:
            return False
        return experiment_count % self._config.rewrite_interval == 0

    def compute_reward(self, current_score: float) -> float:
        """Compute performance delta since last rewrite.

        Reward = score_after_rewrite - score_before_rewrite.
        Returns 0.0 on first rewrite (no baseline).
        """
        if self._last_rewrite_score is None:
            return 0.0
        return current_score - self._last_rewrite_score

    async def rewrite_instructions(
        self,
        recent_records: list[ExperimentRecord],
        current_instructions: str,
        target_description: str,
        failure_distribution: dict[str, int] | None = None,
    ) -> tuple[str, float]:
        """Analyze recent failures and produce rewritten mutation instructions.

        Returns (new_instructions, cost_usd).
        """
        # Format experiment summary
        exp_lines: list[str] = []
        for r in recent_records:
            status = r.outcome.value
            delta = r.score - r.baseline_score
            sign = "+" if delta >= 0 else ""
            fc = ""
            if r.failure_classification:
                fc = f" [{r.failure_classification.category}]"
            exp_lines.append(
                f"- [{status}] {r.hypothesis} | score={r.score:.4f} ({sign}{delta:.4f}){fc}"
            )
        experiment_summary = "\n".join(exp_lines) if exp_lines else "No experiments yet."

        # Format failure distribution if available
        failure_section = ""
        if failure_distribution:
            dist_lines = [f"- {cat}: {count}" for cat, count in
                          sorted(failure_distribution.items(), key=lambda x: x[1], reverse=True)]
            failure_section = "## Failure Distribution\n" + "\n".join(dist_lines)

        prompt = _REWRITE_PROMPT.format(
            current_instructions=current_instructions or "(none \u2014 first rewrite)",
            target_description=target_description,
            window_size=len(recent_records),
            experiment_summary=experiment_summary,
            failure_section=failure_section,
        )

        model = self._config.model
        client = make_client(model)
        api_model = strip_provider_prefix(model)

        cost_usd = 0.0
        try:
            response = await client.chat.completions.create(
                model=api_model,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            new_instructions = (response.choices[0].message.content or "").strip()

            usage = getattr(response, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                cost_usd = compute_cost(model, input_tokens, output_tokens)

        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            logger.warning("Policy rewrite failed: %s. Keeping current instructions.", exc)
            return current_instructions, 0.0

        if not new_instructions:
            logger.warning("Policy agent returned empty instructions, keeping current.")
            return current_instructions, cost_usd

        self._current_instructions = new_instructions
        self._last_rewrite_score = (
            recent_records[-1].score if recent_records else None
        )
        self._rewrite_count += 1

        logger.info(
            "Policy rewrite #%d completed (cost=$%.4f, instructions=%d chars)",
            self._rewrite_count, cost_usd, len(new_instructions),
        )

        return new_instructions, cost_usd
