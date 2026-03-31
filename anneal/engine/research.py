"""Research operator for external knowledge injection during optimization plateaus."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from anneal.engine.client import compute_cost, make_client, strip_provider_prefix
from anneal.engine.types import AgentConfig, ResearchConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchSuggestion:
    """One technique suggestion from external knowledge."""

    technique: str
    description: str
    source: str
    relevance: str


@dataclass(frozen=True)
class ResearchResult:
    """Output from the research operator."""

    suggestions: list[ResearchSuggestion]
    cost_usd: float
    query_used: str


class ResearchOperator:
    """Queries external knowledge sources when optimization plateaus.

    Provides technique suggestions to inject into mutation context.
    Uses an LLM with web search capability or tool-augmented generation.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._total_cost: float = 0.0
        self._disabled: bool = False
        self._consecutive_failures: int = 0

    async def research(
        self,
        target_description: str,
        current_artifact_summary: str,
        failed_criteria: list[str],
        recent_hypotheses: list[str],
        agent_config: AgentConfig,
    ) -> ResearchResult | None:
        """Run research step. Returns None if disabled or over budget."""
        if self._disabled:
            logger.info("Research operator disabled (consecutive failures)")
            return None

        if self._total_cost >= self._config.max_budget_usd:
            logger.info(
                "Research operator budget exhausted ($%.4f/$%.4f)",
                self._total_cost, self._config.max_budget_usd,
            )
            return None

        model = self._config.model or agent_config.model
        query = self._build_query(
            target_description, failed_criteria, recent_hypotheses,
        )

        prompt = (
            f"You are a research assistant. Search for techniques relevant to "
            f"this optimization problem:\n\n"
            f"## Problem\n{target_description}\n\n"
            f"## Current artifact (summary)\n{current_artifact_summary[:500]}\n\n"
            f"## Failed criteria\n{', '.join(failed_criteria)}\n\n"
            f"## Already tried\n{chr(10).join(f'- {h[:100]}' for h in recent_hypotheses[-5:])}\n\n"
            f"Suggest 2-3 techniques that might help. For each, provide:\n"
            f"1. Technique name\n"
            f"2. Description (2-3 sentences)\n"
            f"3. Source (paper, blog, docs URL, or 'general knowledge')\n"
            f"4. Why it's relevant to this specific problem\n\n"
            f"Output valid JSON array of objects with keys: "
            f"technique, description, source, relevance"
        )

        try:
            client = make_client(model)
            response = await client.chat.completions.create(
                model=strip_provider_prefix(model),
                temperature=0.5,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.choices[0].message.content or "[]"
            cost = compute_cost(
                strip_provider_prefix(model),
                getattr(response.usage, "prompt_tokens", 0),
                getattr(response.usage, "completion_tokens", 0),
            )
            self._total_cost += cost

            data = json.loads(raw)
            suggestions_raw = data if isinstance(data, list) else data.get("suggestions", [])
            suggestions = [
                ResearchSuggestion(
                    technique=s.get("technique", "Unknown"),
                    description=s.get("description", ""),
                    source=s.get("source", "general knowledge"),
                    relevance=s.get("relevance", ""),
                )
                for s in suggestions_raw[: self._config.max_suggestions]
            ]

            return ResearchResult(
                suggestions=suggestions,
                cost_usd=cost,
                query_used=query,
            )

        except Exception as exc:
            logger.warning("Research operator failed: %s", exc)
            return None

    def record_outcome(self, improved: bool) -> None:
        """Track whether research-informed mutations improve outcomes."""
        if improved:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._config.disable_after_failures:
                self._disabled = True
                logger.warning(
                    "Research operator disabled: %d consecutive failures",
                    self._consecutive_failures,
                )

    def _build_query(
        self,
        target_description: str,
        failed_criteria: list[str],
        recent_hypotheses: list[str],
    ) -> str:
        """Build a research query from target context."""
        return (
            f"{target_description}. "
            f"Improve: {', '.join(failed_criteria[:3])}. "
            f"Avoid: {', '.join(h[:50] for h in recent_hypotheses[-3:])}"
        )

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def is_disabled(self) -> bool:
        return self._disabled
