"""Failure taxonomy: classification, distribution, and blind spot detection."""
from __future__ import annotations

import json
import logging

import openai

from anneal.engine.client import compute_cost, make_client, strip_provider_prefix
from anneal.engine.types import ExperimentRecord, FailureClassification, Outcome

logger = logging.getLogger(__name__)


SEED_CATEGORIES: list[dict[str, str]] = [
    {"category": "output_format", "description": "Output structure/schema violation"},
    {"category": "logic_error", "description": "Incorrect reasoning or calculation"},
    {"category": "regression", "description": "Degraded a previously-passing criterion"},
    {"category": "scope_violation", "description": "Edits outside permitted scope"},
    {"category": "syntax_error", "description": "Code/config fails to parse"},
    {"category": "semantic_drift", "description": "Correct format but wrong meaning"},
    {"category": "over_optimization", "description": "Improved metric by gaming evaluation"},
    {"category": "incomplete_edit", "description": "Partial change leaving inconsistent state"},
]

_CLASSIFICATION_PROMPT = """Classify this failed experiment into one of the categories below.

## Hypothesis
{hypothesis}

## Failure Mode
{failure_mode}

## Score
{score}

## Categories
{categories}

Respond with ONLY a JSON object:
{{"category": "<category_name>", "description": "<what went wrong>", "fix_direction": "<suggested axis for future attempts>", "confidence": <0.0-1.0>}}"""


class FailureTaxonomy:
    """Extensible registry of known failure modes with LLM-based classification."""

    def __init__(self, custom_categories: list[dict[str, str]] | None = None) -> None:
        self._categories = list(SEED_CATEGORIES)
        if custom_categories:
            self._categories.extend(custom_categories)

    @property
    def categories(self) -> list[dict[str, str]]:
        return list(self._categories)

    @property
    def category_names(self) -> list[str]:
        return [c["category"] for c in self._categories]

    async def classify(
        self,
        hypothesis: str,
        failure_mode: str | None,
        score: float,
        model: str,
    ) -> tuple[FailureClassification, float]:
        """Classify a failure using a lightweight LLM call.

        Returns (classification, cost_usd).
        """
        categories_text = "\n".join(
            f"- {c['category']}: {c['description']}" for c in self._categories
        )
        prompt = _CLASSIFICATION_PROMPT.format(
            hypothesis=hypothesis,
            failure_mode=failure_mode or "unknown",
            score=f"{score:.4f}",
            categories=categories_text,
        )

        client = make_client(model)
        api_model = strip_provider_prefix(model)

        try:
            response = await client.chat.completions.create(
                model=api_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            logger.warning("Taxonomy classification failed: %s", exc)
            return self._fallback_classify(hypothesis, failure_mode), 0.0

        raw = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        cost_usd = 0.0
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            cost_usd = compute_cost(model, input_tokens, output_tokens)

        try:
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            data = json.loads(raw)
            category = data.get("category", "unknown")
            # Validate category is in the taxonomy
            if category not in self.category_names:
                category = self._best_match_category(category)
            return FailureClassification(
                category=category,
                description=data.get("description", ""),
                fix_direction=data.get("fix_direction", ""),
                confidence=float(data.get("confidence", 0.5)),
            ), cost_usd
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Failed to parse taxonomy response: %s", raw[:200])
            return self._fallback_classify(hypothesis, failure_mode), cost_usd

    def _fallback_classify(
        self, hypothesis: str, failure_mode: str | None
    ) -> FailureClassification:
        """Rule-based fallback when LLM classification fails."""
        fm = (failure_mode or "").lower()
        if "scope" in fm:
            return FailureClassification(
                category="scope_violation", description=failure_mode or "",
                fix_direction="Restrict edits to permitted scope", confidence=0.8,
            )
        if "syntax" in fm or "parse" in fm:
            return FailureClassification(
                category="syntax_error", description=failure_mode or "",
                fix_direction="Ensure output parses correctly", confidence=0.7,
            )
        if "verifier" in fm:
            return FailureClassification(
                category="syntax_error",
                description=f"Verifier gate failed: {failure_mode}",
                fix_direction="Fix verifier violation before resubmitting", confidence=0.6,
            )
        if "constraint" in fm:
            return FailureClassification(
                category="regression", description=failure_mode or "",
                fix_direction="Maintain constraint thresholds", confidence=0.6,
            )
        return FailureClassification(
            category="logic_error", description=failure_mode or hypothesis,
            fix_direction="Review mutation logic", confidence=0.3,
        )

    def _best_match_category(self, raw_category: str) -> str:
        """Find the closest matching category name."""
        raw_lower = raw_category.lower().replace(" ", "_").replace("-", "_")
        for name in self.category_names:
            if raw_lower == name or raw_lower in name or name in raw_lower:
                return name
        return "logic_error"

    @staticmethod
    def distribution(records: list[ExperimentRecord]) -> dict[str, int]:
        """Count failure classifications across records."""
        counts: dict[str, int] = {}
        for r in records:
            if r.failure_classification is not None:
                cat = r.failure_classification.category
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def blind_spot_check(self, records: list[ExperimentRecord]) -> list[str]:
        """Identify categories with zero attributions despite repeated failures.

        Returns category names that have never been attributed but could
        plausibly apply given the failure count.
        """
        failed = [r for r in records if r.outcome in (Outcome.DISCARDED, Outcome.BLOCKED)]
        if len(failed) < 10:
            return []

        dist = self.distribution(records)
        attributed_categories = set(dist.keys())
        all_categories = set(self.category_names)
        unattributed = all_categories - attributed_categories

        return sorted(unattributed)
