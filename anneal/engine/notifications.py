"""Webhook notification hooks with retry and fallback.

Fires notifications on runner state transitions and milestone events.
Writes enriched status files to worktrees for local observability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx

from anneal.engine.types import NotificationConfig, RunnerState

logger = logging.getLogger(__name__)
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logger.addHandler(_stderr_handler)
logger.setLevel(logging.WARNING)


class NotificationManager:
    """Manages webhook notifications, milestone alerts, and status file writes."""

    def __init__(self, config: NotificationConfig) -> None:
        self._config = config
        self._notify_on: set[str] = set(config.notify_on)

    async def notify_state(
        self,
        target_id: str,
        state: RunnerState,
        message: str,
        score: float | None = None,
        experiment_count: int = 0,
    ) -> bool:
        """Fire notification if state is in notify_on list. Returns True if delivered."""
        if state.value not in self._notify_on:
            return False

        if self._config.webhook_url is None:
            return False

        payload = {
            "event": "state_change",
            "target_id": target_id,
            "state": state.value,
            "message": message,
            "timestamp": time.time(),
            "experiment_count": experiment_count,
        }
        if score is not None:
            payload["score"] = score

        return await self._deliver(payload)

    async def notify_milestone(
        self,
        target_id: str,
        kept_count: int,
        score: float,
    ) -> bool:
        """Fire notification if kept_count is a multiple of milestone_interval."""
        if self._config.milestone_interval <= 0:
            return False

        if kept_count == 0 or kept_count % self._config.milestone_interval != 0:
            return False

        if self._config.webhook_url is None:
            return False

        payload = {
            "event": "milestone",
            "target_id": target_id,
            "kept_count": kept_count,
            "score": score,
            "timestamp": time.time(),
        }

        return await self._deliver(payload)

    async def _deliver(self, payload: dict) -> bool:
        """Deliver payload to primary webhook, falling back to secondary on failure."""
        primary = self._config.webhook_url
        if primary is None:
            return False

        if await self._fire_webhook(primary, payload):
            return True

        fallback = self._config.fallback_webhook_url
        if fallback is not None:
            if await self._fire_webhook(fallback, payload):
                return True

        logger.warning(
            "All webhook endpoints failed for payload: %s",
            json.dumps(payload, default=str),
        )
        return False

    async def _fire_webhook(self, url: str, payload: dict) -> bool:
        """POST JSON to url with retry (config.webhook_retry_count attempts, exponential backoff)."""
        retry_count = self._config.webhook_retry_count
        base_delay = self._config.webhook_retry_delay_seconds

        for attempt in range(retry_count):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                    return True
            except Exception as exc:
                if attempt < retry_count - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Webhook POST to %s failed (attempt %d/%d): %s — retrying in %.1fs",
                        url,
                        attempt + 1,
                        retry_count,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "Webhook POST to %s failed after %d attempts: %s",
                        url,
                        retry_count,
                        exc,
                    )

        return False

    async def write_status(
        self,
        worktree_path: Path,
        target_id: str,
        state: RunnerState,
        score: float,
        experiment_count: int,
        budget_spent: float = 0.0,
        budget_cap: float = 0.0,
        tags_frequency: dict[str, int] | None = None,
        last_experiment_time: float = 0.0,
    ) -> None:
        """Write enriched .anneal-status JSON to worktree."""
        budget_remaining = max(0.0, budget_cap - budget_spent)

        status = {
            "target_id": target_id,
            "state": state.value,
            "score": score,
            "experiment_count": experiment_count,
            "timestamp": time.time(),
            "budget_spent": budget_spent,
            "budget_cap": budget_cap,
            "budget_remaining": budget_remaining,
            "tags_frequency": tags_frequency or {},
            "last_experiment_time": last_experiment_time,
        }

        status_path = worktree_path / self._config.status_file
        status_path.write_text(json.dumps(status, indent=2) + "\n")
