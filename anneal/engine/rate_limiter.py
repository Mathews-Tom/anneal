from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Global token bucket rate limiter for API calls.

    Shared across all targets to prevent concurrent stochastic evals
    from hitting API rate limits.
    """

    def __init__(
        self,
        calls_per_second: float = 10.0,
        burst: int | None = None,
    ) -> None:
        """Initialize with rate limit.

        Args:
            calls_per_second: Maximum sustained rate.
            burst: Maximum burst size. Defaults to calls_per_second.
        """
        self._rate = calls_per_second
        self._burst = float(burst if burst is not None else int(calls_per_second))
        self._tokens = self._burst
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            deficit = 1.0 - self._tokens
            wait_time = deficit / self._rate
            await asyncio.sleep(wait_time)

    async def __aenter__(self) -> RateLimiter:
        """Async context manager: acquire on enter."""
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens."""
        self._refill()
        return self._tokens


# Module-level singleton for shared rate limiting
_global_limiter: RateLimiter | None = None


def get_rate_limiter(calls_per_second: float = 10.0) -> RateLimiter:
    """Get or create the global rate limiter."""
    global _global_limiter  # noqa: PLW0603
    if _global_limiter is None:
        _global_limiter = RateLimiter(calls_per_second=calls_per_second)
    return _global_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (for testing)."""
    global _global_limiter  # noqa: PLW0603
    _global_limiter = None
