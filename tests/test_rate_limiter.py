from __future__ import annotations

import time

import pytest

from anneal.engine.rate_limiter import RateLimiter, get_rate_limiter, reset_rate_limiter


@pytest.fixture(autouse=True)
def _clean_singleton() -> None:
    """Reset the module-level singleton before each test."""
    reset_rate_limiter()


@pytest.mark.asyncio
async def test_acquire_consumes_token() -> None:
    limiter = RateLimiter(calls_per_second=10.0, burst=5)
    before = limiter.available_tokens
    await limiter.acquire()
    after = limiter.available_tokens
    assert after < before


@pytest.mark.asyncio
async def test_multiple_acquires_within_burst_succeed_immediately() -> None:
    limiter = RateLimiter(calls_per_second=10.0, burst=5)
    start = time.monotonic()
    for _ in range(5):
        await limiter.acquire()
    elapsed = time.monotonic() - start
    # All 5 should complete nearly instantly (well under 1 second)
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_acquire_waits_when_tokens_exhausted() -> None:
    # 2 calls/sec, burst=1 → first acquire instant, second must wait ~0.5s
    limiter = RateLimiter(calls_per_second=2.0, burst=1)
    await limiter.acquire()  # consume the single burst token
    start = time.monotonic()
    await limiter.acquire()  # must wait for refill
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3


@pytest.mark.asyncio
async def test_get_rate_limiter_returns_singleton() -> None:
    a = get_rate_limiter(10.0)
    b = get_rate_limiter(10.0)
    assert a is b


@pytest.mark.asyncio
async def test_reset_rate_limiter_clears_singleton() -> None:
    a = get_rate_limiter(10.0)
    reset_rate_limiter()
    b = get_rate_limiter(10.0)
    assert a is not b


@pytest.mark.asyncio
async def test_async_context_manager() -> None:
    limiter = RateLimiter(calls_per_second=10.0, burst=5)
    before = limiter.available_tokens
    async with limiter as ctx:
        assert ctx is limiter
    after = limiter.available_tokens
    assert after < before
