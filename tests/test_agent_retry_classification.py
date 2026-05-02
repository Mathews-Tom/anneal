"""Tests for transient/structural failure classification (RH.2).

Covers:
  - ``_classify_api_exception`` rule table (HTTP statuses, network, JSON, timeout)
  - ``_classify_subprocess_failure`` stderr substring rules
  - Subclass relationships required for backward compatibility
  - Runner-level transient retry loop in
    :meth:`ExperimentRunner._invoke_with_transient_retries`
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from anneal.engine.agent import (
    AgentInvocationError,
    AgentInvoker,
    AgentInvocationResult,
    AgentStructuralError,
    AgentTimeoutError,
    AgentTransientError,
    _classify_api_exception,
    _classify_subprocess_failure,
)
from anneal.engine.runner import ExperimentRunner
from anneal.engine.types import AgentConfig


# ---------------------------------------------------------------------------
# _classify_api_exception
# ---------------------------------------------------------------------------


class _HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _ResponseHttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_classify_transient_http_status(status: int) -> None:
    assert _classify_api_exception(_HttpError(status)) is AgentTransientError


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_classify_structural_http_status(status: int) -> None:
    assert _classify_api_exception(_HttpError(status)) is AgentStructuralError


def test_classify_status_via_response_attribute() -> None:
    """Some SDKs surface status via ``exc.response.status_code`` rather than
    a top-level attribute. Both shapes must classify identically.
    """
    assert _classify_api_exception(_ResponseHttpError(429)) is AgentTransientError
    assert _classify_api_exception(_ResponseHttpError(401)) is AgentStructuralError


def test_classify_timeout_transient() -> None:
    assert _classify_api_exception(asyncio.TimeoutError()) is AgentTransientError


def test_classify_jsondecode_transient() -> None:
    exc = json.JSONDecodeError("expecting value", "", 0)
    assert _classify_api_exception(exc) is AgentTransientError


def test_classify_connection_error_transient() -> None:
    assert _classify_api_exception(ConnectionResetError()) is AgentTransientError
    assert _classify_api_exception(OSError("network down")) is AgentTransientError


def test_classify_unknown_exception_structural() -> None:
    """Unknown exception types default to structural — conservative bias
    prevents silent infinite-retry loops on unrecognized failure modes.
    """

    class _Custom(Exception): ...

    assert _classify_api_exception(_Custom("???")) is AgentStructuralError


# ---------------------------------------------------------------------------
# _classify_subprocess_failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stderr_text",
    [
        "Error: rate_limit exceeded",
        "RATE LIMIT reached, slow down",
        "anthropic.OverloadedError: overloaded_error",
        "service_unavailable",
        "connection reset by peer",
        "read timed out after 30s",
        "Bad gateway from upstream",
    ],
)
def test_subprocess_known_transient_stderr(stderr_text: str) -> None:
    assert _classify_subprocess_failure(stderr_text) is AgentTransientError


@pytest.mark.parametrize(
    "stderr_text",
    [
        "Authentication failed: invalid API key",
        "Unknown flag: --foo",
        "Permission denied",
        "",
        "Some other error nobody mapped yet",
    ],
)
def test_subprocess_unknown_or_structural_stderr(stderr_text: str) -> None:
    assert _classify_subprocess_failure(stderr_text) is AgentStructuralError


# ---------------------------------------------------------------------------
# Exception subclass invariants
# ---------------------------------------------------------------------------


def test_transient_is_subclass_of_invocation_error() -> None:
    """Existing ``except AgentInvocationError`` sites in runner.py must
    keep catching transient errors so behavior is additive."""
    assert issubclass(AgentTransientError, AgentInvocationError)
    assert issubclass(AgentStructuralError, AgentInvocationError)


def test_structural_not_caught_by_timeout_handler() -> None:
    """Sanity check: structural errors are NOT timeouts — runner's
    Outcome.KILLED arm must not swallow them."""
    assert not issubclass(AgentStructuralError, AgentTimeoutError)
    assert not issubclass(AgentTransientError, AgentTimeoutError)


# ---------------------------------------------------------------------------
# Runner._invoke_with_transient_retries
# ---------------------------------------------------------------------------


def _agent_config(
    *,
    max_transient_retries: int = 3,
    base: float = 0.0,
    cap: float = 0.0,
) -> AgentConfig:
    return AgentConfig(
        mode="api",
        model="gpt-4.1",
        evaluator_model="gpt-4.1-mini",
        max_transient_retries=max_transient_retries,
        transient_retry_base_seconds=base,
        transient_retry_cap_seconds=cap,
    )


def _target(config: AgentConfig) -> Any:
    return SimpleNamespace(
        id="t-test",
        agent_config=config,
        time_budget_seconds=60,
    )


def _success_result() -> AgentInvocationResult:
    return AgentInvocationResult(
        success=True,
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        hypothesis=None,
        hypothesis_source="synthesized",
        tags=[],
        raw_output="",
    )


def _runner_with_agent(invoker: AgentInvoker) -> ExperimentRunner:
    """Bypass __init__ — we only exercise one method that uses ``self._agent``
    and ``logger``. Avoids dragging in registry/git/eval setup.
    """
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner._agent = invoker  # type: ignore[attr-defined]
    return runner


@pytest.mark.asyncio
async def test_transient_retry_then_success() -> None:
    """One transient error followed by success returns the success result
    without surfacing as a CRASHED experiment."""
    invoker = MagicMock(spec=AgentInvoker)
    invoker.invoke = AsyncMock(
        side_effect=[
            AgentTransientError("429"),
            _success_result(),
        ]
    )
    runner = _runner_with_agent(invoker)
    target = _target(_agent_config(max_transient_retries=3))

    result = await runner._invoke_with_transient_retries(
        target,
        prompt="p",
        worktree=Path("/tmp"),
        deployment=False,
    )
    assert result.success is True
    assert invoker.invoke.await_count == 2


@pytest.mark.asyncio
async def test_transient_exhaustion_promotes_to_structural() -> None:
    """``max_transient_retries`` extra attempts; the (N+1)th transient
    becomes a terminal :class:`AgentStructuralError` so the runner's
    existing CRASHED arm handles it."""
    invoker = MagicMock(spec=AgentInvoker)
    invoker.invoke = AsyncMock(side_effect=AgentTransientError("429 always"))
    runner = _runner_with_agent(invoker)
    target = _target(_agent_config(max_transient_retries=2))

    with pytest.raises(AgentStructuralError) as excinfo:
        await runner._invoke_with_transient_retries(
            target,
            prompt="p",
            worktree=Path("/tmp"),
            deployment=False,
        )
    assert "Transient retries exhausted" in str(excinfo.value)
    assert invoker.invoke.await_count == 3  # initial + 2 retries
    assert isinstance(excinfo.value.__cause__, AgentTransientError)


@pytest.mark.asyncio
async def test_structural_error_no_retry() -> None:
    """Structural errors are terminal — zero retries, original exception
    propagates unchanged."""
    invoker = MagicMock(spec=AgentInvoker)
    invoker.invoke = AsyncMock(side_effect=AgentStructuralError("401"))
    runner = _runner_with_agent(invoker)
    target = _target(_agent_config(max_transient_retries=3))

    with pytest.raises(AgentStructuralError, match="401"):
        await runner._invoke_with_transient_retries(
            target,
            prompt="p",
            worktree=Path("/tmp"),
            deployment=False,
        )
    assert invoker.invoke.await_count == 1


@pytest.mark.asyncio
async def test_timeout_propagates_unwrapped() -> None:
    """Wall-clock and stall timeouts skip the retry loop entirely — the
    runner's KILLED arm must see them as :class:`AgentTimeoutError`."""
    invoker = MagicMock(spec=AgentInvoker)
    invoker.invoke = AsyncMock(side_effect=AgentTimeoutError("budget"))
    runner = _runner_with_agent(invoker)
    target = _target(_agent_config(max_transient_retries=3))

    with pytest.raises(AgentTimeoutError):
        await runner._invoke_with_transient_retries(
            target,
            prompt="p",
            worktree=Path("/tmp"),
            deployment=False,
        )
    assert invoker.invoke.await_count == 1


@pytest.mark.asyncio
async def test_max_retries_zero_reproduces_legacy_behavior() -> None:
    """``max_transient_retries=0`` means a single attempt; transient
    failures surface as structural immediately, matching pre-RH.2
    semantics where every failure went straight to CRASHED."""
    invoker = MagicMock(spec=AgentInvoker)
    invoker.invoke = AsyncMock(side_effect=AgentTransientError("429"))
    runner = _runner_with_agent(invoker)
    target = _target(_agent_config(max_transient_retries=0))

    with pytest.raises(AgentStructuralError):
        await runner._invoke_with_transient_retries(
            target,
            prompt="p",
            worktree=Path("/tmp"),
            deployment=False,
        )
    assert invoker.invoke.await_count == 1


@pytest.mark.asyncio
async def test_deployment_path_uses_invoke_deployment() -> None:
    """``deployment=True`` routes to ``invoke_deployment``; retry
    semantics are identical."""
    invoker = MagicMock(spec=AgentInvoker)
    invoker.invoke_deployment = AsyncMock(
        side_effect=[
            AgentTransientError("503"),
            _success_result(),
        ]
    )
    runner = _runner_with_agent(invoker)
    target = _target(_agent_config(max_transient_retries=3))

    result = await runner._invoke_with_transient_retries(
        target,
        prompt="p",
        worktree=Path("/tmp"),
        deployment=True,
    )
    assert result.success is True
    assert invoker.invoke_deployment.await_count == 2


@pytest.mark.asyncio
async def test_backoff_caps_at_configured_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exponential backoff respects ``transient_retry_cap_seconds`` so the
    retry loop cannot starve the experiment slot indefinitely."""
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    invoker = MagicMock(spec=AgentInvoker)
    invoker.invoke = AsyncMock(
        side_effect=[
            AgentTransientError("a"),
            AgentTransientError("b"),
            AgentTransientError("c"),
            _success_result(),
        ]
    )
    runner = _runner_with_agent(invoker)
    target = _target(
        _agent_config(max_transient_retries=5, base=10.0, cap=15.0),
    )

    await runner._invoke_with_transient_retries(
        target,
        prompt="p",
        worktree=Path("/tmp"),
        deployment=False,
    )

    # base=10, cap=15 → expected backoffs: 10, 15, 15
    assert sleeps == [10.0, 15.0, 15.0]
