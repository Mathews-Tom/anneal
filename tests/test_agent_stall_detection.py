"""Tests for AgentInvoker._run_with_stall_detection (RH.1).

Exercises the activity-heartbeat watchdog with real bash subprocesses so
the streaming + watchdog interaction is tested end-to-end rather than via
mocks of asyncio internals. All subprocesses are bounded by short
``time_budget`` and ``stall`` windows so the suite finishes in seconds.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time

import pytest

from anneal.engine.agent import (
    AgentInvocationError,
    AgentInvoker,
    AgentStalledError,
    AgentTimeoutError,
)


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="bash required to drive stall-detection subprocesses",
)


async def _spawn_bash(script: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )


@pytest.mark.asyncio
async def test_silent_subprocess_killed_at_stall_window() -> None:
    """A subprocess that produces no output for longer than ``stall_seconds``
    is killed via SIGKILL and surfaces as :class:`AgentStalledError`.
    """
    invoker = AgentInvoker()
    proc = await _spawn_bash("sleep 30; echo done")

    start = time.monotonic()
    with pytest.raises(AgentStalledError):
        await invoker._run_with_stall_detection(
            proc,
            prompt=b"",
            wall_clock_seconds=20,
            stall_seconds=1,
        )
    elapsed = time.monotonic() - start

    # Detection happens within stall window + one watchdog tick (~stall/4).
    assert elapsed < 5, f"stall detection too slow: {elapsed:.2f}s"
    assert elapsed >= 1, f"stall detection fired before window: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_active_subprocess_runs_to_completion() -> None:
    """A subprocess that emits a heartbeat well within ``stall_seconds``
    completes normally without false-positive stalls.
    """
    invoker = AgentInvoker()
    proc = await _spawn_bash(
        "for i in 1 2 3 4; do echo tick-$i; sleep 0.2; done; exit 0"
    )

    stdout, stderr = await invoker._run_with_stall_detection(
        proc,
        prompt=b"",
        wall_clock_seconds=10,
        stall_seconds=2,
    )
    assert proc.returncode == 0
    assert b"tick-1" in stdout and b"tick-4" in stdout
    assert stderr == b""


@pytest.mark.asyncio
async def test_steady_activity_hits_wallclock() -> None:
    """A subprocess that keeps emitting forever escapes the stall watchdog
    but is killed by the wall-clock timeout via :class:`AgentTimeoutError`.
    """
    invoker = AgentInvoker()
    proc = await _spawn_bash("while true; do echo .; sleep 0.05; done")

    start = time.monotonic()
    with pytest.raises(AgentTimeoutError) as excinfo:
        await invoker._run_with_stall_detection(
            proc,
            prompt=b"",
            wall_clock_seconds=1,
            stall_seconds=10,
        )
    elapsed = time.monotonic() - start

    # Stall watchdog must NOT fire here — wall-clock should
    assert not isinstance(excinfo.value, AgentStalledError)
    assert elapsed < 4, f"wall-clock enforcement too slow: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_stall_disabled_when_zero() -> None:
    """``stall_seconds=0`` reproduces pre-RH.1 behavior — silent
    subprocesses run until wall-clock or natural exit.
    """
    invoker = AgentInvoker()
    proc = await _spawn_bash("sleep 0.5; echo woke")

    stdout, _ = await invoker._run_with_stall_detection(
        proc,
        prompt=b"",
        wall_clock_seconds=5,
        stall_seconds=0,
    )
    assert proc.returncode == 0
    assert b"woke" in stdout


@pytest.mark.asyncio
async def test_stdin_prompt_is_delivered() -> None:
    """Streaming prompt write reaches the subprocess and round-trips."""
    invoker = AgentInvoker()
    proc = await _spawn_bash("cat")

    stdout, _ = await invoker._run_with_stall_detection(
        proc,
        prompt=b"hello-from-test\n",
        wall_clock_seconds=5,
        stall_seconds=2,
    )
    assert proc.returncode == 0
    assert stdout == b"hello-from-test\n"


@pytest.mark.asyncio
async def test_no_orphan_children_after_stall() -> None:
    """SIGKILL on the process group leaves no descendants of the killed
    subprocess. Verified by polling /proc-equivalent via ``ps``.
    """
    if shutil.which("ps") is None:
        pytest.skip("ps required to verify orphan cleanup")

    invoker = AgentInvoker()
    proc = await _spawn_bash("sleep 60 & sleep 60")
    pid = proc.pid

    with pytest.raises(AgentStalledError):
        await invoker._run_with_stall_detection(
            proc,
            prompt=b"",
            wall_clock_seconds=10,
            stall_seconds=1,
        )

    # Give the OS a brief moment to reap.
    await asyncio.sleep(0.3)
    res = await asyncio.create_subprocess_exec(
        "ps",
        "-o",
        "pid=",
        "-g",
        str(pid),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await res.communicate()
    survivors = [line for line in out.decode().split() if line.strip()]
    assert not survivors, f"orphaned pids in group {pid}: {survivors}"


@pytest.mark.asyncio
async def test_stalled_error_is_timeout_subclass() -> None:
    """Existing handlers that catch :class:`AgentTimeoutError` continue
    to catch stalls without modification — required for runner-level
    backward compatibility.
    """
    assert issubclass(AgentStalledError, AgentTimeoutError)
    assert issubclass(AgentStalledError, AgentInvocationError)


@pytest.mark.asyncio
async def test_returncode_preserved_on_clean_exit() -> None:
    """Non-zero exits without stderr-known transient signatures keep their
    returncode visible to the caller (used by classification logic).
    """
    invoker = AgentInvoker()
    proc = await _spawn_bash("echo out; echo err >&2; exit 3")

    stdout, stderr = await invoker._run_with_stall_detection(
        proc,
        prompt=b"",
        wall_clock_seconds=5,
        stall_seconds=2,
    )
    assert proc.returncode == 3
    assert b"out" in stdout
    assert b"err" in stderr


@pytest.mark.asyncio
async def test_pid_dies_on_wallclock_kill() -> None:
    """Wall-clock SIGKILL actually terminates the subprocess (not just
    raises) — verified by ``waitpid``.
    """
    invoker = AgentInvoker()
    proc = await _spawn_bash("sleep 30")
    pid = proc.pid

    with pytest.raises(AgentTimeoutError):
        await invoker._run_with_stall_detection(
            proc,
            prompt=b"",
            wall_clock_seconds=1,
            stall_seconds=10,
        )

    # waitpid with WNOHANG returns (0, 0) if still alive; (pid, status) if reaped
    # asyncio already reaped via proc.wait(); checking via os.kill 0
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
