"""Tests for deployment-domain runner (F6): invoke_deployment and allowed_tools logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anneal.engine.agent import AgentInvoker
from anneal.engine.types import AgentConfig


def _make_config(mode: str = "claude_code") -> AgentConfig:
    return AgentConfig(
        mode=mode,
        model="gpt-4.1",
        evaluator_model="gpt-4.1-mini",
        max_budget_usd=0.10,
    )


def _make_streaming_proc(stdout: bytes) -> MagicMock:
    """Build a subprocess mock for AgentInvoker's streaming read contract."""

    def _reader(payload: bytes) -> AsyncMock:
        chunks = iter([payload, b""])
        reader = AsyncMock()
        reader.read = AsyncMock(side_effect=lambda _n=4096: next(chunks, b""))
        return reader

    proc = MagicMock()
    proc.returncode = 0
    proc.pid = 12345
    proc.stdin = MagicMock()
    proc.stdin.drain = AsyncMock(return_value=None)
    proc.stdout = _reader(stdout)
    proc.stderr = _reader(b"")
    proc.wait = AsyncMock(return_value=0)
    return proc


class TestDeploymentMode:
    """Test that deployment mode constrains allowed tools to Read."""

    @pytest.mark.asyncio
    async def test_deployment_sets_read_only_tools(self, tmp_path: Path) -> None:
        invoker = AgentInvoker()
        config = _make_config(mode="claude_code")

        with patch(
            "anneal.engine.agent.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_proc = _make_streaming_proc(
                b'{"result": "output", "total_cost_usd": 0.01, "usage": {"input_tokens": 10, "output_tokens": 5}}',
            )
            mock_exec.return_value = mock_proc

            await invoker.invoke_deployment(
                config,
                "test prompt",
                tmp_path,
                time_budget_seconds=60,
            )

            # Verify the command includes --allowedTools Read
            call_args = mock_exec.call_args
            cmd_list = call_args[0]
            tools_idx = list(cmd_list).index("--allowedTools")
            assert cmd_list[tools_idx + 1] == "Read"

    @pytest.mark.asyncio
    async def test_deployment_via_invoke_passes_deployment_flag(
        self, tmp_path: Path
    ) -> None:
        invoker = AgentInvoker()
        config = _make_config(mode="claude_code")

        with patch(
            "anneal.engine.agent.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_proc = _make_streaming_proc(
                b'{"result": "", "total_cost_usd": 0.0, "usage": {}}',
            )
            mock_exec.return_value = mock_proc

            await invoker.invoke(
                config,
                "prompt",
                tmp_path,
                time_budget_seconds=60,
                deployment_mode=True,
            )

            call_args = mock_exec.call_args
            cmd_list = call_args[0]
            tools_idx = list(cmd_list).index("--allowedTools")
            assert cmd_list[tools_idx + 1] == "Read"

    @pytest.mark.asyncio
    async def test_normal_mode_uses_edit_write(self, tmp_path: Path) -> None:
        invoker = AgentInvoker()
        config = _make_config(mode="claude_code")

        with patch(
            "anneal.engine.agent.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_proc = _make_streaming_proc(
                b'{"result": "", "total_cost_usd": 0.0, "usage": {}}',
            )
            mock_exec.return_value = mock_proc

            await invoker.invoke(
                config,
                "prompt",
                tmp_path,
                time_budget_seconds=60,
            )

            call_args = mock_exec.call_args
            cmd_list = call_args[0]
            tools_idx = list(cmd_list).index("--allowedTools")
            assert cmd_list[tools_idx + 1] == "Edit,Write"

    @pytest.mark.asyncio
    async def test_bash_never_in_allowed_tools(self, tmp_path: Path) -> None:
        """Bash must never appear in --allowedTools per safety assertion."""
        invoker = AgentInvoker()
        config = _make_config(mode="claude_code")

        with patch(
            "anneal.engine.agent.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.side_effect = lambda *args, **kwargs: _make_streaming_proc(
                b'{"result": "", "total_cost_usd": 0.0, "usage": {}}',
            )

            for deployment_mode in [True, False]:
                await invoker.invoke(
                    config,
                    "prompt",
                    tmp_path,
                    time_budget_seconds=60,
                    deployment_mode=deployment_mode,
                )
                call_args = mock_exec.call_args
                cmd_list = call_args[0]
                tools_idx = list(cmd_list).index("--allowedTools")
                assert "Bash" not in cmd_list[tools_idx + 1]
