"""Tests for meta-optimization (F8): AgentInvoker.invoke_meta."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from anneal.engine.agent import AgentInvoker, AgentInvocationError
from anneal.engine.types import AgentConfig


def _make_config(mode: str = "claude_code") -> AgentConfig:
    return AgentConfig(
        mode=mode,
        model="gpt-4.1",
        evaluator_model="gpt-4.1-mini",
        max_budget_usd=0.10,
    )


class TestInvokeMeta:
    """Tests for invoke_meta command construction and prompt assembly."""

    @pytest.mark.asyncio
    async def test_meta_mode_sets_edit_only_tools(self, tmp_path: Path) -> None:
        invoker = AgentInvoker()
        config = _make_config(mode="claude_code")

        program_md = tmp_path / "program.md"
        program_md.write_text("# Optimization Program\nStep 1: do stuff")

        with patch(
            "anneal.engine.agent.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b'{"result": "modified", "total_cost_usd": 0.01, "usage": {"input_tokens": 100, "output_tokens": 50}}',
                b"",
            )
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            mock_exec.return_value = mock_proc

            await invoker.invoke_meta(
                config,
                meta_prompt="Improve the strategy",
                worktree_path=tmp_path,
                time_budget_seconds=60,
                program_md_path=program_md,
            )

            call_args = mock_exec.call_args
            cmd_list = call_args[0]
            tools_idx = list(cmd_list).index("--allowedTools")
            assert cmd_list[tools_idx + 1] == "Edit"

    @pytest.mark.asyncio
    async def test_meta_prompt_includes_program_content(self, tmp_path: Path) -> None:
        invoker = AgentInvoker()
        config = _make_config(mode="claude_code")

        program_md = tmp_path / "program.md"
        program_md.write_text("UNIQUE_PROGRAM_CONTENT_XYZ")

        with patch(
            "anneal.engine.agent.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b'{"result": "", "total_cost_usd": 0.0, "usage": {}}',
                b"",
            )
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            mock_exec.return_value = mock_proc

            await invoker.invoke_meta(
                config,
                meta_prompt="Improve it",
                worktree_path=tmp_path,
                time_budget_seconds=60,
                program_md_path=program_md,
            )

            # The prompt sent to stdin should contain the program.md content
            call_args = mock_proc.communicate.call_args
            stdin_bytes = call_args[1]["input"] if "input" in call_args[1] else call_args[0][0]
            stdin_text = stdin_bytes.decode()
            assert "UNIQUE_PROGRAM_CONTENT_XYZ" in stdin_text

    @pytest.mark.asyncio
    async def test_meta_prompt_includes_meta_prompt_text(self, tmp_path: Path) -> None:
        invoker = AgentInvoker()
        config = _make_config(mode="claude_code")

        program_md = tmp_path / "program.md"
        program_md.write_text("content")

        with patch(
            "anneal.engine.agent.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b'{"result": "", "total_cost_usd": 0.0, "usage": {}}',
                b"",
            )
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            mock_exec.return_value = mock_proc

            await invoker.invoke_meta(
                config,
                meta_prompt="SPECIFIC_META_INSTRUCTION",
                worktree_path=tmp_path,
                time_budget_seconds=60,
                program_md_path=program_md,
            )

            call_args = mock_proc.communicate.call_args
            stdin_bytes = call_args[1]["input"] if "input" in call_args[1] else call_args[0][0]
            stdin_text = stdin_bytes.decode()
            assert "SPECIFIC_META_INSTRUCTION" in stdin_text

    @pytest.mark.asyncio
    async def test_meta_prompt_contains_meta_optimization_marker(self, tmp_path: Path) -> None:
        invoker = AgentInvoker()
        config = _make_config(mode="claude_code")

        program_md = tmp_path / "program.md"
        program_md.write_text("content")

        with patch(
            "anneal.engine.agent.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b'{"result": "", "total_cost_usd": 0.0, "usage": {}}',
                b"",
            )
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            mock_exec.return_value = mock_proc

            await invoker.invoke_meta(
                config,
                meta_prompt="improve",
                worktree_path=tmp_path,
                time_budget_seconds=60,
                program_md_path=program_md,
            )

            call_args = mock_proc.communicate.call_args
            stdin_bytes = call_args[1]["input"] if "input" in call_args[1] else call_args[0][0]
            stdin_text = stdin_bytes.decode()
            assert "meta-optimizing" in stdin_text

    def test_meta_unknown_mode_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be 'claude_code' or 'api'"):
            _make_config(mode="unknown")

    @pytest.mark.asyncio
    async def test_meta_bash_excluded(self, tmp_path: Path) -> None:
        invoker = AgentInvoker()
        config = _make_config(mode="claude_code")

        program_md = tmp_path / "program.md"
        program_md.write_text("content")

        with patch(
            "anneal.engine.agent.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                b'{"result": "", "total_cost_usd": 0.0, "usage": {}}',
                b"",
            )
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            mock_exec.return_value = mock_proc

            await invoker.invoke_meta(
                config,
                meta_prompt="improve",
                worktree_path=tmp_path,
                time_budget_seconds=60,
                program_md_path=program_md,
            )

            call_args = mock_exec.call_args
            cmd_list = call_args[0]
            tools_idx = list(cmd_list).index("--allowedTools")
            assert "Bash" not in cmd_list[tools_idx + 1]
