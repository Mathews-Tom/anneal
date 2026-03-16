"""Agent invoker — calls Claude Code as a subprocess, parses structured output,
and extracts cost/usage metadata.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
from pathlib import Path

from anneal.engine.types import AgentConfig, AgentInvocationResult


class AgentInvocationError(Exception):
    """Base error for agent invocation failures."""


class AgentTimeoutError(AgentInvocationError):
    """Agent exceeded time budget."""


def _extract_hypothesis(text: str) -> str | None:
    """Extract hypothesis text after '## Hypothesis' header."""
    match = re.search(
        r"## Hypothesis\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL
    )
    if match:
        content = match.group(1).strip()
        return content if content else None
    return None


def _extract_tags(text: str) -> list[str]:
    """Extract comma-separated tags after '## Tags' header."""
    match = re.search(r"## Tags\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        if raw:
            return [tag.strip() for tag in raw.split(",") if tag.strip()]
    return []


class AgentInvoker:
    """Invokes Claude Code as a subprocess and parses the structured response."""

    async def invoke(
        self,
        config: AgentConfig,
        prompt: str,
        worktree_path: Path,
        time_budget_seconds: int,
    ) -> AgentInvocationResult:
        allowed_tools = "Edit,Write"
        assert "Bash" not in allowed_tools, (
            "Bash must never appear in --allowedTools"
        )

        cmd = [
            "claude",
            "-p",
            "--output-format", "json",
            "--allowedTools", allowed_tools,
            "--max-budget-usd", str(config.max_budget_usd),
            "--model", config.model,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(worktree_path.resolve()),
            start_new_session=True,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=prompt.encode()),
                timeout=time_budget_seconds,
            )
        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise AgentTimeoutError(
                f"Agent exceeded time budget of {time_budget_seconds}s"
            )

        stderr_text = stderr_bytes.decode(errors="replace")

        if proc.returncode != 0:
            raise AgentInvocationError(
                f"Claude Code exited with code {proc.returncode}: {stderr_text}"
            )

        stdout_text = stdout_bytes.decode(errors="replace")

        try:
            response = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            raise AgentInvocationError(
                f"Invalid JSON response from Claude Code: {exc}"
            ) from exc

        cost_usd = float(response.get("cost_usd", 0.0))
        usage = response.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        raw_output = response.get("result", "")

        hypothesis = _extract_hypothesis(raw_output)
        hypothesis_source: str = "agent" if hypothesis is not None else "synthesized"
        tags = _extract_tags(raw_output)

        return AgentInvocationResult(
            success=True,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            hypothesis=hypothesis,
            hypothesis_source=hypothesis_source,
            tags=tags,
            raw_output=raw_output,
        )
