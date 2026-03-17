"""Agent invoker — supports Claude Code subprocess mode and API mode.

Dispatches based on AgentConfig.mode to either shell out to Claude Code
or call an OpenAI-compatible chat completions endpoint directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from pathlib import Path

import openai

from anneal.engine.types import AgentConfig, AgentInvocationResult

logger = logging.getLogger(__name__)


class AgentInvocationError(Exception):
    """Base error for agent invocation failures."""


class AgentTimeoutError(AgentInvocationError):
    """Agent exceeded time budget."""


# Per-million-token costs: (input, output)
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


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


def _make_api_client(model: str) -> openai.AsyncOpenAI:
    """Create OpenAI-compatible client for the model's provider."""
    if model.startswith("gemini-"):
        return openai.AsyncOpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    if model.startswith("claude-"):
        return openai.AsyncOpenAI(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.anthropic.com/v1/",
        )
    if model.startswith("gpt-"):
        return openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise AgentInvocationError(f"No API client mapping for model: {model}")


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from token counts using _MODEL_COSTS."""
    costs = _MODEL_COSTS.get(model)
    if costs is None:
        logger.warning("No cost data for model %s; reporting $0.00", model)
        return 0.0
    input_cost_per_tok, output_cost_per_tok = costs[0] / 1_000_000, costs[1] / 1_000_000
    return input_tokens * input_cost_per_tok + output_tokens * output_cost_per_tok


class AgentInvoker:
    """Invokes a mutation agent via Claude Code subprocess or direct API call."""

    async def invoke(
        self,
        config: AgentConfig,
        prompt: str,
        worktree_path: Path,
        time_budget_seconds: int,
    ) -> AgentInvocationResult:
        if config.mode == "claude_code":
            return await self._invoke_claude_code(config, prompt, worktree_path, time_budget_seconds)
        elif config.mode == "api":
            return await self._invoke_api(config, prompt, worktree_path, time_budget_seconds)
        else:
            raise AgentInvocationError(f"Unknown agent mode: {config.mode}")

    async def _invoke_claude_code(
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
            "--no-session-persistence",
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

        cost_usd = float(response.get("total_cost_usd", response.get("cost_usd", 0.0)))
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

    async def _invoke_api(
        self,
        config: AgentConfig,
        prompt: str,
        worktree_path: Path,
        time_budget_seconds: int,
    ) -> AgentInvocationResult:
        client = _make_api_client(config.model)

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=config.model,
                    temperature=config.temperature,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=time_budget_seconds,
            )
        except asyncio.TimeoutError:
            raise AgentTimeoutError(
                f"API agent exceeded time budget of {time_budget_seconds}s"
            )

        raw_output = response.choices[0].message.content or ""

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost_usd = _compute_cost(config.model, input_tokens, output_tokens)

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
