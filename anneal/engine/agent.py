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

from anneal.engine.client import compute_cost, make_client, strip_provider_prefix
from anneal.engine.environment import GitEnvironment
from anneal.engine.types import (
    AgentConfig,
    AgentInvocationResult,
    DiagnosisResult,
    EvalResult,
    ExperimentRecord,
)

logger = logging.getLogger(__name__)

DIAGNOSIS_SYSTEM_PROMPT = (
    "You are an optimization diagnostician. Analyze the artifact and evaluation results.\n"
    "Output valid JSON with these fields:\n"
    "- weakest_criteria: list of criterion names that failed or scored lowest\n"
    "- root_cause: one sentence explaining why these criteria failed\n"
    "- fix_category: one of \"structural\", \"content\", \"formatting\", \"logic\", \"coverage\", \"other\"\n"
    "- suggested_direction: 1-2 sentences describing what change would improve the weakest criteria"
)


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
    """Invokes a mutation agent via Claude Code subprocess or direct API call."""

    async def invoke(
        self,
        config: AgentConfig,
        prompt: str,
        worktree_path: Path,
        time_budget_seconds: int,
        deployment_mode: bool = False,
    ) -> AgentInvocationResult:
        if config.mode == "claude_code":
            return await self._invoke_claude_code(
                config, prompt, worktree_path, time_budget_seconds,
                deployment_mode=deployment_mode,
            )
        elif config.mode == "api":
            return await self._invoke_api(config, prompt, worktree_path, time_budget_seconds)
        else:
            raise AgentInvocationError(f"Unknown agent mode: {config.mode}")

    async def invoke_deployment(
        self,
        config: AgentConfig,
        prompt: str,
        worktree_path: Path,
        time_budget_seconds: int,
    ) -> AgentInvocationResult:
        """Invoke agent in deployment mode — read-only, no file modifications.

        The agent outputs proposed changes as text only.
        """
        return await self.invoke(
            config, prompt, worktree_path, time_budget_seconds,
            deployment_mode=True,
        )

    async def invoke_meta(
        self,
        config: AgentConfig,
        meta_prompt: str,
        worktree_path: Path,
        time_budget_seconds: int,
        program_md_path: Path,
    ) -> AgentInvocationResult:
        """Invoke agent to mutate program.md instead of the artifact.

        Uses a special prompt that instructs the agent to modify program.md.
        Only Edit tool is allowed, scoped to the program.md file.
        """
        content = program_md_path.read_text()
        full_prompt = (
            "You are meta-optimizing. Instead of modifying the artifact, "
            f"modify the program.md file at {program_md_path} to improve "
            f"the optimization strategy. Current program.md:\n{content}"
            f"\n\n{meta_prompt}"
        )

        if config.mode == "claude_code":
            return await self._invoke_claude_code(
                config, full_prompt, worktree_path, time_budget_seconds,
                meta_mode=True,
            )
        elif config.mode == "api":
            return await self._invoke_api(
                config, full_prompt, worktree_path, time_budget_seconds,
            )
        else:
            raise AgentInvocationError(f"Unknown agent mode: {config.mode}")

    async def _invoke_claude_code(
        self,
        config: AgentConfig,
        prompt: str,
        worktree_path: Path,
        time_budget_seconds: int,
        deployment_mode: bool = False,
        meta_mode: bool = False,
    ) -> AgentInvocationResult:
        if meta_mode:
            allowed_tools = "Edit"
        elif deployment_mode:
            allowed_tools = "Read"
        else:
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

        # Detect Claude Code error responses (exit code 0 but no result)
        is_error = response.get("is_error", False)
        subtype = response.get("subtype", "")
        if is_error or (subtype and subtype.startswith("error_")):
            raise AgentInvocationError(
                f"Claude Code returned error: subtype={subtype}, "
                f"cost=${response.get('total_cost_usd', 0):.4f}"
            )

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
        client = make_client(config.model)
        api_model = strip_provider_prefix(config.model)

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=api_model,
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
        cost_usd = compute_cost(config.model, input_tokens, output_tokens)

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

    def _build_diagnosis_prompt(
        self,
        artifact_content: str,
        eval_result: EvalResult,
        recent_history: list[ExperimentRecord],
    ) -> str:
        lines: list[str] = []

        lines.append("## Artifact")
        lines.append(artifact_content)
        lines.append("")

        lines.append("## Evaluation Score")
        lines.append(f"Overall: {eval_result.score:.4f}")
        if eval_result.ci_lower is not None and eval_result.ci_upper is not None:
            lines.append(f"CI: [{eval_result.ci_lower:.4f}, {eval_result.ci_upper:.4f}]")
        lines.append("")

        if eval_result.per_criterion_scores:
            lines.append("## Per-Criterion Scores")
            for criterion, score in sorted(
                eval_result.per_criterion_scores.items(), key=lambda kv: kv[1]
            ):
                lines.append(f"  {criterion}: {score:.4f}")
            lines.append("")

        if recent_history:
            lines.append("## Recent Experiment History")
            for record in recent_history[-5:]:
                lines.append(
                    f"  [{record.outcome.value}] {record.hypothesis} "
                    f"(score={record.score:.4f})"
                )
            lines.append("")

        return "\n".join(lines)

    async def diagnose(
        self,
        config: AgentConfig,
        artifact_content: str,
        eval_result: EvalResult,
        recent_history: list[ExperimentRecord],
        worktree_path: Path,
    ) -> DiagnosisResult:
        diagnosis_model = config.diagnosis_model or config.exploration_model or config.model
        client = make_client(diagnosis_model)
        api_model = strip_provider_prefix(diagnosis_model)
        user_prompt = self._build_diagnosis_prompt(artifact_content, eval_result, recent_history)

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=api_model,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                ),
                timeout=60,
            )
        except asyncio.TimeoutError:
            raise AgentTimeoutError("Diagnosis agent exceeded 60s timeout")
        except Exception as exc:
            raise AgentInvocationError(f"Diagnosis API call failed: {exc}") from exc

        raw = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost_usd = compute_cost(diagnosis_model, input_tokens, output_tokens)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AgentInvocationError(
                f"Diagnosis returned invalid JSON: {exc}"
            ) from exc

        data["cost_usd"] = cost_usd
        return DiagnosisResult(**data)

    async def invoke_api_text(self, config: AgentConfig, prompt: str) -> str:
        client = make_client(config.model)
        api_model = strip_provider_prefix(config.model)

        try:
            response = await client.chat.completions.create(
                model=api_model,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise AgentInvocationError(f"invoke_api_text failed: {exc}") from exc

        return response.choices[0].message.content or ""

    async def generate_drafts(
        self,
        config: AgentConfig,
        prompt: str,
        worktree_path: Path,
        time_budget_seconds: int,
        n_drafts: int,
        git: GitEnvironment,
    ) -> list[tuple[AgentInvocationResult, str]]:
        """Generate N draft mutations, capturing each as a diff.

        For API mode: concurrent invocations with varied temperature.
        For claude_code mode: sequential invocations with worktree reset between each.

        Returns list of (agent_result, diff_text) tuples.
        """
        pre_sha = await git.rev_parse(worktree_path, "HEAD")
        drafts: list[tuple[AgentInvocationResult, str]] = []

        if config.mode == "api":
            # Concurrent API calls with varied temperature
            tasks = []
            for i in range(n_drafts):
                temp_offset = (i - n_drafts // 2) * 0.1
                draft_config = config.model_copy(update={
                    "temperature": max(0.0, min(2.0, config.temperature + temp_offset)),
                    "max_budget_usd": config.max_budget_usd / n_drafts,
                })
                tasks.append(self._invoke_api(draft_config, prompt, worktree_path, time_budget_seconds))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning("Draft generation failed: %s", result)
                    continue
                # For API mode, result is the text response — no worktree diff
                # We store the raw_output as a pseudo-diff (the agent's proposed changes)
                drafts.append((result, result.raw_output))
        else:
            # Sequential claude_code invocations with diff capture
            for i in range(n_drafts):
                draft_config = config.model_copy(update={
                    "max_budget_usd": config.max_budget_usd / n_drafts,
                })
                try:
                    result = await self._invoke_claude_code(
                        draft_config, prompt, worktree_path, time_budget_seconds,
                    )
                    diff_text = await git.capture_diff(worktree_path)
                    drafts.append((result, diff_text))
                except (AgentTimeoutError, AgentInvocationError) as exc:
                    logger.warning("Draft %d/%d failed: %s", i + 1, n_drafts, exc)
                finally:
                    # Reset worktree for next draft
                    await git.reset_hard(worktree_path, pre_sha)
                    await git.clean_untracked(worktree_path)

        return drafts
