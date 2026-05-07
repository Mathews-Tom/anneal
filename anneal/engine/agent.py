"""Agent invoker — supports subprocess and API agent modes.

Dispatches based on AgentConfig.mode to either shell out to a local coding
agent CLI or call an OpenAI-compatible chat completions endpoint directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import tempfile
from pathlib import Path
from typing import Literal

from anneal.engine.client import (
    compute_cost,
    effective_temperature,
    make_client,
    strip_provider_prefix,
)
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
    '- fix_category: one of "structural", "content", "formatting", "logic", "coverage", "other"\n'
    "- suggested_direction: 1-2 sentences describing what change would improve the weakest criteria"
)


class AgentInvocationError(Exception):
    """Base error for agent invocation failures."""


class AgentTimeoutError(AgentInvocationError):
    """Agent exceeded time budget."""


class AgentStalledError(AgentTimeoutError):
    """Agent emitted no output for stall_timeout_seconds.

    Subclass of AgentTimeoutError so runner-level handlers that route
    timeouts to ``Outcome.KILLED`` keep working without modification, but
    distinguishable when retry classification needs the original signal.
    """


class AgentTransientError(AgentInvocationError):
    """Retry-eligible failure (rate limit, 5xx, network blip, partial JSON).

    Runner wraps invocation in an exponential-backoff retry loop that
    catches this class and reissues the call within the same experiment
    slot. Promoted to ``AgentStructuralError`` once the retry budget is
    exhausted.
    """


class AgentStructuralError(AgentInvocationError):
    """Terminal failure (auth error, schema break, agent-declined task).

    Surfaces immediately as ``Outcome.CRASHED`` with no retry attempt.
    """


_SUBPROCESS_MODES: frozenset[str] = frozenset({"claude_code", "codex_exec"})


_TRANSIENT_HTTP_STATUSES: frozenset[int] = frozenset(
    {408, 425, 429, 500, 502, 503, 504}
)

_TRANSIENT_STDERR_PATTERNS: tuple[str, ...] = (
    "rate_limit",
    "rate limit",
    "overloaded",
    "overloaded_error",
    "service_unavailable",
    "temporarily unavailable",
    "connection reset",
    "connection refused",
    "connection timed out",
    "read timed out",
    "bad gateway",
    "gateway timeout",
    "ecconnreset",
    "etimedout",
)


def _classify_api_exception(exc: BaseException) -> type[AgentInvocationError]:
    """Map an arbitrary exception raised during an LLM API call to a
    transient/structural class.

    Conservative: defaults to ``AgentStructuralError`` on unknown types so
    silent infinite-retry loops are impossible without an explicit
    classification rule.
    """
    if isinstance(exc, asyncio.TimeoutError):
        return AgentTransientError
    if isinstance(exc, json.JSONDecodeError):
        return AgentTransientError
    if isinstance(exc, (ConnectionError, OSError)):
        return AgentTransientError
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
    if isinstance(status, int) and status in _TRANSIENT_HTTP_STATUSES:
        return AgentTransientError
    return AgentStructuralError


def _classify_subprocess_failure(stderr_text: str) -> type[AgentInvocationError]:
    """Classify a non-zero subprocess exit. Stderr substring match is
    conservative: structural by default, transient only on known signals.
    """
    haystack = stderr_text.lower()
    if any(needle in haystack for needle in _TRANSIENT_STDERR_PATTERNS):
        return AgentTransientError
    return AgentStructuralError


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the subprocess's session group. Idempotent."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _extract_hypothesis(text: str) -> str | None:
    """Extract hypothesis text after '## Hypothesis' header."""
    match = re.search(r"## Hypothesis\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return content if content else None
    return None


def extract_code_block(text: str) -> str | None:
    """Extract the content of the longest fenced code block in ``text``.

    API-mode mutations (see ``runner.run_one``) have no file-editing tool,
    so the agent is prompted to emit the complete replacement artifact
    inside a single fenced code block. This helper parses that block.

    Accepts both triple-backtick and triple-tilde fences with an optional
    language hint on the opening line. When multiple blocks are present,
    returns the longest (agents sometimes include small illustrative
    blocks before the full replacement). Returns None when no fenced
    block is found; the caller should treat that as "agent produced no
    applyable mutation" and let scope enforcement emit a BLOCKED record.
    """
    pattern = re.compile(
        r"^(?P<fence>```|~~~)[^\n]*\n(?P<body>.*?)^(?P=fence)\s*$",
        re.DOTALL | re.MULTILINE,
    )
    matches = [m.group("body") for m in pattern.finditer(text)]
    if not matches:
        return None
    return max(matches, key=len)


def _extract_tags(text: str) -> list[str]:
    """Extract comma-separated tags after '## Tags' header."""
    match = re.search(r"## Tags\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        if raw:
            return [tag.strip() for tag in raw.split(",") if tag.strip()]
    return []


def _extract_json_object(text: str) -> dict[str, object]:
    """Parse the first JSON object from an LLM text response."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected JSON object", stripped, 0)
    return parsed


def _diagnosis_result_from_data(data: dict[str, object]) -> DiagnosisResult:
    """Validate untyped JSON data as a DiagnosisResult."""
    return DiagnosisResult.model_validate(data)


def _extract_codex_json_event_text(stdout_text: str) -> str:
    """Best-effort extraction of a final assistant message from Codex JSONL."""
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("message", "item", "event"):
            value = event.get(key)
            if isinstance(value, dict):
                content = value.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    parts: list[str] = []
                    for part in content:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict):
                            text = part.get("text") or part.get("content")
                            if isinstance(text, str):
                                parts.append(text)
                    if parts:
                        return "\n".join(parts).strip()
        result = event.get("result") or event.get("content") or event.get("text")
        if isinstance(result, str) and result.strip():
            return result.strip()
    return ""


def _number_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _int_value(value: object) -> int:
    number = _number_value(value)
    if number is None:
        return 0
    return int(number)


def _extract_cost_value(data: object) -> float | None:
    if isinstance(data, dict):
        for key in ("total_cost_usd", "cost_usd"):
            value = _number_value(data.get(key))
            if value is not None:
                return value
        for value in data.values():
            nested = _extract_cost_value(value)
            if nested is not None:
                return nested
    elif isinstance(data, list):
        for item in data:
            nested = _extract_cost_value(item)
            if nested is not None:
                return nested
    return None


def _extract_usage_tokens(data: object) -> tuple[int, int] | None:
    if isinstance(data, dict):
        input_tokens = (
            _int_value(data.get("input_tokens"))
            or _int_value(data.get("prompt_tokens"))
            or _int_value(data.get("cached_input_tokens"))
        )
        output_tokens = (
            _int_value(data.get("output_tokens"))
            or _int_value(data.get("completion_tokens"))
            or _int_value(data.get("generated_tokens"))
        )
        if input_tokens or output_tokens:
            return input_tokens, output_tokens

        for key in ("usage", "token_usage", "usage_metadata", "response"):
            nested = _extract_usage_tokens(data.get(key))
            if nested is not None:
                return nested

        for value in data.values():
            nested = _extract_usage_tokens(value)
            if nested is not None:
                return nested
    elif isinstance(data, list):
        for item in data:
            nested = _extract_usage_tokens(item)
            if nested is not None:
                return nested
    return None


def _extract_codex_usage(stdout_text: str, model: str) -> tuple[float, int, int]:
    """Extract cost and token telemetry from Codex JSONL stdout."""
    cost_usd: float | None = None
    input_tokens = 0
    output_tokens = 0

    for line in stdout_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_cost = _extract_cost_value(event)
        if event_cost is not None:
            cost_usd = event_cost

        event_usage = _extract_usage_tokens(event)
        if event_usage is not None:
            input_tokens, output_tokens = event_usage

    if cost_usd is None and (input_tokens or output_tokens):
        cost_usd = compute_cost(model, input_tokens, output_tokens)

    return cost_usd or 0.0, input_tokens, output_tokens


class AgentInvoker:
    """Invokes a mutation agent via subprocess or direct API call."""

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
                config,
                prompt,
                worktree_path,
                time_budget_seconds,
                deployment_mode=deployment_mode,
            )
        elif config.mode == "codex_exec":
            return await self._invoke_codex_exec(
                config,
                prompt,
                worktree_path,
                time_budget_seconds,
                deployment_mode=deployment_mode,
            )
        elif config.mode == "api":
            return await self._invoke_api(
                config, prompt, worktree_path, time_budget_seconds
            )
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
            config,
            prompt,
            worktree_path,
            time_budget_seconds,
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
                config,
                full_prompt,
                worktree_path,
                time_budget_seconds,
                meta_mode=True,
            )
        elif config.mode == "codex_exec":
            return await self._invoke_codex_exec(
                config,
                full_prompt,
                worktree_path,
                time_budget_seconds,
                deployment_mode=False,
            )
        elif config.mode == "api":
            return await self._invoke_api(
                config,
                full_prompt,
                worktree_path,
                time_budget_seconds,
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
        assert "Bash" not in allowed_tools, "Bash must never appear in --allowedTools"

        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--allowedTools",
            allowed_tools,
            "--max-budget-usd",
            str(config.max_budget_usd),
            "--model",
            config.model,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(worktree_path.resolve()),
            start_new_session=True,
        )

        stdout_bytes, stderr_bytes = await self._run_with_stall_detection(
            proc,
            prompt.encode(),
            wall_clock_seconds=time_budget_seconds,
            stall_seconds=config.stall_timeout_seconds,
        )

        stderr_text = stderr_bytes.decode(errors="replace")

        if proc.returncode != 0:
            raise _classify_subprocess_failure(stderr_text)(
                f"Claude Code exited with code {proc.returncode}: {stderr_text}"
            )

        stdout_text = stdout_bytes.decode(errors="replace")

        try:
            response = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            raise AgentTransientError(
                f"Invalid JSON response from Claude Code: {exc}"
            ) from exc

        # Detect Claude Code error responses (exit code 0 but no result)
        is_error = response.get("is_error", False)
        subtype = response.get("subtype", "")
        if is_error or (subtype and subtype.startswith("error_")):
            raise AgentStructuralError(
                f"Claude Code returned error: subtype={subtype}, "
                f"cost=${response.get('total_cost_usd', 0):.4f}"
            )

        cost_usd = float(response.get("total_cost_usd", response.get("cost_usd", 0.0)))
        usage = response.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        raw_output = response.get("result", "")

        hypothesis = _extract_hypothesis(raw_output)
        hypothesis_source: Literal["agent", "synthesized"] = (
            "agent" if hypothesis is not None else "synthesized"
        )
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

    async def _invoke_codex_exec(
        self,
        config: AgentConfig,
        prompt: str,
        worktree_path: Path,
        time_budget_seconds: int,
        deployment_mode: bool = False,
    ) -> AgentInvocationResult:
        sandbox = "read-only" if deployment_mode else "workspace-write"
        fd, output_name = tempfile.mkstemp(prefix="anneal-codex-", suffix=".md")
        os.close(fd)
        output_path = Path(output_name)
        proc: asyncio.subprocess.Process | None = None

        cmd = [
            "codex",
            "exec",
            "--cd",
            str(worktree_path.resolve()),
            "--model",
            config.model,
            "--sandbox",
            sandbox,
            "--color",
            "never",
            "--json",
            "--output-last-message",
            str(output_path),
            "--ephemeral",
            "-",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(worktree_path.resolve()),
                start_new_session=True,
            )
            stdout_bytes, stderr_bytes = await self._run_with_stall_detection(
                proc,
                prompt.encode(),
                wall_clock_seconds=time_budget_seconds,
                stall_seconds=config.stall_timeout_seconds,
            )
        except Exception:
            if proc is not None and proc.returncode is None:
                _kill_process_group(proc)
            output_path.unlink(missing_ok=True)
            raise

        stderr_text = stderr_bytes.decode(errors="replace")

        if proc.returncode != 0:
            raise _classify_subprocess_failure(stderr_text)(
                f"Codex exec exited with code {proc.returncode}: {stderr_text}"
            )

        stdout_text = stdout_bytes.decode(errors="replace")
        raw_output = ""
        try:
            if output_path.exists():
                raw_output = output_path.read_text(encoding="utf-8").strip()
        finally:
            output_path.unlink(missing_ok=True)

        if not raw_output:
            raw_output = _extract_codex_json_event_text(stdout_text)
        if not raw_output:
            raise AgentTransientError("Codex exec completed without a final message")

        cost_usd, input_tokens, output_tokens = _extract_codex_usage(
            stdout_text,
            config.model,
        )
        hypothesis = _extract_hypothesis(raw_output)
        hypothesis_source: Literal["agent", "synthesized"] = (
            "agent" if hypothesis is not None else "synthesized"
        )
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

    async def _run_with_stall_detection(
        self,
        proc: asyncio.subprocess.Process,
        prompt: bytes,
        wall_clock_seconds: int,
        stall_seconds: int,
    ) -> tuple[bytes, bytes]:
        """Drive ``proc`` to completion while watching for silent stalls.

        Streams stdout/stderr concurrently, refreshing ``last_activity_ts``
        on every chunk. A watchdog SIGKILLs the process group if the gap
        between activity exceeds ``stall_seconds`` (``stall_seconds=0``
        disables the watchdog). ``wall_clock_seconds`` remains the hard
        backstop and is enforced as before via :class:`AgentTimeoutError`.

        Returns the concatenated ``(stdout, stderr)`` byte buffers on
        success. Raises :class:`AgentStalledError` on stall (subclass of
        :class:`AgentTimeoutError`) or :class:`AgentTimeoutError` on
        wall-clock exhaustion.
        """
        loop = asyncio.get_running_loop()
        last_activity = loop.time()
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        stall_seen: list[float] = []

        async def pump(stream: asyncio.StreamReader | None, sink: list[bytes]) -> None:
            nonlocal last_activity
            if stream is None:
                return
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    return
                sink.append(chunk)
                last_activity = loop.time()

        async def watchdog() -> None:
            if stall_seconds <= 0:
                return
            tick = max(1.0, min(stall_seconds / 4, 15.0))
            while proc.returncode is None:
                await asyncio.sleep(tick)
                elapsed = loop.time() - last_activity
                if elapsed > stall_seconds:
                    stall_seen.append(elapsed)
                    _kill_process_group(proc)
                    return

        if proc.stdin is not None:
            try:
                proc.stdin.write(prompt)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            proc.stdin.close()

        pump_stdout = asyncio.create_task(pump(proc.stdout, stdout_chunks))
        pump_stderr = asyncio.create_task(pump(proc.stderr, stderr_chunks))
        watch_task = asyncio.create_task(watchdog())
        wait_task = asyncio.create_task(proc.wait())

        try:
            await asyncio.wait_for(
                asyncio.gather(pump_stdout, pump_stderr, watch_task, wait_task),
                timeout=wall_clock_seconds,
            )
        except asyncio.TimeoutError:
            _kill_process_group(proc)
            for task in (pump_stdout, pump_stderr, watch_task, wait_task):
                task.cancel()
            await asyncio.gather(
                pump_stdout,
                pump_stderr,
                watch_task,
                wait_task,
                return_exceptions=True,
            )
            raise AgentTimeoutError(
                f"Agent exceeded wall-clock budget of {wall_clock_seconds}s"
            )

        if stall_seen:
            raise AgentStalledError(
                f"Agent emitted no output for {stall_seen[0]:.1f}s "
                f"(stall_timeout={stall_seconds}s)"
            )

        return b"".join(stdout_chunks), b"".join(stderr_chunks)

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
                    temperature=effective_temperature(config.model, config.temperature),
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=time_budget_seconds,
            )
        except asyncio.TimeoutError:
            raise AgentTimeoutError(
                f"API agent exceeded time budget of {time_budget_seconds}s"
            )
        except Exception as exc:
            raise _classify_api_exception(exc)(f"API call failed: {exc}") from exc

        raw_output = response.choices[0].message.content or ""

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost_usd = compute_cost(config.model, input_tokens, output_tokens)

        hypothesis = _extract_hypothesis(raw_output)
        hypothesis_source: Literal["agent", "synthesized"] = (
            "agent" if hypothesis is not None else "synthesized"
        )
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
            lines.append(
                f"CI: [{eval_result.ci_lower:.4f}, {eval_result.ci_upper:.4f}]"
            )
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
        diagnosis_model = (
            config.diagnosis_model or config.exploration_model or config.model
        )
        user_prompt = self._build_diagnosis_prompt(
            artifact_content, eval_result, recent_history
        )

        if config.mode == "codex_exec":
            diagnosis_config = config.model_copy(update={"model": diagnosis_model})
            result = await self._invoke_codex_exec(
                diagnosis_config,
                f"{DIAGNOSIS_SYSTEM_PROMPT}\n\n{user_prompt}\n\n"
                "Return only the JSON object, with no markdown fence.",
                worktree_path,
                time_budget_seconds=60,
                deployment_mode=True,
            )
            try:
                data = _extract_json_object(result.raw_output)
            except json.JSONDecodeError as exc:
                raise AgentInvocationError(
                    f"Diagnosis returned invalid JSON: {exc}"
                ) from exc
            data["cost_usd"] = result.cost_usd
            return _diagnosis_result_from_data(data)

        client = make_client(diagnosis_model)
        api_model = strip_provider_prefix(diagnosis_model)

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=api_model,
                    temperature=effective_temperature(diagnosis_model, 0.3),
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
            data = _extract_json_object(raw)
        except json.JSONDecodeError as exc:
            raise AgentInvocationError(
                f"Diagnosis returned invalid JSON: {exc}"
            ) from exc

        data["cost_usd"] = cost_usd
        return _diagnosis_result_from_data(data)

    async def invoke_api_text(self, config: AgentConfig, prompt: str) -> str:
        client = make_client(config.model)
        api_model = strip_provider_prefix(config.model)

        try:
            response = await client.chat.completions.create(
                model=api_model,
                temperature=effective_temperature(config.model, 0.7),
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
                draft_config = config.model_copy(
                    update={
                        "temperature": max(
                            0.0, min(2.0, config.temperature + temp_offset)
                        ),
                        "max_budget_usd": config.max_budget_usd / n_drafts,
                    }
                )
                tasks.append(
                    self._invoke_api(
                        draft_config, prompt, worktree_path, time_budget_seconds
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning("Draft generation failed: %s", result)
                    continue
                # For API mode, result is the text response — no worktree diff
                # We store the raw_output as a pseudo-diff (the agent's proposed changes)
                drafts.append((result, result.raw_output))
        else:
            # Sequential subprocess invocations with diff capture
            for i in range(n_drafts):
                draft_config = config.model_copy(
                    update={
                        "max_budget_usd": config.max_budget_usd / n_drafts,
                    }
                )
                try:
                    result = await self.invoke(
                        draft_config,
                        prompt,
                        worktree_path,
                        time_budget_seconds,
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
