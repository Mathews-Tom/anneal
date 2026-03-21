"""Problem analyzer — parse natural-language problem into structured intent."""

from __future__ import annotations

import json
import os
import re

import openai

from anneal.suggest.types import CriterionSuggestion, Domain, ProblemIntent

_ANALYSIS_PROMPT = """\
You analyze optimization problems and produce structured JSON output.

Given a problem description, artifact paths, and optional metric/eval-cmd,
determine:
1. The optimization domain (code, prompt, config, document)
2. The metric name and direction (maximize or minimize)
3. The evaluation mode (deterministic if an eval command is provided, stochastic otherwise)
4. A short target name (lowercase, hyphens, no spaces)
5. Any constraints mentioned
6. For stochastic mode: 3-5 binary evaluation criteria (yes/no questions)

Output ONLY valid JSON matching this schema:
{
  "domain": "code|prompt|config|document|unknown",
  "metric_name": "string",
  "direction": "maximize|minimize",
  "eval_mode": "deterministic|stochastic",
  "suggested_name": "string",
  "constraints": ["string"],
  "criteria": [
    {
      "name": "short_name",
      "question": "Yes/no question about the output",
      "pass_description": "What yes looks like",
      "fail_description": "What triggers no"
    }
  ]
}

Rules for criteria:
- Binary only. Yes or no. No scales.
- Specific enough that two agents would agree on the answer.
- Not so narrow the artifact can game the eval.
- 3-5 criteria maximum.
- Only include criteria for stochastic mode. For deterministic mode, return empty list.
"""


def _detect_domain(problem: str, artifact_paths: list[str]) -> Domain:
    """Fast heuristic domain detection from file extensions and keywords."""
    extensions = {p.rsplit(".", 1)[-1].lower() for p in artifact_paths if "." in p}

    code_exts = {"py", "ts", "js", "tsx", "jsx", "go", "rs", "java", "c", "cpp", "rb"}
    config_exts = {"toml", "yaml", "yml", "json", "ini", "conf", "cfg", "env"}
    doc_exts = {"md", "txt", "rst", "adoc"}
    prompt_keywords = {"skill", "prompt", "system_prompt", "instruction"}

    if extensions & code_exts:
        return Domain.CODE
    if extensions & config_exts:
        return Domain.CONFIG
    if any(kw in p.lower() for p in artifact_paths for kw in prompt_keywords):
        return Domain.PROMPT
    if extensions & doc_exts:
        return Domain.DOCUMENT

    return Domain.UNKNOWN


async def analyze_problem(
    problem: str,
    artifact_paths: list[str],
    eval_cmd: str | None = None,
    parse_cmd: str | None = None,
    metric: str | None = None,
    direction: str | None = None,
    model: str = "gpt-4.1",
) -> ProblemIntent:
    """Parse a natural-language problem into structured ProblemIntent.

    Uses an LLM to analyze the problem description and produce structured
    output. Falls back to heuristic detection if LLM fails.
    """
    # Build the user prompt
    parts = [f"Problem: {problem}"]
    parts.append(f"Artifact paths: {', '.join(artifact_paths)}")
    if eval_cmd:
        parts.append(f"Eval command provided: {eval_cmd}")
    if parse_cmd:
        parts.append(f"Parse command: {parse_cmd}")
    if metric:
        parts.append(f"Metric: {metric}")
    if direction:
        parts.append(f"Direction: {direction}")

    user_prompt = "\n".join(parts)

    # Call LLM for structured analysis
    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    response = await client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _ANALYSIS_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content or ""

    # Extract JSON from response (may be wrapped in markdown code block)
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        # Fallback to heuristic analysis
        return _heuristic_intent(problem, artifact_paths, eval_cmd, metric, direction)

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return _heuristic_intent(problem, artifact_paths, eval_cmd, metric, direction)

    # Parse criteria
    criteria = [
        CriterionSuggestion(
            name=c.get("name", f"criterion_{i}"),
            question=c.get("question", ""),
            pass_description=c.get("pass_description", ""),
            fail_description=c.get("fail_description", ""),
        )
        for i, c in enumerate(data.get("criteria", []))
    ]

    return ProblemIntent(
        problem=problem,
        domain=Domain(data.get("domain", "unknown")),
        metric_name=data.get("metric_name", metric or "score"),
        direction=data.get("direction", direction or "maximize"),
        eval_mode=data.get("eval_mode", "deterministic" if eval_cmd else "stochastic"),
        suggested_name=data.get("suggested_name", _slugify(problem)),
        constraints=data.get("constraints", []),
        criteria=criteria,
    )


def _heuristic_intent(
    problem: str,
    artifact_paths: list[str],
    eval_cmd: str | None,
    metric: str | None,
    direction: str | None,
) -> ProblemIntent:
    """Fallback intent when LLM is unavailable or fails."""
    domain = _detect_domain(problem, artifact_paths)
    eval_mode = "deterministic" if eval_cmd else "stochastic"

    # Infer direction from problem keywords
    if direction is None:
        lower = problem.lower()
        minimize_keywords = {"reduce", "minimize", "decrease", "lower", "shrink", "cut", "less"}
        direction = "minimize" if any(kw in lower for kw in minimize_keywords) else "maximize"

    return ProblemIntent(
        problem=problem,
        domain=domain,
        metric_name=metric or "score",
        direction=direction,
        eval_mode=eval_mode,
        suggested_name=_slugify(problem),
        constraints=[],
        criteria=[],
    )


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug for target names."""
    slug = re.sub(r"[^a-z0-9\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:40] if slug else "experiment"
