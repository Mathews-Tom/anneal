"""Unified OpenAI-compatible client factory with local model routing.

Routes model names to the appropriate API endpoint by prefix:
  gemini-*   → Google AI API
  gpt-*      → OpenAI API
  claude-*   → Anthropic API
  ollama/*   → http://localhost:11434/v1 (Ollama)
  lmstudio/* → http://localhost:1234/v1 (LM Studio)
  local/*    → ANNEAL_LOCAL_BASE_URL env var

Both eval.py and agent.py import from this module.
"""

from __future__ import annotations

import logging
import os

import openai

logger = logging.getLogger(__name__)

# Default local server endpoints
_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# Per-million-token costs: (input, output). Local models are $0.
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-5": (5.0, 20.0),
    "gpt-5-mini": (1.0, 4.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-haiku-4-5": (0.8, 4.0),
}


def make_client(model: str) -> openai.AsyncOpenAI:
    """Build an OpenAI-compatible async client routed by model prefix.

    Raises ValueError for unknown model prefixes without cloud API keys.
    """
    # Local model routing
    if model.startswith("ollama/"):
        return openai.AsyncOpenAI(
            api_key="ollama",
            base_url=_OLLAMA_BASE_URL,
        )

    if model.startswith("lmstudio/"):
        return openai.AsyncOpenAI(
            api_key="lmstudio",
            base_url=_LMSTUDIO_BASE_URL,
        )

    if model.startswith("local/"):
        base_url = os.environ.get("ANNEAL_LOCAL_BASE_URL", "http://localhost:8000/v1")
        return openai.AsyncOpenAI(
            api_key="local",
            base_url=base_url,
        )

    # Cloud API routing
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

    # Fallback: assume OpenAI-compatible with OPENAI_API_KEY
    logger.warning("Unknown model prefix '%s', defaulting to OpenAI client", model)
    return openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def strip_provider_prefix(model: str) -> str:
    """Strip the provider prefix from a model name for API calls.

    ollama/llama3.1:8b → llama3.1:8b
    lmstudio/qwen2.5:7b → qwen2.5:7b
    local/my-model → my-model
    gpt-4.1 → gpt-4.1 (no prefix to strip)
    """
    for prefix in ("ollama/", "lmstudio/", "local/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def is_local_model(model: str) -> bool:
    """Check if a model is routed to a local server."""
    return model.startswith(("ollama/", "lmstudio/", "local/"))


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from token counts. Returns $0.00 for local models."""
    if is_local_model(model):
        return 0.0

    # Strip prefix for cost lookup
    costs = _MODEL_COSTS.get(model)
    if costs is None:
        logger.warning("No cost data for model %s; reporting $0.00", model)
        return 0.0

    input_rate, output_rate = costs[0] / 1_000_000, costs[1] / 1_000_000
    return input_tokens * input_rate + output_tokens * output_rate


async def check_local_server(model: str) -> tuple[bool, str]:
    """Check if a local server is healthy and the model is available.

    Returns (healthy, message).
    """
    if not is_local_model(model):
        return False, f"Not a local model: {model}"

    client = make_client(model)
    api_model = strip_provider_prefix(model)

    try:
        response = await client.chat.completions.create(
            model=api_model,
            temperature=0.0,
            max_tokens=5,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        content = response.choices[0].message.content or ""
        return True, f"Model {api_model} responded: {content.strip()[:20]}"
    except openai.APIConnectionError:
        return False, f"Cannot connect to local server for {model}"
    except openai.APIStatusError as exc:
        return False, f"Server error for {model}: {exc.status_code} {exc.message}"
    except Exception as exc:
        return False, f"Unexpected error for {model}: {type(exc).__name__}: {exc}"
