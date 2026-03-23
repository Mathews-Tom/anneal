"""MCP server exposing anneal operations for IDE integration.

Provides tools for querying target status, experiment history,
and running single experiments from MCP-compatible clients.

Usage:
    uv run python -m anneal.mcp_server
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from anneal.engine.knowledge import KnowledgeStore
from anneal.engine.registry import Registry
from anneal.engine.runner import RunLoopState

mcp = FastMCP("anneal", instructions="Autonomous optimization engine for code, configs, and prompts.")


def _find_repo_root() -> Path:
    """Walk up from cwd to find .anneal/ directory."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".anneal").is_dir():
            return parent
    raise FileNotFoundError("No .anneal/ directory found. Run 'anneal init' first.")


@mcp.tool()
def anneal_status(target_id: str) -> str:
    """Get current status for an optimization target.

    Returns score, experiment count, runner state, and cost.
    """
    repo_root = _find_repo_root()
    registry = Registry(repo_root)
    target = registry.get_target(target_id)

    status_path = Path(target.worktree_path) / ".anneal-status"
    if status_path.exists():
        status_data = json.loads(status_path.read_text())
    else:
        status_data = {"state": "UNKNOWN", "last_score": target.baseline_score}

    loop_path = Path(target.knowledge_path) / ".loop-state.json"
    loop = RunLoopState.load(loop_path)

    return json.dumps({
        "target_id": target.id,
        "runner_state": status_data.get("state", "UNKNOWN"),
        "baseline_score": target.baseline_score,
        "current_score": status_data.get("last_score", target.baseline_score),
        "total_experiments": loop.total_experiments,
        "kept_count": loop.kept_count,
        "total_cost_usd": loop.cumulative_cost_usd,
        "eval_mode": target.eval_mode.value,
    }, indent=2)


@mcp.tool()
def anneal_history(target_id: str, limit: int = 10) -> str:
    """Get recent experiment records for a target.

    Returns the last N experiments with outcome, score, hypothesis, and git SHA.
    """
    repo_root = _find_repo_root()
    registry = Registry(repo_root)
    target = registry.get_target(target_id)

    knowledge = KnowledgeStore(repo_root / target.knowledge_path)
    records = knowledge.load_records(limit=limit)

    return json.dumps([
        {
            "id": r.id,
            "outcome": r.outcome.value,
            "score": r.score,
            "hypothesis": r.hypothesis,
            "git_sha": r.git_sha,
            "cost_usd": r.cost_usd,
            "duration_seconds": r.duration_seconds,
        }
        for r in records
    ], indent=2)


@mcp.tool()
def anneal_list_targets() -> str:
    """List all registered optimization targets."""
    repo_root = _find_repo_root()
    registry = Registry(repo_root)

    return json.dumps([
        {
            "id": t.id,
            "eval_mode": t.eval_mode.value,
            "baseline_score": t.baseline_score,
            "worktree": t.worktree_path,
        }
        for t in registry.all_targets()
    ], indent=2)


if __name__ == "__main__":
    mcp.run()
