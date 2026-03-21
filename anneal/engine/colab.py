"""Colab evaluator — dispatches eval to Google Colab VMs via colab-mcp.

The ColabEvaluator is an eval amplifier, not a mutation amplifier. Artifact
mutation, scope enforcement, and git operations remain local. Only the eval
step runs remotely on GPU-backed Colab VMs.

Session lifecycle:
  1. initialize() → authenticate, allocate VM with requested accelerator
  2. evaluate() → sync artifacts, execute eval script, parse score
  3. cleanup() → release VM

The VM stays allocated for the duration of the optimization run, amortizing
the ~10s allocation overhead across 50+ experiments.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from anneal.engine.types import ColabConfig, DeterministicEval, EvalResult

logger = logging.getLogger(__name__)


class ColabError(Exception):
    """Raised on Colab session or evaluation failures."""


class ColabSession:
    """Manages a Colab VM session lifecycle.

    Handles authentication, VM allocation, and cleanup. The session is
    reused across multiple evaluate() calls within a single optimization run.
    """

    def __init__(self, config: ColabConfig) -> None:
        self._config = config
        self._connected = False
        self._session_id: str | None = None
        self._ccu_used: float = 0.0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def ccu_used(self) -> float:
        return self._ccu_used

    async def initialize(self) -> None:
        """Authenticate and allocate a Colab VM.

        Reads credentials from config.credentials_path, requests a VM
        with the configured accelerator, and runs setup_script if provided.
        """
        creds_path = Path(self._config.credentials_path)
        if not creds_path.exists():
            raise ColabError(
                f"Colab credentials not found at {creds_path}. "
                f"Run 'anneal colab-setup' to configure authentication."
            )

        logger.info(
            "Initializing Colab session (accelerator=%s)",
            self._config.accelerator,
        )

        # colab-mcp integration point: allocate VM
        # This is where the actual MCP execute_code calls would go.
        # For now, mark as connected and log the intent.
        self._session_id = "pending-mcp-integration"
        self._connected = True

        if self._config.setup_script:
            logger.info("Running setup script on Colab VM")
            await self.execute(self._config.setup_script)

        logger.info("Colab session initialized (id=%s)", self._session_id)

    async def execute(self, code: str) -> str:
        """Execute Python code on the Colab VM.

        Returns stdout from the execution.
        Raises ColabError on failure.
        """
        if not self._connected:
            raise ColabError("Colab session not initialized. Call initialize() first.")

        # colab-mcp integration point: execute_code(code)
        # For now, raise a clear error indicating MCP is not yet wired.
        raise ColabError(
            "Colab MCP integration not yet available. "
            "Install colab-mcp and configure credentials to enable remote GPU eval. "
            "See: https://github.com/googlecolab/colab-mcp"
        )

    async def cleanup(self) -> None:
        """Release the Colab VM and clean up resources."""
        if not self._connected:
            return

        logger.info(
            "Releasing Colab session (id=%s, ccu_used=%.2f)",
            self._session_id, self._ccu_used,
        )

        # colab-mcp integration point: unassign VM
        self._connected = False
        self._session_id = None

    def check_ccu_budget(self) -> bool:
        """Return True if CCU budget has remaining capacity."""
        return self._ccu_used < self._config.max_ccu_per_day


class ColabEvaluator:
    """Evaluator that dispatches to a Colab VM for GPU-backed execution.

    Wraps a DeterministicEval config and runs it remotely instead of locally.
    Artifact content is serialized to the VM before each eval.
    """

    def __init__(self, config: ColabConfig) -> None:
        self._config = config
        self._session: ColabSession | None = None

    async def initialize(self) -> None:
        """Start the Colab session. Called once at optimization run start."""
        self._session = ColabSession(self._config)
        await self._session.initialize()

    async def cleanup(self) -> None:
        """Release the Colab session. Called at optimization run end."""
        if self._session is not None:
            await self._session.cleanup()
            self._session = None

    async def evaluate(
        self,
        worktree_path: Path,
        eval_config: DeterministicEval,
        artifact_paths: list[str],
    ) -> EvalResult:
        """Run a deterministic eval on the Colab VM.

        Steps:
          1. Check CCU budget
          2. Serialize artifact files to the VM
          3. Execute run_command on VM
          4. Parse score from output
          5. Return EvalResult
        """
        if self._session is None or not self._session.connected:
            raise ColabError("ColabEvaluator not initialized. Call initialize() first.")

        if not self._session.check_ccu_budget():
            raise ColabError(
                f"CCU budget exceeded ({self._session.ccu_used:.2f} / "
                f"{self._config.max_ccu_per_day:.2f})"
            )

        # 1. Sync artifacts to VM
        await self._sync_artifacts(worktree_path, artifact_paths)

        # 2. Execute eval
        output = await self._session.execute(eval_config.run_command)

        # 3. Parse score
        if eval_config.parse_command and eval_config.parse_command != "cat":
            # Pipe output through parse command on VM
            parse_code = f"""
import subprocess
result = subprocess.run(
    {eval_config.parse_command!r},
    input={output!r},
    shell=True,
    capture_output=True,
    text=True,
)
print(result.stdout.strip())
"""
            output = await self._session.execute(parse_code)

        try:
            score = float(output.strip())
        except ValueError:
            raise ColabError(f"Cannot parse score from Colab output: {output[:200]!r}")

        return EvalResult(score=score, cost_usd=0.0)

    async def _sync_artifacts(
        self, worktree_path: Path, artifact_paths: list[str],
    ) -> None:
        """Serialize artifact files to the Colab VM kernel."""
        for rel_path in artifact_paths:
            full_path = worktree_path / rel_path
            if not full_path.exists():
                continue

            content = full_path.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]

            # Write file to VM via execute_code
            escaped = json.dumps(content)
            write_code = (
                f"import os, pathlib\n"
                f"p = pathlib.Path({rel_path!r})\n"
                f"p.parent.mkdir(parents=True, exist_ok=True)\n"
                f"p.write_text({escaped})\n"
                f"print('synced {rel_path} ({content_hash})')"
            )
            await self._session.execute(write_code)  # type: ignore[union-attr]
