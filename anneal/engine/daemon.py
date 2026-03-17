from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class DaemonManager:
    """Manages background anneal processes."""

    PID_FILE = "daemon.pid"
    LOG_FILE = "daemon.log"
    STOP_TIMEOUT = 10

    def __init__(self, anneal_dir: Path) -> None:
        """Initialize with the anneal/ directory path."""
        self._dir = anneal_dir
        self._pid_path = anneal_dir / self.PID_FILE
        self._log_path = anneal_dir / self.LOG_FILE

    def start(self, target_args: list[str]) -> int:
        """Start anneal run as a background daemon.

        Args:
            target_args: Arguments to pass to 'anneal run' (e.g., ['--target', 'skill'])

        Returns:
            PID of the spawned process.

        Creates:
            anneal/daemon.pid -- PID file
            anneal/daemon.log -- stdout/stderr log file

        Raises:
            RuntimeError: If a daemon is already running.
        """
        if self.is_running():
            pid = self.get_pid()
            raise RuntimeError(f"Daemon already running with PID {pid}")

        self._cleanup_stale_pid()
        self._dir.mkdir(parents=True, exist_ok=True)

        log_file = self._log_path.open("a")
        cmd = [sys.executable, "-m", "anneal", "run", *target_args]

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        self._pid_path.write_text(str(proc.pid))
        return proc.pid

    def stop(self) -> bool:
        """Stop the running daemon.

        Returns:
            True if stopped, False if not running.
        """
        pid = self.get_pid()
        if pid is None:
            return False

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            self._cleanup_stale_pid()
            return False

        deadline = time.monotonic() + self.STOP_TIMEOUT
        while time.monotonic() < deadline:
            if not self._pid_alive(pid):
                self._cleanup_stale_pid()
                return True
            time.sleep(0.1)

        # Process did not exit after SIGTERM; escalate to SIGKILL.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        self._cleanup_stale_pid()
        return True

    def is_running(self) -> bool:
        """Check if the daemon is currently running."""
        pid = self.get_pid()
        return pid is not None

    def get_pid(self) -> int | None:
        """Read PID from daemon.pid file.

        Returns None if file doesn't exist or PID is stale.
        """
        if not self._pid_path.exists():
            return None

        try:
            pid = int(self._pid_path.read_text().strip())
        except (ValueError, OSError):
            return None

        if not self._pid_alive(pid):
            return None

        return pid

    def tail_log(self, lines: int = 20) -> str:
        """Return last N lines of the daemon log."""
        if not self._log_path.exists():
            return ""

        all_lines = self._log_path.read_text().splitlines()
        return "\n".join(all_lines[-lines:])

    def _pid_alive(self, pid: int) -> bool:
        """Check whether a process with the given PID exists."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we lack permission to signal it.
            return True
        return True

    def _cleanup_stale_pid(self) -> None:
        """Remove the PID file if it exists."""
        try:
            self._pid_path.unlink()
        except FileNotFoundError:
            pass
