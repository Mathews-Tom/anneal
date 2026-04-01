# Security Model

## Threat Model

anneal executes user-defined shell commands (`--run-cmd`, `run_command`, `parse_command`) in the evaluation loop. These commands run in the user's shell with the user's permissions.

The optimization agent does NOT have shell access. It is restricted to file-edit tools only (`Edit`, `Write`, or `Read` depending on mode). The Bash tool is explicitly excluded from `--allowedTools` on every agent invocation — enforced by an assertion in `anneal/engine/agent.py`.

The primary attack surface is:

- Malicious or misconfigured `run_command` / `parse_command` in eval configs
- A compromised artifact that triggers unexpected behavior when the eval command runs against it
- An agent that attempts to modify its own scope or eval definitions to influence scoring

## Isolation Boundaries

**Agent tool restrictions**
The Claude Code subprocess is spawned with `--allowedTools Edit,Write` (mutation mode) or `--allowedTools Read` (deployment mode). `Bash` is never in the allowed tools list. This is asserted at call time in `agent.py` and enforced by the Claude Code CLI. The agent cannot execute shell commands.

**Git worktree isolation**
Every experiment runs in a dedicated git worktree. Changes are committed before the eval command runs. If the agent violates scope, the worktree is reset to the pre-experiment SHA. The main working tree is never touched during an experiment.

**Scope enforcement**
`scope.yaml` defines which files the agent may edit (`editable`) and which are protected (`immutable`). After the agent exits, `anneal/engine/scope.py` inspects every changed file against the editable list. Path traversal attempts (paths starting with `..` or absolute paths) are unconditionally rejected via `os.path.normpath`. Files that violate scope are reset via `git checkout`.

`scope.yaml`, `metrics.yaml`, and (for stochastic targets) `eval_criteria.toml` are required to be in the `immutable` list — enforced at registration time in `scope.py:validate_scope`. This prevents the agent from modifying its own evaluation criteria.

**Scope hash verification**
The SHA-256 hash of `scope.yaml` is recorded at registration time. Before each experiment, the hash is re-verified. A mismatch raises `ScopeHashMismatchError` and halts the run. This detects out-of-band tampering.

**Eval command execution**
Eval commands are run by the user's shell, not by the agent. The agent has no visibility into or influence over eval command execution. Each subprocess is time-boxed; on timeout, `.kill()` is called.

**Agent process isolation**
The Claude Code subprocess is spawned with `start_new_session=True`, creating a new process group. On timeout, `os.killpg(os.getpgid(proc.pid), signal.SIGKILL)` kills the entire process group, preventing orphaned child processes.

**Budget enforcement**
`anneal/engine/safety.py` estimates cost before each experiment and compares against the configured daily budget cap. If the estimate would exceed the cap, the run is paused before the agent is invoked. This is not a security boundary, but it limits unintended cost accumulation.

## Recommendations

- Run anneal inside Docker or a sandboxed CI environment when optimizing code from untrusted sources. See the [Dockerfile](Dockerfile) in this repository for a ready-to-use image.
- Never run as root. anneal does not require elevated privileges.
- Review `scope.yaml` before starting any optimization run. Confirm the `editable` list contains only the files you intend the agent to modify.
- Keep eval commands simple and auditable. Avoid eval pipelines that fetch remote resources or execute network calls.
- Use `--dry-run` (when available) to preview cost estimates before committing to a long run.

## Reporting Vulnerabilities

Open a [private security advisory](https://github.com/Mathews-Tom/anneal/security/advisories/new) on GitHub. Do not report security issues in public issues.

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce
- The version of anneal affected
- Any suggested mitigations if known
