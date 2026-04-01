# ADR-004: Scope Enforcement Design

## Status
Accepted

## Context

The agent is an LLM with file-edit tools. Without constraints, it can write to any file in the worktree — including its own mutation instructions (`program.md`), its eval criteria (`eval_criteria.toml`), configuration (`metrics.yaml`, `scope.yaml`), other registered targets' configs, or unrelated project files.

The risks this creates:

- **Eval gaming**: The agent modifies `eval_criteria.toml` or `metrics.yaml` to make the scoring easier, allowing it to report improvements without actually improving the artifact.
- **Cross-target contamination**: The agent edits another target's config or scope, breaking other optimization runs.
- **Repository corruption**: The agent overwrites unrelated source files, tests, or CI configs.

Two enforcement models were considered:

- **Intercept writes at the tool level**: Hook into the file-write tool to block writes outside permitted paths in real time. This requires deep integration with the agent invocation layer and is not possible in Claude Code subprocess mode, where the agent writes directly to the filesystem.
- **Post-write validation**: Let the agent write freely, then inspect `git status --porcelain` after the agent exits and before eval runs. Reset any unauthorized changes. Enforce before executing eval, so unauthorized mutations never influence the score.

## Decision

Scope enforcement uses post-write validation (`anneal/engine/scope.py`).

`scope.yaml` in each target's worktree directory declares two required lists:
- `editable`: glob patterns or exact paths the agent is permitted to modify.
- `immutable`: files that must never be changed. Always includes `scope.yaml` and `metrics.yaml`. Stochastic targets additionally require `eval_criteria.toml`.

At registration time, `validate_scope` checks:
- Both `editable` and `immutable` are non-empty.
- No path appears in both lists.
- Stochastic targets have `eval_criteria.toml` in `immutable`.
- The `scope.yaml` of a sibling target is not listed as `editable`.

At runtime, after each agent invocation, `enforce_scope` receives the parsed `git status --porcelain` output as `(status_code, path)` tuples. For every changed or new file:
- Modified/added (`M`, `A`, `??`): accepted if the path matches any entry in `editable`.
- Deleted (`D`): accepted only if the path is in both `editable` and `allowed_deletions`.
- Renamed (`R`): both the source and destination must be in their respective permitted lists.
- Unknown status codes: treated as violations.

Path matching supports exact paths, directory prefixes (entries ending with `/`), and glob patterns (`*`, `?`). Path traversal is blocked unconditionally: paths beginning with `..` or that are absolute are rejected regardless of `editable` contents.

The `scope.yaml` file itself is hash-verified at experiment start using a SHA-256 digest stored at registration time. If the hash has drifted, the run is halted with a `ScopeIntegrityError`.

When violations are found, the runner resets the violated paths via `git checkout -- <path>` or removes untracked files, then either retries (if some paths were valid) or marks the experiment as BLOCKED.

## Consequences

- The agent cannot modify eval configuration, other targets' configs, or files outside the declared editable set, regardless of what the LLM outputs.
- Post-write enforcement is compatible with Claude Code subprocess mode, which writes directly to the filesystem without an interceptable tool layer.
- The scope hash guard prevents gradual drift of `scope.yaml` between registration and runtime.
- The enforcement boundary means unauthorized writes do happen on disk transiently; they are cleaned up before eval executes. A crash between agent exit and scope enforcement could leave the worktree in a dirty state, which is recovered by `git reset --hard` at the next run start.
- In-place targets (where the worktree is the main checkout) relax the `metrics.yaml` immutability requirement because worktree-level scope enforcement does not apply. Users operating in in-place mode take on the responsibility of ensuring their eval configuration is not editable by the agent.
