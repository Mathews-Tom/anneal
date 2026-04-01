# ADR-001: Git Worktrees for Experiment Isolation

## Status
Accepted

## Context

Each experiment mutates one or more artifact files and then evaluates the result. The system needs a way to isolate mutations so that:

- A failed or crashed experiment does not corrupt the working state of the repository.
- Multiple experiments can be staged, rolled back, and compared using standard git tooling.
- The agent operates on a real filesystem path (required by Claude Code subprocess mode, which writes files directly to disk).
- The original checked-out repository remains usable during optimization runs.

Alternatives considered:

- **Temporary directories with file copies**: Requires manual tracking of which files to copy, no diff history, no rollback guarantee, incompatible with eval commands that expect a real git repository context.
- **Branches with stash/checkout**: `git stash` is unreliable under concurrent access and loses untracked files. Checking out branches on the main worktree blocks the user from using the repo during a run.
- **Containers or VMs**: Heavyweight, require Docker or similar runtimes, increase setup friction, and do not solve the rollback problem at the file level.
- **In-process file patching**: Cannot support shell-command-based evals that execute tools like `pytest` or `wc` against files on disk.

## Decision

Each optimization target gets a dedicated git worktree created at `.anneal/worktrees/<target-id>` on a branch named `anneal/<target-id>`. The `GitEnvironment` class (`anneal/engine/environment.py`) manages all worktree lifecycle operations.

The experiment cycle within the worktree is:

1. Record the HEAD SHA before the agent runs (`pre_sha`).
2. Invoke the agent, which writes mutations directly to the worktree path.
3. Run scope enforcement to validate which files were changed.
4. Run the eval command against the worktree.
5. On KEPT: commit the mutation as a permanent record on the `anneal/<target-id>` branch.
6. On DISCARDED/BLOCKED: `git reset --hard` to `pre_sha`, returning the worktree to baseline.

The system never uses `git stash`. The only snapshot and restore mechanism is commit and `reset --hard`.

## Consequences

- Every kept mutation is a real git commit, making the full optimization history inspectable with `git log` on the `anneal/<target-id>` branch.
- Rollback to any prior state is a standard `git reset --hard <sha>` operation.
- The main worktree is untouched during optimization; the user can continue working in it.
- Worktree creation requires that the branch name `anneal/<target-id>` is not already in use for a different worktree. Stale worktree references are pruned at creation time via `git worktree prune`.
- In-place targets (where the worktree is the main checkout) bypass worktree-level scope enforcement; `metrics.yaml` immutability is relaxed accordingly.
- The `.anneal/worktrees/` directory accumulates disk usage proportional to the number of registered targets and the size of the repository.
