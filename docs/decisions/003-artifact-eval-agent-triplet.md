# ADR-003: Artifact / Eval / Agent Triplet as Core Abstraction

## Status
Accepted

## Context

The system needs to be domain-agnostic: it must optimize a Python file, a system prompt, a configuration file, or any other text artifact without changes to the engine. The engine cannot know in advance what "better" means, what commands to run, or what mutation strategy is appropriate for a given domain.

Three design options were considered:

- **Monolithic target config**: A single configuration structure where the artifact, scoring logic, and mutation instructions are mixed together. Simple to start, but makes it hard to swap evaluation strategies or reuse mutation programs across artifacts.
- **Plugin system**: The user implements a Python interface for each component. Powerful but requires code for every new target; raises install and security concerns since user code runs inside the engine process.
- **Declarative triplet with shell-command eval**: The user declares the artifact (what to optimize), the eval (a shell command that produces a score), and the agent instructions (a prompt file). All three are plain files. The engine is the only code that runs.

## Decision

Every optimization target is registered as a triplet:

1. **Artifact** (`artifact_paths`): One or more files the agent is permitted to mutate. Specified in `scope.yaml` under the `editable` list. The registry (`anneal/engine/registry.py`) validates that at least one path is declared.

2. **Eval** (`eval_config`): Either a deterministic shell command (`run_command` + `parse_command`) or a stochastic LLM-judged evaluation (`StochasticEval` with `criteria` and `test_prompts`). The `EvalEngine` (`anneal/engine/eval.py`) dispatches based on `eval_mode`. The eval definition is immutable once registered: `scope.yaml`, `metrics.yaml`, and `eval_criteria.toml` are always in `scope.immutable`, preventing the agent from influencing its own scoring.

3. **Agent** (`agent_config` + `program.md`): An `AgentConfig` specifying the model, budget, and invocation mode (`claude_code` or `api`), paired with a `program.md` mutation instruction file. `AgentInvoker` (`anneal/engine/agent.py`) dispatches to either Claude Code subprocess mode or a direct API call based on `AgentConfig.mode`.

The triplet is the schema of `OptimizationTarget` (`anneal/engine/types.py`). It is persisted to `.anneal/config.toml` via the `Registry` class and loaded at runner startup.

## Consequences

- Any file-based artifact can be optimized without changing engine code: the user only writes a `scope.yaml`, a `metrics.yaml` (or `eval_criteria.toml` for stochastic), and a `program.md`.
- The eval is fully external to the engine: it is a shell command, not Python code. This prevents the agent from gaming the evaluator and keeps the engine free of domain-specific logic.
- The immutability constraint on eval config files is enforced by scope enforcement at every experiment, not just at registration. This closes the attack surface where an agent could modify its own scoring criteria.
- The agent operates only on the `editable` paths declared in `scope.yaml`. It has no knowledge of the eval command and cannot read `metrics.yaml` or `eval_criteria.toml` (they are not injected into the agent context).
- The `program.md` abstraction means mutation strategy can be updated between runs without re-registering the target, by editing the program file.
- The triplet couples the artifact, eval, and agent to a single `OptimizationTarget` ID. Multi-artifact joint optimization is supported by listing multiple paths in `artifact_paths`, but each target has a single eval signal.
