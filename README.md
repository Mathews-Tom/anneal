# anneal

Let an AI agent improve your code, prompts, and configs — overnight, unattended.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/infographic.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/infographic.svg">
  <img alt="anneal infographic — autonomous optimization for any measurable artifact" src="assets/infographic.png" width="100%">
</picture>

<video src="https://github.com/user-attachments/assets/06f8848d-a404-4b58-817a-8c0bf8bb71e3" width="100%" autoplay loop muted playsinline></video>

Point anneal at any text file in a git repo, tell it how to measure "better," and walk away. The agent generates hypotheses, runs experiments, keeps winners, discards losers, and compounds learnings — all while you sleep.

## How It Works

1. **Register** a target — the file to improve, how to score it, what's off-limits
2. **Run** the loop — the agent mutates → evaluates → keeps or reverts → learns → repeats
3. **Review** — every experiment is a git commit; check the history, scores, and cost

```bash
anneal register \
  --name my-target \
  --artifact path/to/file.py \
  --eval-mode deterministic \
  --run-cmd "python benchmark.py" \
  --parse-cmd "grep 'score' | awk '{print \$2}'" \
  --direction maximize \
  --scope scope.yaml

anneal run --target my-target --experiments 20
```

## Install

```bash
uv tool install anneal-cli

# Or with pip
pip install anneal-cli
```

Requires Python 3.12+.

## Examples

### [Prompt Optimization](examples/prompt-optimizer/) — stochastic eval

Improve an article summarizer prompt. The agent rewrites `system_prompt.md`, generates summaries from 5 test articles, and an LLM judge scores each against 4 binary criteria (key points captured? concise? plain language? factually accurate?).

```bash
anneal register \
  --name prompt-optimizer \
  --artifact examples/prompt-optimizer/system_prompt.md \
  --eval-mode stochastic \
  --criteria examples/prompt-optimizer/eval_criteria.toml \
  --direction maximize \
  --scope examples/prompt-optimizer/scope.yaml

anneal run --target prompt-optimizer --experiments 10
```

### [Test Coverage](examples/test-coverage/) — deterministic eval, maximize

Improve pytest test coverage of a Python module. The agent adds tests to cover untested code paths. `pytest --cov` provides the score.

```bash
anneal register \
  --name test-coverage \
  --artifact examples/test-coverage/tests/test_calculator.py \
  --eval-mode deterministic \
  --run-cmd "bash examples/test-coverage/eval.sh" \
  --parse-cmd "cat" \
  --direction maximize \
  --scope examples/test-coverage/scope.yaml

anneal run --target test-coverage --experiments 10
```

### [Code Golf](examples/code-golf/) — deterministic eval, minimize

Shrink a verbose Python file while preserving byte-identical output. In a test run: **3,592 → 228 characters (93.7% reduction)** in 7 experiments.

```bash
anneal register \
  --name code-golf \
  --artifact examples/code-golf/app.py \
  --eval-mode deterministic \
  --run-cmd "bash examples/code-golf/eval.sh" \
  --parse-cmd "cat" \
  --direction minimize \
  --scope examples/code-golf/scope.yaml

anneal run --target code-golf --experiments 10
```

## Two Eval Modes

**Deterministic** — a shell command produces a number. Run code, parse output, compare. Use for: performance benchmarks, test coverage, file size, build time.

**Stochastic** — an LLM judges N samples against K binary (YES/NO) criteria. Use for: prompt quality, documentation clarity, content optimization — anything where output varies between runs.

## Documentation

| Doc                                    | What's in it                                             |
| -------------------------------------- | -------------------------------------------------------- |
| [Overview](docs/overview.md)           | Motivation, lineage, and the core idea                   |
| [Eval Guide](docs/eval-guide.md)       | Writing good binary evaluation criteria                  |
| [Recipes](docs/recipes.md)             | Copy-paste registration commands for common targets      |
| [Use Cases](docs/use-cases.md)         | Where anneal works, where it doesn't, and why            |
| [Features](docs/features.md)           | Search strategies, statistical methods, knowledge system |
| [Architecture](docs/architecture.md)   | Module map and design principles                         |
| [System Design](docs/system-design.md) | Full technical design document                           |

## Testing

```bash
uv run pytest tests/ -x -q          # 492 tests
uv run pytest tests/ --cov=anneal    # With coverage
```

## License

MIT
