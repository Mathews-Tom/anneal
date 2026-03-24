# Example: Test Coverage

Improve pytest test coverage of a Python module.

## What it does

Anneal mutates `tests/test_calculator.py` to add tests for untested code paths in `src/calculator.py`. The eval script runs `pytest --cov` and extracts the coverage percentage. Only mutations that increase coverage are kept.

**Eval mode:** Deterministic — `pytest --cov` outputs a coverage percentage.

## Run it

```bash
anneal init
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

## What to expect

- ~$0.10–0.20 per experiment (agent mutation only, no judge calls)
- ~30–60 seconds per experiment
- Starting coverage: ~45% (3 basic tests, no edge cases)
- The agent adds tests for: error paths, boundary conditions, untested functions (`fibonacci`, `clamp`, `mean`, `median`)
- Coverage should reach 90%+ within 5–8 experiments

## Files

| File | Purpose |
|------|---------|
| `tests/test_calculator.py` | Test file being optimized (editable) |
| `src/calculator.py` | Module under test (immutable) |
| `eval.sh` | Runs pytest, extracts coverage % (immutable) |
| `scope.yaml` | Declares what the agent can and cannot modify |
