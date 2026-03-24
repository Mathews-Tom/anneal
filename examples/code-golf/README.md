# Example: Code Golf

Minimize Python file size while preserving exact output.

## What it does

Anneal mutates `app.py` to reduce its character count (`wc -c`). The eval script runs the file, compares output byte-for-byte against `expected_output.txt`, and returns a penalty score (99999) if the output changes. The agent must find shorter ways to produce identical output.

**Eval mode:** Deterministic — `wc -c` on the source file (lower is better).

## Run it

```bash
anneal init
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

## What to expect

- ~$0.20–0.40 per experiment
- ~1–2 minutes per experiment
- Starting size: 3,592 characters (deliberately verbose Python)
- The agent replaces verbose loops with builtins, inlines functions, collapses output
- In a test run: **3,592 → 228 characters (93.7% reduction)** in 7 kept experiments

## Files

| File | Purpose |
|------|---------|
| `app.py` | The artifact being optimized (editable) |
| `eval.sh` | Runs app.py, verifies output, returns char count (immutable) |
| `expected_output.txt` | Reference output for correctness check (immutable) |
| `scope.yaml` | Declares what the agent can and cannot modify |
