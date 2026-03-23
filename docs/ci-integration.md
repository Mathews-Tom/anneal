# CI/CD Integration

Run anneal as part of a GitHub Actions workflow to optimize code on every PR.

## Reference Workflow

```yaml
# .github/workflows/anneal-optimize.yml
name: Anneal Optimization
on:
  pull_request:
    types: [opened, synchronize]
    paths:
      - 'src/api/**'

jobs:
  optimize:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.head_ref }}
          fetch-depth: 0

      - uses: astral-sh/setup-uv@v4
      - run: uv tool install anneal-cli

      - name: Initialize anneal
        run: anneal init || true

      - name: Register target
        run: |
          anneal register \
            --name pr-${{ github.event.number }}-perf \
            --artifact src/api/handlers.py \
            --eval-mode deterministic \
            --run-cmd "pytest tests/api/ -x --tb=short -q && ./scripts/bench-api.sh" \
            --parse-cmd "grep 'p99_latency' | awk '{print \$2}'" \
            --direction minimize \
            --scope scope.yaml \
            --budget-cap-daily 5.00
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Run optimization
        run: anneal run --target pr-${{ github.event.number }}-perf --experiments 10
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Push improvements
        run: |
          cd .anneal/worktrees/pr-${{ github.event.number }}-perf
          git push origin HEAD:${{ github.head_ref }}

      - name: Comment results
        uses: actions/github-script@v7
        with:
          script: |
            const { execSync } = require('child_process');
            const raw = execSync(
              'anneal status --target pr-${{ github.event.number }}-perf --json'
            ).toString();
            const data = JSON.parse(raw);
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: ${{ github.event.number }},
              body: `## Anneal Optimization Results\n` +
                    `- Experiments: ${data.total_experiments}\n` +
                    `- Improvements kept: ${data.kept_count}\n` +
                    `- Score: ${data.baseline_score} → ${data.current_score}\n` +
                    `- Cost: $${data.total_cost_usd.toFixed(2)}`
            });
```

## `anneal status --json` Output

```json
{
  "target_id": "pr-42-perf",
  "runner_state": "HALTED",
  "total_experiments": 10,
  "kept_count": 3,
  "baseline_score": 142.5,
  "current_score": 98.3,
  "total_cost_usd": 1.87,
  "experiments": [
    {
      "id": "a1b2c3d4",
      "outcome": "KEPT",
      "hypothesis": "Replaced synchronous DB call with connection pool",
      "score": 118.2,
      "git_sha": "a1b2c3d"
    }
  ]
}
```

## Commit Trail

Each KEPT experiment is a discrete git commit with a hypothesis message.
Improvements are individually revertable and traceable:

```
a1b2c3d hypothesis: Replace synchronous DB call with connection pool
e4f5g6h hypothesis: Batch validation queries to reduce round-trips
```
