# Recipes

Copy-paste registration commands for common optimization archetypes.
Each recipe includes scope, eval command, and registration.

## API Backend (Deterministic, Minimize Latency)

Optimize a FastAPI/Flask handler for response time using `wrk` benchmarks.

```yaml
# scope.yaml
editable:
  - src/api/handlers.py
  - src/api/queries.py
immutable:
  - scope.yaml
  - tests/**
  - requirements.txt
```

```bash
anneal register \
  --name api-perf \
  --artifact src/api/handlers.py src/api/queries.py \
  --eval-mode deterministic \
  --run-cmd "wrk -t4 -c100 -d10s http://localhost:8000/api/endpoint | tail -1" \
  --parse-cmd "awk '{print \$2}'" \
  --direction minimize \
  --scope scope.yaml \
  --budget-cap-daily 5.00

anneal run --target api-perf --experiments 20
```

## Data Pipeline (Deterministic, Maximize Accuracy)

Optimize SQL models for correctness using dbt test + DuckDB.

```yaml
# scope.yaml
editable:
  - models/staging/*.sql
immutable:
  - scope.yaml
  - tests/**
  - seeds/**
```

```bash
anneal register \
  --name dbt-accuracy \
  --artifact models/staging/stg_orders.sql \
  --eval-mode deterministic \
  --run-cmd "dbt test --target dev && dbt run --target dev" \
  --parse-cmd "grep 'rows_matched' | awk '{print \$2}'" \
  --direction maximize \
  --scope scope.yaml

anneal run --target dbt-accuracy --experiments 15
```

## Config Tuning (Deterministic, Maximize Throughput)

Optimize Nginx or Redis config for requests/sec using `ab`.

```yaml
# scope.yaml
editable:
  - nginx.conf
immutable:
  - scope.yaml
  - docker-compose.yml
```

```bash
anneal register \
  --name nginx-throughput \
  --artifact nginx.conf \
  --eval-mode deterministic \
  --run-cmd "docker compose restart nginx && sleep 2 && ab -n 10000 -c 100 http://localhost/" \
  --parse-cmd "grep 'Requests per second' | awk '{print \$4}'" \
  --direction maximize \
  --scope scope.yaml

anneal run --target nginx-throughput --experiments 10
```

## Test Coverage (Deterministic, Maximize Coverage %)

Generate or improve test cases to increase pytest coverage.

```yaml
# scope.yaml
editable:
  - tests/**/*.py
immutable:
  - scope.yaml
  - src/**
```

```bash
anneal register \
  --name test-coverage \
  --artifact tests/ \
  --eval-mode deterministic \
  --run-cmd "pytest --cov=src --cov-report=term-missing -q" \
  --parse-cmd "grep TOTAL | awk '{print \$4}' | tr -d '%'" \
  --direction maximize \
  --scope scope.yaml

anneal run --target test-coverage --experiments 20
```

## Remote/Cloud Eval (with Environment Config)

For targets that require cloud services, declare environment requirements:

```toml
# In config.toml after registration, add:
[targets.api-perf.eval_environment]
requires_network = true
env_vars = ["DATABASE_URL", "API_STAGING_URL"]
setup_command = "docker compose -f docker-compose.staging.yml up -d"
teardown_command = "docker compose -f docker-compose.staging.yml down"
```

For flaky network eval, enable retry and flake detection:

```toml
[targets.api-perf.eval_config.deterministic]
run_command = "curl -s http://staging/api/health"
parse_command = "jq '.latency_ms'"
timeout_seconds = 30
max_retries = 3
retry_delay_seconds = 5.0
flake_detection = true
```

## Advanced: Multi-Draft with Verification Gates

Use multiple drafts with structural verification for high-reliability optimization.

```yaml
# scope.yaml
editable:
  - src/core/engine.py
immutable:
  - scope.yaml
  - tests/**
watch:
  - src/core/types.py
```

```bash
anneal register \
  --name engine-perf \
  --artifact src/core/engine.py \
  --eval-mode deterministic \
  --run-cmd "python -m pytest tests/bench.py -q --tb=no | tail -1" \
  --parse-cmd "awk '{print \$1}'" \
  --direction minimize \
  --scope scope.yaml \
  --verifier "typecheck:python -m mypy src/core/engine.py --no-error-summary" \
  --verifier "lint:ruff check src/core/engine.py" \
  --n-drafts 3

anneal run --target engine-perf --experiments 50
```

Each cycle generates 3 drafts, verifies each against mypy and ruff, and promotes the first survivor. Failed drafts are discarded before eval — no wasted eval budget.

## Advanced: UCB Tree Search with Policy Agent

Combine tree search (backtracking) with continuous instruction tuning for complex optimization landscapes.

```bash
anneal register \
  --name prompt-quality \
  --artifact prompts/system.md \
  --eval-mode stochastic \
  --criteria eval_criteria.toml \
  --scope scope.yaml \
  --policy-model gpt-4.1-mini \
  --policy-interval 5 \
  --restart-probability 0.05

anneal run --target prompt-quality --search ucb_tree --experiments 100
```

The policy agent rewrites mutation instructions every 5 experiments based on failure patterns. UCB tree search backtracks to high-scoring ancestors when recent branches degrade. Random restart (5% probability, decaying with SA temperature) introduces fresh perspectives.
