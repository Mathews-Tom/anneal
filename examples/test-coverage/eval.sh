#!/bin/bash
# Eval script for test-coverage: outputs coverage percentage as a float.
# If tests fail, outputs 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

# Run pytest with coverage, extract the TOTAL percentage
RESULT=$(python -m pytest tests/ --cov=src --cov-report=term-missing -q 2>&1) || {
    echo "0"
    exit 0
}

# Parse the TOTAL line: "TOTAL    67    20    70%"
echo "$RESULT" | grep "^TOTAL" | awk '{print $NF}' | tr -d '%'
