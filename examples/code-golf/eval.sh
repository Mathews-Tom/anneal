#!/bin/bash
# Eval script for code-golf: outputs character count only if output is correct.
# If output differs from expected, outputs a high penalty score (99999).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPECTED="$SCRIPT_DIR/expected_output.txt"

# Run the script and capture output
ACTUAL=$(python "$SCRIPT_DIR/app.py" 2>&1) || { echo "99999"; exit 0; }

# Compare output (strip trailing whitespace)
if diff <(echo "$ACTUAL" | sed 's/[[:space:]]*$//') <(cat "$EXPECTED" | sed 's/[[:space:]]*$//') > /dev/null 2>&1; then
    # Output matches — report character count
    wc -c < "$SCRIPT_DIR/app.py" | tr -d ' '
else
    # Output differs — penalty
    echo "99999"
fi
