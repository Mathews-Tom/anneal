from __future__ import annotations

import re
import sys

PENALTY = 99999.0
LINE_PATTERN = re.compile(r"^execution_time_ms:\s*([0-9]+(?:\.[0-9]*)?)$")


def main() -> None:
    stdin_text = sys.stdin.read()
    for line in stdin_text.splitlines():
        match = LINE_PATTERN.match(line.strip())
        if match:
            print(match.group(1))
            return
    print(PENALTY)


if __name__ == "__main__":
    main()
