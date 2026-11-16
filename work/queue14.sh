#!/bin/bash
# A1 on the best model: agent v2 on the 27B (af-coder-27, port 8087), bench C with replay.
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
run .venv/bin/python -m forge bench-c data/snapshots/v0002 --model llamacpp:8087@qwen38-heretic --replay-tools --agent v2
echo "=== queue14 done $(date '+%F %T')"
