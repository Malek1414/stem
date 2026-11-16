#!/bin/bash
# A2 retry: agent v4 (v2 + a self-check that quotes the current request and keeps the final message the answer), 35B coder.
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
run .venv/bin/python -m forge bench-c data/snapshots/v0002 --model llamacpp:8084@qwen36-heretic --replay-tools --agent v4
echo "=== queue13 done $(date '+%F %T')"
