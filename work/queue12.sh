#!/bin/bash
# A1: agent prompt v2 (find the subject first, check every item) on the 35B coder, same benchmark C with replay.
# Start only after `svc.sh start af-coder-h` (Heretic Q4, port 8084) is healthy and nothing else heavy runs.
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
run .venv/bin/python -m forge bench-c data/snapshots/v0002 --model llamacpp:8084@qwen36-heretic --replay-tools --agent v2
echo "=== queue12 done $(date '+%F %T')"
