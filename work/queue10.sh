#!/bin/bash
# 27B base: Qwen3.8-27B Heretic Q4_K_M (dense, ~9x the active parameters of the 35B-A3B), af-coder-27 port 8087.
# Start only after `svc.sh stop af-coder-t; svc.sh start af-coder-27` is healthy on the Spark.
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
M=llamacpp:8087@qwen38-heretic
run .venv/bin/python -m forge bench-c data/snapshots/v0002 --model $M --replay-tools
run .venv/bin/python -m forge bench-c data/snapshots/v0002 --model $M
echo "=== queue10 done $(date '+%F %T')"
