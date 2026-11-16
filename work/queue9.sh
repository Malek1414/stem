#!/bin/bash
# T3: evaluate the tuned coder (Heretic Q4 + adapters/coder-v1 LoRA, af-coder-t port 8085). Start only after
# `svc.sh start af-coder-t` is healthy on the Spark (one model + judge at a time).
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
M=llamacpp:8085@qwen36-heretic-tuned
run .venv/bin/python -m forge bench-c data/snapshots/v0002 --model $M --replay-tools
run .venv/bin/python -m forge bench-c data/snapshots/v0002 --model $M
run .venv/bin/python -m forge bench-b data/snapshots/v0002 --model $M
echo "=== queue9 done $(date '+%F %T')"
