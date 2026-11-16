#!/bin/bash
# A3 round 2 data (N5): second Claude worker (claude -p, Malek's plan) on 400 more train turns, own runs-w2.jsonl.
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
run .venv/bin/python -m forge.distill data/snapshots/v0002 --n 400 --seed 2 --agent v2 --tag w2
echo "=== queue16 done $(date '+%F %T')"
