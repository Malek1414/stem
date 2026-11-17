#!/bin/bash
# A3 round 2 data through the API (Malek, 17.11.: organisation credits instead of claude -p): summarised thinking kept,
# token usage logged per row. Skips turns already done through claude -p.
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
run .venv/bin/python -m forge.distill data/snapshots/v0002 --n 400 --seed 2 --agent v2 --tag api1 --via api
echo "=== queue18 done $(date '+%F %T')"
