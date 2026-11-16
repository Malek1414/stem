#!/bin/bash
# A3: Claude (claude -p, Malek's plan) on 100 train-split turns through the harness, agent v2, checked by Claude.
# No Spark load. Resumable: a usage-limit stop is picked up by rerunning this script.
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
run .venv/bin/python -m forge.distill data/snapshots/v0002 --n 100 --seed 1 --agent v2
echo "=== queue15 done $(date '+%F %T')"
