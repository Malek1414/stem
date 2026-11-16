#!/bin/bash
# Reference ceiling: Claude Opus 5.5 (effort high) through the same harness, replayed tools and judge as the local models.
# Not a candidate (abliterated models only); runs in parallel with queue10 since it only shares the judge.
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
run .venv/bin/python -m forge bench-c data/snapshots/v0002 --model anthropic:claude-opus-5-5 --replay-tools
echo "=== queue11 done $(date '+%F %T')"
