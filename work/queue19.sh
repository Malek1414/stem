#!/bin/bash
# N4: the 27B + LoRA coder27-v2 (67 checked Claude attempts, 19 with thinking; empty think blocks not trained), af-coder-27t port 8088, ngram-mod.
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
run .venv/bin/python -m forge bench-c data/snapshots/v0002 --model llamacpp:8088@qwen38-heretic-tuned-v2 --replay-tools
echo "=== queue19 done $(date '+%F %T')"
