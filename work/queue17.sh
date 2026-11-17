#!/bin/bash
# N4: the 27B + LoRA coder27-v1 (39 checked Claude attempts through this harness), af-coder-27t port 8088, ngram-mod.
cd "$(dirname "$0")/.."
run() { echo "=== $(date '+%F %T') $*"; "$@" || echo "!!! failed: $*"; }
run .venv/bin/python -m forge bench-c data/snapshots/v0002 --model llamacpp:8088@qwen38-heretic-tuned --replay-tools
echo "=== queue17 done $(date '+%F %T')"
