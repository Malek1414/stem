#!/bin/bash
# D4: more distilled data on the Layla hackathon key (Malek 22.11 00:50: approved with the organizer). Cap $40 for this
# worker (runs-layla1.jsonl), continues the untried turns of tasks_n400_seed2. Key read from wohin-clip/.dev.vars, never printed.
cd "$(dirname "$0")/.."
export ANTHROPIC_API_KEY=$(grep ^ANTHROPIC_API_KEY= ~/Desktop/Projects/wohin-clip/.dev.vars | cut -d= -f2)
echo "=== $(date '+%F %T') queue26 start"
caffeinate -i .venv/bin/python -m forge.distill data/snapshots/v0002 --n 400 --seed 2 --agent v2 --tag layla1 --via api --budget-usd 40
echo "=== $(date '+%F %T') queue26 done"
