#!/bin/bash
# after the latent sweep: hard-task pilot (one heavy job at a time), then the operator + coder servers
cd ~/agent-forge
while pgrep -f run_sweep.sh >/dev/null; do sleep 60; done
echo "=== hard pilot $(date '+%F %T')"
PYTHONPATH=$HOME/agent-forge HF_HUB_OFFLINE=1 timeout 45m ~/vllm-nano/.venv/bin/python -m latent bench --model Qwen/Qwen3-8B \
  --n 40 --level hard --conditions none,text_full,text_card,kv_full,kv_q8,kv_goal,latentmas
echo "=== hard pilot exit $? $(date '+%F %T')"
PYTHONPATH=$HOME/agent-forge .venv/bin/python -m link send malek "hard-done: ~/agent-forge/results/latent/*_hard.json"
./svc.sh start af-operator af-coder
for i in $(seq 1 60); do
  curl -s localhost:8081/health | grep -q ok && curl -s localhost:8082/health | grep -q ok && break; sleep 10; done
echo "=== health $(date '+%F %T'): $(curl -s localhost:8081/health) $(curl -s localhost:8082/health)"
PYTHONPATH=$HOME/agent-forge .venv/bin/python -m link send malek "servers-up: operator $(curl -s localhost:8081/health) coder $(curl -s localhost:8082/health)"
