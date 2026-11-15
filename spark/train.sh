#!/bin/bash
# usage: train.sh <coder|operator> <version> [max_tokens] [extra train_lora args...]
# Trains a LoRA adapter on the Spark (needs the GPU to itself: stop the llama-servers and unload Ollama first),
# converts it to GGUF for llama.cpp and sends a link message when done.
set -o pipefail
cd ~/agent-forge
ROLE=$1; V=$2; TOK=${3:-3000000}; shift 3 2>/dev/null
case $ROLE in coder) BASE=hf/qwen36-heretic;; operator) BASE=hf/holo4;; *) echo "role?"; exit 1;; esac
OUT=adapters/$ROLE-$V
export PYTHONPATH=$HOME/agent-forge HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== train $ROLE $V start $(date '+%F %T') base=$BASE tokens=$TOK"
~/vllm-nano/.venv/bin/python -m forge.train_lora --model $BASE --data data/sft/v0002/$ROLE.jsonl --out $OUT \
  --max-tokens $TOK "$@"
rc=$?
echo "=== train exit $rc $(date '+%F %T')"
if [ $rc -eq 0 ]; then
  PYTHONPATH=$HOME/agent-forge/llama.cpp/gguf-py ~/vllm-nano/.venv/bin/python llama.cpp/convert_lora_to_gguf.py \
    --base $BASE --outfile $OUT/$ROLE-$V.gguf $OUT && echo "=== gguf $OUT/$ROLE-$V.gguf $(date '+%F %T')"
fi
PYTHONPATH=$HOME/agent-forge .venv/bin/python -m link send malek "train-done: $ROLE $V rc=$rc $(cat $OUT/result.json 2>/dev/null | tr -d '\n ')"
