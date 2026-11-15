#!/bin/bash
# systemd --user died on the Spark (no linger, no sudo), so run the unit commands as plain background processes.
# usage: svc.sh start|stop NAME... | svc.sh status        names: link af-operator af-coder af-coder-h af-dense
cd ~/agent-forge && mkdir -p run logs
M=$HOME/agent-forge/models; B=$HOME/agent-forge/llama.cpp/build/bin/llama-server
COMMON="-c 65536 -ngl 99 -np 1 --jinja --host 127.0.0.1 --seed 0"
cmd() { case $1 in
  link) echo "env PYTHONPATH=$HOME/agent-forge $HOME/agent-forge/.venv/bin/python -m link run --name spark --bind 0.0.0.0:47100 --mode relay";;
  af-operator) echo "$B -m $M/holo4-heretic/holo4-35b-a3b-heretic-Q4_K_M.gguf --mmproj $M/holo4-heretic/mmproj.f16.gguf --alias holo4-heretic --port 8081 $COMMON";;
  af-coder) echo "$B -m $M/qwen36-ablit/Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf --mmproj $M/qwen36-ablit/Huihui-Qwen3.6-35B-A3B-abliterated.mmproj-f16.gguf --alias qwen36-ablit --port 8082 $COMMON";;
  af-coder-h) echo "$B -m $M/qwen36-heretic/Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf --mmproj $M/qwen36-heretic/Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf --alias qwen36-heretic --port 8084 $COMMON";;
  af-coder-t) A=${CODER_ADAPTER:-$(ls -t $HOME/agent-forge/adapters/coder-*/coder-*.gguf 2>/dev/null | head -1)}
     echo "$B -m $M/qwen36-heretic/Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf --mmproj $M/qwen36-heretic/Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf --lora $A --alias qwen36-heretic-tuned --port 8085 $COMMON";;
  af-operator-t) A=${OPERATOR_ADAPTER:-$(ls -t $HOME/agent-forge/adapters/operator-*/operator-*.gguf 2>/dev/null | head -1)}
     echo "$B -m $M/holo4-heretic/holo4-35b-a3b-heretic-Q4_K_M.gguf --mmproj $M/holo4-heretic/mmproj.f16.gguf --lora $A --alias holo4-heretic-tuned --port 8086 $COMMON";;
  af-dense) echo "$B -m $M/qwen38-ablit/Huihui-Qwen3.8-27B-abliterated.Q4_K_M.gguf --mmproj $M/qwen38-ablit/Huihui-Qwen3.8-27B-abliterated.mmproj-f16.gguf --alias qwen38-ablit --port 8083 $COMMON";;
  *) return 1;; esac; }
alive() { [ -f run/$1.pid ] && kill -0 "$(cat run/$1.pid)" 2>/dev/null; }
case $1 in
  start) for n in "${@:2}"; do c=$(cmd $n) || { echo "unknown $n"; continue; }
           if alive $n; then echo "$n already running $(cat run/$n.pid)"; continue; fi
           setsid nohup $c > logs/$n.log 2>&1 < /dev/null & echo $! > run/$n.pid; echo "$n started $!"; done;;
  stop) for n in "${@:2}"; do alive $n && kill "$(cat run/$n.pid)"; rm -f run/$n.pid; echo "$n stopped"; done;;
  status) for n in link af-operator af-coder af-coder-h af-coder-t af-operator-t af-dense; do alive $n && echo "$n running $(cat run/$n.pid)" || echo "$n stopped"; done;;
esac
