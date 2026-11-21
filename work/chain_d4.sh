#!/bin/bash
# D4 chain (Malek 22.11 00:55 "set up the chain and the bigger benchmark"), runs unattended on the Mac:
#  1 wait for queue26 (distill on the Layla key, $40 cap)   2 --build   3 push coder.jsonl to the Spark (sha256)
#  4 train v6 (v3 settings) on the Spark                    5 wait for "v6-ready" (local link inbox only)
#  6 bench C --all-tasks (39 held-out turns): v6, v3, v5 on :8088 (adapter swapped and checked), untuned on :8087
#  7 serve the best adapter on af-coder-27t, restart af-operator if >= 30 GB.  Spark access only via work/spark.py.
cd "$(dirname "$0")/.."
NV="/Applications/NVIDIA Sync.app/Contents/Resources/bin/nvsync-arm64"
PY=.venv/bin/python; S="$PY work/spark.py"
say() { echo "=== $(date '+%F %T') $*"; }
bench() { caffeinate -i $PY -m forge bench-c data/snapshots/v0002 --model "$1" --replay-tools --all-tasks | tail -1; }
serve() {  # serve adapter $1 (v3, v5, v6) on :8088 and check the loaded lora
  $S sh "cd ~/agent-forge && ./svc.sh stop af-coder-27t; export CODER27_ADAPTER=\$HOME/agent-forge/adapters/coder27-$1/coder27-$1.gguf; ./svc.sh start af-coder-27t; for i in \$(seq 1 60); do curl -s -m 3 localhost:8088/health | grep -q ok && break; sleep 5; done; curl -s localhost:8088/health; curl -s localhost:8088/lora-adapters | head -c 200" 420
}
say "chain_d4 start"
q26done() { awk '/queue26 start/{d=0} /queue26 done/{d=1} END{exit !d}' work/queue26.log; }
until q26done; do sleep 120; done
say "queue26 done: $(grep -o '"stopped".*' work/queue26.log | tail -1)"
before=$(wc -l < data/sft/v0002-distill/coder.jsonl)
$PY -m forge.distill data/snapshots/v0002 --build | tail -12
after=$(wc -l < data/sft/v0002-distill/coder.jsonl)
say "build: $before -> $after examples"
"$NV" open malekhassan 11000 >/dev/null 2>&1; "$NV" open malekhassan 11012 >/dev/null 2>&1; sleep 3
if [ "$after" -gt "$before" ]; then
  say "push: $($S put data/sft/v0002-distill/coder.jsonl agent-forge/data/sft/v0002-distill/coder.jsonl)"
  inbox_after=$($PY -m link inbox | python3 -c 'import sys,json;d=json.load(sys.stdin);print(max([e.get("seq",0) for e in d] or [0]))')
  out=$($S py "$(cat work/spark_train_v6.py)" 200); echo "$out"
  if ! echo "$out" | grep -q STARTED; then say "!!! v6 training did not start"; V6=0; else
    say "waiting for v6-ready (inbox after $inbox_after)"
    until $PY -m link inbox --after "$inbox_after" | grep -q '"v6-ready'; do sleep 120; done
    msg=$($PY -m link inbox --after "$inbox_after" | grep -o 'v6-ready[^"]*' | head -1); say "$msg"
    echo "$msg" | grep -q FAILED && V6=0 || V6=1
  fi
else
  say "no new examples: skipping v6, running the bigger benchmark only"; V6=0
fi
"$NV" open malekhassan 8088; "$NV" open malekhassan 8087; "$NV" open malekhassan 11434; sleep 3
for v in $([ "$V6" = 1 ] && echo v6) v3 v5; do
  say "serve $v: $(serve $v | tr '\n' ' ')"
  say "bench $v: $(bench llamacpp:8088@qwen38-heretic-tuned-$v)"
done
say "untuned: $($S sh 'cd ~/agent-forge && ./svc.sh stop af-coder-27t; ./svc.sh start af-coder-27; for i in $(seq 1 60); do curl -s -m 3 localhost:8087/health | grep -q ok && break; sleep 5; done; curl -s localhost:8087/health' 420 | tr '\n' ' ')"
say "bench untuned: $(bench llamacpp:8087@qwen38-heretic)"
$S sh 'cd ~/agent-forge && ./svc.sh stop af-coder-27' 60 >/dev/null
best=$($PY - <<'PYEOF'
import json, pathlib
r = {}
for v in ("v3", "v5", "v6"):
    p = pathlib.Path(f"results/qwen38-heretic-tuned-{v}/v0002/bench_c_replay_all.json")
    if p.exists():
        o = json.loads(p.read_text())["overall"]; r[v] = o["overall"] if o["graded"] >= 35 else None
p = pathlib.Path("results/qwen38-heretic/v0002/bench_c_replay_all.json")
base = json.loads(p.read_text())["overall"]["overall"] if p.exists() else None
ok = {k: x for k, x in r.items() if x is not None}
best = max(ok, key=lambda k: (ok[k], k == "v3")) if ok else "v3"  # a tie keeps the incumbent v3
json.dump({"scores": r, "untuned": base, "best": best}, open("work/chain_d4.result.json", "w"), indent=1)
print(best)
PYEOF
)
say "scores: $(cat work/chain_d4.result.json | tr -d '\n ')"
say "serve best $best: $(serve $best | tr '\n' ' ')"
say "operator: $($S sh 'cd ~/agent-forge && AV=$(free -g | awk "/^Mem:/{print \$7}"); echo avail $AV; if [ $AV -ge 30 ]; then ./svc.sh start af-operator; fi' 120 | tr '\n' ' ')"
say "chain_d4 done"
