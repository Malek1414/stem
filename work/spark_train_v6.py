# Run on the Spark kernel by chain_d4.sh (work/spark.py py): v6 = v3's settings on the D4 data. Mirrors push_v5.js.
import subprocess, time, os
os.environ.pop('CODER27_ADAPTER', None)
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, executable='/bin/bash', timeout=90).stdout.decode()
busy = sh("pgrep -fa 'forge.train_lora|convert_lora|after_train' | grep -v pgrep")
if busy.strip():
    print('ABORT: something is already training:', busy); raise SystemExit
print(sh("cd ~/agent-forge && sed -e 's/v5/v6/g' after_train_v5.sh > after_train_v6.sh && chmod +x after_train_v6.sh; grep -c v6 after_train_v6.sh; grep -c v5 after_train_v6.sh"))
print(sh("curl -s -m 20 localhost:11434/api/generate -d '{\"model\":\"gpt-oss:120b\",\"keep_alive\":0}' | head -c 100; echo; cd ~/agent-forge && ./svc.sh stop af-coder-27t; ./svc.sh stop af-coder-27; ./svc.sh stop af-operator; sleep 8; free -g | sed -n 2p"))
av = int(sh("free -g | awk '/^Mem:/{print $7}'").strip())
if av < 100:
    print('ABORT: only', av, 'GB available, need > 100 before training'); raise SystemExit
subprocess.Popen("cd ~/agent-forge && DATA=data/sft/v0002-distill/coder.jsonl exec setsid ./train.sh coder27 v6 0 --max-len 16384 --accum 2 --warmup 3 --eval-frac 0.1 --save-every 5 --epochs 2 > train_coder27_v6.log 2>&1 < /dev/null", shell=True, executable='/bin/bash', stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
time.sleep(4)
subprocess.Popen("cd ~/agent-forge && exec setsid ./train_guard.sh > run/train_guard.log 2>&1 < /dev/null", shell=True, executable='/bin/bash', stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
time.sleep(2)
subprocess.Popen("cd ~/agent-forge && sleep 120; exec setsid ./after_train_v6.sh > run/after_train_v6.log 2>&1 < /dev/null", shell=True, executable='/bin/bash', stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
time.sleep(3)
print('STARTED'); print(sh("ps -eo pid,args | grep -E '[t]rain_guard|[f]orge.train_lora|[a]fter_train_v6' | cut -c1-140; head -3 ~/agent-forge/train_coder27_v6.log"))
