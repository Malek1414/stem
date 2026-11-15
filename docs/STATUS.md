# agent-forge status (read by the hourly watcher)

Updated: 2026-11-15. The watcher does the first unchecked item that does not need Malek's approval,
ticks it here, commits, and stops. Items marked (APPROVAL) wait for Malek; never start them unasked.

## Rules for the watcher

- DGX Spark: only via the existing NVIDIA Sync tunnels (Ollama on localhost:11434). Never plain ssh.
  One Spark job at a time; check nothing else is running on it first (`pgrep -fl "python.* -m forge bench"`;
  plain `ps aux | grep` matches its own shell and gives a false positive).
- Skip items marked (APPROVAL) or (IN PROGRESS).
- Never use Haiku for anything.
- Do not send email, Slack or anything outward. Do not delete user files.
- Commit each finished step with the attribution lines from the session.

## Queue

- [x] Read results/qwen3-coder-next-abliterated-128k/v0002/bench_b.json (the "one phone" rerun after the
      tool-name fix). Record score, checks, steps, seconds and whether the model now used Bash/Read/Edit
      in docs/RESULTS.md. If it failed for a harness reason, fix the harness, add a test, rerun once.
- [x] Read results/published.jsonl. Confirm v0002 is on the private repo malekhassan23/agent-forge-data
      with zero mismatches; note the revision in docs/RESULTS.md. If the upload failed, rerun
      `uv run --no-project --with huggingface_hub python -m forge publish data/snapshots/v0002` once.
- [x] Link spike 1 (no install on the Spark, scratch folder only): in a throwaway uv env, check that PyPI
      `iroh` (Python bindings) and `a2a-sdk` install on the Mac and can open one iroh connection between
      two local endpoints and carry one A2A message. Write findings to docs/research/2026-11-15-link-spikes.md.
- [x] Link spike 2: read code.claude.com/docs/en/cross-session-messaging and the Claude Code Channels docs;
      note whether inbound push into a session works without a developer flag. Same findings file.
- [x] Link spike 3: measure Mac -> Spark round-trip latency over the existing Ollama tunnel
      (20 tiny `/api/version` calls, median and p95). Same findings file.
- [x] Build the link per the design; approved by Malek ("keep going",
      2026-11-15). Spec: docs/superpowers/specs/2026-11-15-link-design.md.
- [ ] (APPROVAL) Decisions from docs/research/2026-11-15-training-and-eval-evidence.md: rotating
      held-out project; operator base from an open computer-use model; human calibration of judge/B rule.

When every non-APPROVAL item is ticked, the watcher deletes its own cron job and leaves a summary.


## Overnight plan 2026-11-15 (Malek asleep; "do whatever is needed to keep running through the night")

Rules on top of the ones above: abliterated models only (Malek, 2026-11-15). Spark access only through Sync:
JupyterLab at http://localhost:11012/lab in the chrome-real browser tab (load the kernel helper with
mcp__chrome-real__evaluate_script sourcePath=/Users/malekhassan/Desktop/Projects/agent-forge/work/jhelper.js,
then call window.__run(python_code, timeout_ms)). If the tab lost the page, navigate it to that URL first.
llama-servers on the Spark: systemd --user died at ~21:30 (Linger=no, no sudo), so every service now runs through
~/agent-forge/svc.sh (start|stop NAME..., status; pid files in run/, logs in logs/). Names: af-operator (8081, Holo4-35B-A3B-heretic), af-coder
(8082, Huihui-Qwen3.6-35B-A3B-abliterated), af-dense (8083, Huihui-Qwen3.8-27B-abliterated). Mac reaches them
through Sync tunnels (`nvsync open malekhassan <port>`; forge reopens forgotten ports itself). Memory budget:
121 GB; operator+coder ~46 GB, Ollama judge gpt-oss:120b 65 GB, old model 52 GB: never all at once.

Check each hour, in order, and do the first thing that applies:
- [ ] A0. Hard latent pilot (results/latent/Qwen_Qwen3-8B_n40_s20261115_hard.json): add to RESULTS.md next to the
      sweep. Productive = text_full well off ceiling and off the floor, and conditions differ beyond the CI.
- [ ] A. Spark latent sweep (~/agent-forge/run_sweep.sh, log sweep.log, results ~/agent-forge/results/latent/*ctx*):
      when finished, pull the summaries (accuracy, CI, bytes, co-located seconds per condition per ctx length,
      incl. kv_goal) and add a table + reading to docs/RESULTS.md; commit.
- [ ] B. Launcher: ~/agent-forge/chain.sh (log chain.log) replaced launch_servers.sh at 21:39. It waits for the
      sweep, runs the hard-task pilot (<=45 min, link message hard-done), then `svc.sh start af-operator af-coder`.
      Old notes: if the llama.cpp build failed (llamacpp_build.log), fix and rebuild
      (PATH=/usr/local/cuda/bin, CUDACXX, -DCMAKE_CUDA_ARCHITECTURES=121, -j 4), then rerun launch_servers.sh.
      If af-operator/af-coder fail to start (~/agent-forge/logs/af-coder.log), fix flags (e.g. drop --mmproj, lower -c).
- [ ] C. Mac queue work/queue3.sh (log work/queue3.log): if not running and not "queue3 done", restart it with
      `nohup caffeinate -i work/queue3.sh > work/queue3.log 2>&1 &` (benches resume from saved results and rerun
      infra_error rows). If a job ended with only infra errors, find out why (curl sync://8082/v1/chat/completions
      by hand), fix, rerun that job.
- [ ] D. When queue3 is done: write results per model to docs/RESULTS.md (B mean score + CI + infra errors, C overall
      + null control, D accuracy + seconds), compare with the baseline, commit.
- [ ] E. Then, on the Spark: `~/agent-forge/svc.sh stop af-operator af-coder`; on the Mac run harness-v2 baseline reruns
      via Ollama: `python -m forge bench-c data/snapshots/v0002 --model qwen3-coder-next-abliterated-128k` then bench-b
      (same model). Record in RESULTS.md, commit.
- [ ] F. Then: `~/agent-forge/svc.sh start af-dense`, `nvsync open malekhassan 8083`, run bench-c and bench-b with
      --model llamacpp:8083@qwen38-ablit; then `svc.sh stop af-dense`; record; commit.
- [ ] G. Then `svc.sh start af-operator af-coder` and leave a short summary at the top of docs/RESULTS.md for Malek
      (what is best per role, by how much, what failed). Delete the watcher cron job.
