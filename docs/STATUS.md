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
- [x] L. (skipped 22:25: malekhassan is not in sudoers; only the Spark admin can enable linger. After a Spark reboot run
      `~/agent-forge/svc.sh start link af-operator af-coder`.) Linger: Malek said he would run `sudo loginctl enable-linger malekhassan` on the Spark. The autostart
      symlink of agent-forge-link.service was removed (22:10) so systemd does not start a second link daemon. Once
      `loginctl show-user malekhassan -p Linger` says yes: `~/agent-forge/svc.sh stop link && XDG_RUNTIME_DIR=/run/user/$(id -u)
      systemctl --user enable --now agent-forge-link` (link survives reboots again). Leave the llama-servers on svc.sh.
- [x] A0. (done 22:00: productive, see RESULTS.md) Hard latent pilot (results/latent/Qwen_Qwen3-8B_n40_s20261115_hard.json): add to RESULTS.md next to the
      sweep. Productive = text_full well off ceiling and off the floor, and conditions differ beyond the CI.
- [x] A. (done 21:50, see RESULTS.md) Spark latent sweep (~/agent-forge/run_sweep.sh, log sweep.log, results ~/agent-forge/results/latent/*ctx*):
      when finished, pull the summaries (accuracy, CI, bytes, co-located seconds per condition per ctx length,
      incl. kv_goal) and add a table + reading to docs/RESULTS.md; commit.
- [x] B. (build OK; chain.sh replaces the launcher) Launcher: ~/agent-forge/chain.sh (log chain.log) replaced launch_servers.sh at 21:39. It waits for the
      sweep, runs the hard-task pilot (<=45 min, link message hard-done), then `svc.sh start af-operator af-coder`.
      Old notes: if the llama.cpp build failed (llamacpp_build.log), fix and rebuild
      (PATH=/usr/local/cuda/bin, CUDACXX, -DCMAKE_CUDA_ARCHITECTURES=121, -j 4), then rerun launch_servers.sh.
      If af-operator/af-coder fail to start (~/agent-forge/logs/af-coder.log), fix flags (e.g. drop --mmproj, lower -c).
Changes at 22:40 (read first): queue3 was stopped. Its coder runs looped and one call generated 29,400 tokens
(runaway thinking filled the 64k context). Fixes (commit f166b06 + follow-up): sampling per model card (Qwen3.x
0.6/0.95/top_k 20/presence 1.5, not greedy), max 4096 new tokens per call, no signals outside the sandbox, a retry
when the forced final answer is a textual tool call. Coder swapped to Heretic Qwen3.6 (af-coder-h, port 8084,
llamacpp:8084@qwen36-heretic); Huihui (af-coder, 8082) is stopped and gets the same benchmarks in F for comparison.
Old greedy partial results: results/qwen36-ablit/v0002/bench_c.greedy-harness-v2.json (3 tasks, 1.33).

23:52: Spark memory hit 121/121 GB (operator + coder + 62 GB judge). af-coder-h was stopped while the operator runs
bench C; ~/agent-forge/coder_after_judge.sh restarts it once Ollama has unloaded the judge (B and D need no judge).
If queue4 waits on port 8084 and the judge is gone, run `svc.sh start af-coder-h` by hand. Coder bench C done:
2.263 (1.75-2.78), 19 graded, 1 infra error (rerun it later: bench-c resumes and reruns infra rows).
00:51: operator (Holo4-heretic) bench C done: 1.90 (1.37-2.43), 20 graded, 0 infra errors. coder_after_judge.sh
restarted af-coder-h (60 GB free); queue4 is on bench-b coder-h (started 00:14).
01:51: coder-h bench B done: 0/68 checks over 11 graded tasks, 1 infra error (#591, 0.5 s; rerun later). Checked:
not a harness bug (check() reads files directly, so new untracked files count). The coder edits other files or
none: it writes a brainstorming spec (copied from the history) or a reportlab PDF instead of the file Claude
changed; only 2 of 11 runs called done. B is strict file matching (old model also 0/7), so it does
not separate models yet; say so in D. queue4 is on bench-b operator+coder-h (started 01:23).
02:51: operator+coder-h bench B done: 0/72 checks, 12 graded, 0 infra errors, only 3 handoffs to the coder.
7 of 12 runs ended at step 0 (~20 s): Holo4-heretic answered in prose (a drop-in code sample, a question back,
or a tool call written as text) instead of calling a tool, and the harness ends a run on the first reply
without tool calls. Candidate fix for D/R (not applied, so queue4 stays comparable): harness v3 nudges once
when a reply has no tool call and no done, and parses textual tool calls. queue4 is on bench-b
coder-h+coder-h (started 01:57).
- [x] C. (queue4 done 03:54; bench-d coder-h stopped: no token cap on vision, rerun in work/queue5.sh) Mac queue work/queue4.sh (log work/queue4.log; jobs: bench-c coder-h, bench-c operator, bench-b coder-h,
      bench-b operator+coder-h, bench-b coder-h+coder-h, bench-d operator, bench-d coder-h): if not running and not
      "queue4 done", restart it with `nohup caffeinate -i work/queue4.sh > work/queue4.log 2>&1 &` (benches resume
      from saved results when settings match, and rerun infra_error rows). If a job ended with only infra errors,
      find out why (curl the server through its Sync port by hand), fix, rerun that job. If the Spark's servers are
      down: `~/agent-forge/svc.sh status`, `svc.sh start af-operator af-coder-h`.
- [x] D. (done 03:58, RESULTS.md "queue4"; coder bench D row is filled in when work/queue5.log says done) When queue4 is done: write results per model to docs/RESULTS.md (B mean score + CI + infra errors, C overall
      + null control, D accuracy + seconds), compare with the baseline, commit.
03:58: bench D fixed (b99617d): Holo4 answers in 0-1000 coordinates (now 53.5%, was 0% read as pixels) and the
vision path had no max_tokens. af-operator is STOPPED so coder + judge fit; work/queue6.sh (log queue6.log) waits
for queue5, then runs R for the coder and reruns the coder's bench C infra row. After queue6: `svc.sh stop
af-coder-h; svc.sh start af-operator` and run R for the operator (one model + judge at a time from now on).
Harness stays at 2 for R so R is comparable with queue4; harness v3 (nudge) is a separate, later decision.
04:54: coder bench D (queue5) 0/43 as pixels: Qwen3.6 also answers in 0-1000 (15/43 = 34.9% re-read post hoc;
22 answers empty, thinking used the 4096 cap). Convention fix now covers Qwen3.5/3.6/VL. Replay run (queue6) showed
two harness bugs, both only on rows that were infra errors: an orphan tool message in the history (Jinja 500,
now sent as user text) and a request over the 64k context (400, now ends the run and is graded, with a trimmed
forced-final turn). work/queue7.sh reruns the replay infra rows after queue6. Spark memory: 1 GB free, 22 GB
available (coder + judge), fine.
05:53: coder R done: replay 2.35 (1.89-2.81), offline 2.13, 20 graded; coder bench C (no replay) now 20 graded,
2.20. Spark swapped: af-coder-h stopped, af-operator up; work/queue8.sh (log queue8.log) runs R for the operator.
06:52: operator R done (queue8 ended 06:12): replay 1.85 (1.49-2.21), offline 2.00, 20 graded, 0 infra.
06:56: Spark servers stopped (link still up), Ollama empty, 102 GB available. D2 started on the Spark (pid 493656,
~/agent-forge/hard200.log, Qwen3-8B loaded). Next watcher: record D2 when it ends, then T1.
- [x] R. Replay baselines (servers af-operator + af-coder-h still up, right after D): on the Mac
      `python -m forge bench-c data/snapshots/v0002 --model llamacpp:8084@qwen36-heretic --replay-tools` and the same
      with `--model llamacpp:8081@holo4-heretic`. Writes bench_c_replay.json (also an "offline" score). Record; commit.
07:55: D2 done (ended 07:31): kinds 1-3 text_card 0.647, kv_goal 0.420, latentmas 0.387, text_full 0.293, kv_full
0.273 (n=150 each); recorded in RESULTS.md. Spark idle (3 GB used). Starting T1.
- [x] D2. `~/agent-forge/svc.sh stop af-operator af-coder-h`, then on the Spark run
      `cd ~/agent-forge && PYTHONPATH=$HOME/agent-forge HF_HUB_OFFLINE=1 setsid nohup ~/vllm-nano/.venv/bin/python -m latent bench --model Qwen/Qwen3-8B --n 200 --seed 7 --level hard --conditions text_full,text_card,kv_full,kv_q8,kv_goal,latentmas > hard200.log 2>&1 &`
      (~1 h, ~20 GB); when it ends record per-kind accuracy for kinds 1-3 in RESULTS.md.
08:12: T1 attempt 1 (from_pretrained then .to(cuda)) and attempt 2 (device_map=cuda) both held two copies of the
65 GB weights and drove the Spark to 113 GB used / swapping; both killed by hand before it locked up. Fix:
train_lora.load_model builds the model on the device and copies tensors in one at a time from safetensors
(tests added, 79 pass). Attempt 3 runs with ~/agent-forge/train_guard.sh (kills training if available < 8 GB):
loaded fine, 21.2M trainable params (LoRA skips the fused MoE experts), 111 GB used during the first eval.
No causal_conv1d / flash-linear-attention on the Spark, so the linear-attention layers use slow reference code.
08:58: T1 attempt 3 got through load and eval but the first training step at max-len 16384 used the last free
memory; the guard killed it (08:13). Added --max-mem-gb (default 100: PyTorch raises OOM instead of swapping the
Spark) and ran at --max-len 8192: works, 124 tok/s with reference kernels. Installed flash-linear-attention 0.6.0
(+ fla-core, --no-deps, pure Triton) in ~/vllm-nano/.venv: 417-550 tok/s (3.4-4.4x). Smoke result: eval loss
1.0636 -> 1.0485 after 3 steps, adapter saved, GGUF 85 MB, llama-server af-coder-t loads it and answers sensibly
(60 tok/s). T1 done. T2 started 08:56: `./train.sh coder v1 5500000 --max-len 8192 --max-mem-gb 100` with
train_guard.sh; 779 examples (about all coder data at 8k), 98 steps, expected ~3.5-4 h (done ~12:30).
09:51: T2 at step 31/98, ~555 tok/s, train loss 1.0-1.5, 98 GB used, guard quiet; ETA ~11:55.
10:52: T2 at step 65/98; eval loss 1.0636 -> 0.935 at step 50. work/queue9.sh is ready for T3 (bench-c replay,
bench-c, bench-b on llamacpp:8085@qwen36-heretic-tuned). When T2 ends: `svc.sh start af-coder-t` (picks the newest
adapter, coder-v1), check /lora-adapters names coder-v1, then start queue9 on the Mac.
11:38: Malek switched the Mac to Uni Wi-Fi (192.168.11.71); hotspot_keeper.sh stopped so it does not pull the
Mac back to the hotspot. Spark reachable through Sync (JupyterLab ok), GitHub ok. T2 at step 92/98.
- [x] T1. Training smoke test (Spark GPU free: no llama-servers, Ollama unloaded with keep_alive 0):
      `cd ~/agent-forge && setsid nohup ./train.sh coder smoke 200000 --max-steps 3 --save-every 3 > train_smoke.log 2>&1 &`.
      Check train_smoke.log and adapters/coder-smoke/log.jsonl: it loads (qwen3_5_moe via AutoModelForCausalLM; if
      not, fix train_lora.py to load the text model), memory stays under ~110 GB (free -g), note tok_per_s, and the
      GGUF conversion works. Fix what fails (max-len 8192 if memory runs out).
11:53: T2 done 11:50: 98 steps, eval loss 1.0636 -> 0.935 (step 50) -> 0.928 (final), adapters/coder-v1/coder-v1.gguf.
af-coder-t up on the Spark with coder-v1 (checked /lora-adapters), Sync tunnels 8085 + 11434 open, work/queue9.sh
(T3) started 11:53 on the Mac (log queue9.log).
- [x] T2. Coder training: pick --max-tokens so the run takes <= 4 h at the measured tok_per_s (cap 15M = all data),
      `setsid nohup ./train.sh coder v1 <tokens> > train_coder_v1.log 2>&1 &` (link message train-done). Check
      eval_loss falls in adapters/coder-v1/log.jsonl.
12:50: Malek: switch the coder base to the 27B (dense, ~9x the active parameters of the 35B-A3B), keep queue9
running first, then score the 27B, then score Claude as the ceiling. Done so far: 0bserverx Qwen3.8-27B heretic
GGUF Q4_K_M + mmproj in ~/agent-forge/models/qwen38-heretic, bf16 safetensors downloading to hf/qwen38-heretic
(Qwen3_5ForCausalLM, 64 layers, hidden 5120); svc.sh entries af-coder-27 (8087) and af-coder-27t (8088, newest
adapters/coder27-*). forge/anthropic_backend.py added (anthropic:claude-opus-5-5, effort high, same harness and
judge); the `ant` login has expired, waiting for Malek to run `ant auth login`.
Next: when queue9 ends, `svc.sh stop af-coder-t; svc.sh start af-coder-27`, `nvsync open malekhassan 8087`,
bench-c --replay-tools --model llamacpp:8087@qwen38-heretic; Claude replay run in parallel once logged in.
12:52: T3 bench C replay: tuned coder 2.15 (1.69-2.61) vs 2.35 untuned, no gain. queue9 now on bench C (no replay),
then bench B. Claude run still blocked: the `ant` login expired on 2026-11-04.
13:51: T3 bench C (no replay): tuned coder 1.90 (1.50-2.30) vs 2.20 untuned. Training lowered both runs. Bench B
running since 13:27 (~35 min). `ant` login still expired.
13:59: T3 done (bench B 0/72). Spark switched: af-coder-t stopped, af-coder-27 (Qwen3.8-27B Heretic Q4_K_M) up
on 8087, 96 GB available, decode ~12 tok/s (dense, vs the MoE's ~3B active). Tunnel 8087 open; queue10
(bench-c --replay-tools, then bench-c, on llamacpp:8087@qwen38-heretic) started 13:59.
14:20: Malek logged in again (`ant auth login`). Claude backend smoke test passed (tool call, thinking replay).
queue11 (Claude Opus 5.5 effort high, bench-c --replay-tools, reference ceiling) started 14:16 alongside
queue10. 27B: first task took 609 s (35B: 181 s), so its replay run needs ~2 h.
14:55: Claude replay at 17/20, mean 3.0 so far; 1 infra error ('why did you stop': 400, a call with no tool_result
when only some calls of a turn got results). Fixed in to_messages (flush after partial results) with a test;
the failed task reruns after queue11 (bench C resumes graded tasks). 27B at 3/20: 1.67 vs 2.00 for the 35B on the
same tasks, ~10-13 min per task.
15:05: Claude ceiling (bench C replay, 20/20 after the rerun): 2.95 (2.31-3.59), offline 3.00. The target 4.0 is
above Claude's CI: 5 tasks are bare follow-ups whose context the harness does not hand over (Claude 1 on all).
Reported to Malek; the target or the benchmark's context window needs his call. 27B run continues.
15:51: 27B replay at 8/20: 1.88, same as the 35B (1.88) and Claude (1.75) on those 8; ~14 min/task, done ~17:30-18:00.
O1 (operator training) waits: the Spark is busy with the 27B run (one heavy job at a time).
16:52: 27B replay at 15/20: 2.80, vs 2.27 for the 35B and 2.73 for Claude on the same 15.
17:20: 27B replay done: 2.85 (2.28-3.42), offline 3.13; new best (35B 2.35, Claude 2.95). No-replay 27B run stopped
(4 h) to free the Spark. Spark back on af-coder-h (8084); queue12 (A1 v2, then A2 v3, 35B) started 17:18. The
winner then runs on the 27B.
17:51: A1 (agent v2, 35B) at 13/20: 2.38 vs 2.23 for v1 on the same tasks (small so far). v3 follows.
18:55: A1 v2 2.45 (+0.10, noise). A2 v3 1.90: the self-check pulled the 35B to the session's first request and
replaced the answer with a checklist. v4 (self-check quotes the current message, final = the answer) on the 35B
in queue13, started 18:56.
19:42: A2 v4 2.45 = v2 (4 better, 4 worse): self-check dropped. Spark on af-coder-27; queue14 (27B, agent v2,
bench C replay, ~3 h) started.
21:52: Malek: do not block reads of ~/.claude/projects ("it can be beneficial"). Open question to him: cut those
reads at each task's time (past readable, the turn's own outcome not). Until he decides, distill --build drops
attempts that read the logs (3 of the first 20 did, one grepping the session files for an id); no benchmark
run has read them so far (checked all bench_c_replay results).
22:00: Malek: leave the log-reading attempts out of training ("more honest"); reads stay open. Bench C now
reports overall.read_session_logs (tasks whose attempt touched ~/.claude/projects) next to every score.
22:20: 27B agent v2: 2.50 vs 2.85 v1 (1 better, 5 worse): two runs ended mid-work at 15 steps (forced final
skips a last reply that has text), one is judge noise (5 vs 1 for the same answer). A1 closed: no gain. 27B keeps
v1. Spark free until the A3 data is ready (~01:00).
22:42: 27B serving with --spec-type ngram-mod (copy-heavy 12 -> 124 tok/s, fresh unchanged). af-coder-27 up.
- [x] T3. Evaluate the tuned coder: `svc.sh start af-operator af-coder-t` (af-coder-t = Heretic Q4 + --lora newest
      coder adapter, port 8085), `nvsync open malekhassan 8085`, then bench-c --replay-tools, bench-c, bench-b with
      --model llamacpp:8085@qwen36-heretic-tuned, and bench-b --model llamacpp:8081@holo4-heretic --coder
      llamacpp:8085@qwen36-heretic-tuned. Compare with the untuned numbers (R, D). Record; commit.
- [ ] O1. Operator training: servers off, `./train.sh operator v1 <tokens>` (LoRA trained on base Holo4 weights,
      applied to the Heretic GGUF: check it loads and that bench numbers do not drop). Then `svc.sh start
      af-operator-t <coder>` (port 8086, `nvsync open malekhassan 8086`), bench-d and bench-c --replay-tools with
      --model llamacpp:8086@holo4-heretic-tuned, bench-b duo with the best coder. Record; commit.

## Overnight 16-17.11 (Malek left at ~22:50: "keep running on schedule overnight")

Watcher steps, in order; one Spark job at a time, never poll the Spark in a loop:
- [x] N1. 27B training smoke test (`DATA=data/sft/v0002/coder.jsonl train.sh coder27 smoke 60000 --max-steps 2
      --max-len 4096`, log ~/agent-forge/train_coder27_smoke.log). Pass = "train exit 0" and a gguf. If it fails:
      diagnose, fix locally with a test, push train_lora.py by base64 (work/push_train27.js pattern), rerun.
      23:05: passed after two fixes: tool-call arguments as dicts for the Qwen3.8 template (70bbd9d), and LoRA
      conversion without an MTP head (work/convert_lora_nomtp.py, used by train.sh for coder27 only). 2 steps at
      ~165 tok/s (a third of the 35B's 555), eval 0.853; af-coder-27t loads the adapter and answers.
23:00: both A3 workers stopped on the plan's 5-hour limit (resets 00:50 Europe/Berlin). 57 attempts saved
(runs.jsonl 50, runs-w2.jsonl 7). After 00:50: rerun work/queue15.sh and work/queue16.sh (both resume where they
stopped), then N2-N4 as below.
00:52: plan limit reset; queue15 and queue16 resumed (logs appended).
02:45: both A3 workers stopped again on the plan limit (resets 05:50). distill --build: 106 attempts, 39 kept
(overall >= 4), 6 dropped for reading the logs, mean 3.06; data/sft/v0002-distill/coder.jsonl, 2.8 MB, median
example ~71k chars (~20k tokens: use --max-len 16384 or accept truncation at 8192). Next: copy it to the Spark
(base64), N3 training, N4 benchmark; after 05:50 rerun queue16 for round 2.
05:55: data on the Spark (hash ok, 39 lines). N3 started: `DATA=data/sft/v0002-distill/coder.jsonl train.sh coder27 v1
0 --max-len 16384 --accum 2 --warmup 3 --eval-frac 0.1 --save-every 5` (log train_coder27_v1.log), guard running.
If 16384 OOMs: rerun with --max-len 12288. queue16 resumed (plan limit reset 05:50).
06:20: N3 first try OOM at 16k (14.3 GiB float32 logits, 248k vocab) and only 15 of 35 examples kept (truncation
cut the new turns). Fixed: sparse_loss (logits only where a label follows) and fit() (drop the oldest history
instead of the end), with tests. Restarted 06:18, same flags; the guard exits on its own when no training runs
(start it after train.sh).
- [x] N2. When queue15 (A3, 100 train turns) is done: `.venv/bin/python -m forge.distill data/snapshots/v0002 --build`
      (overall >= 4, log-reading attempts dropped), copy data/sft/v0002-distill/coder.jsonl to the Spark (base64,
      same folder), record stats in RESULTS.
- [~] N3. Train the 27B LoRA: servers off, train_guard.sh running (start it detached with setsid; `pgrep -f`
      matches itself), `DATA=data/sft/v0002-distill/coder.jsonl train.sh coder27 v1 0 --max-len 8192` (all
      examples; 2 epochs = --max-tokens 0 then resume if the data is small). Record eval loss.
- [ ] N4. `svc.sh start af-coder-27t` (port 8088, ngram-mod on), `nvsync open malekhassan 8088`, then
      bench-c --replay-tools --model llamacpp:8088@qwen38-heretic-tuned (queue16). Compare with 2.85 (27B v1).
- [~] N5. Meanwhile, if queue15 finished and the Spark is busy: extend A3 to more train turns
      (`forge.distill --n 400 --seed 1` reuses the first 100 only if the task file matches; use a new seed and
      append) so round 2 has more data.
      22:55: started early (only ~1/3 of attempts reach 4): queue16, worker w2, --n 400 --seed 2, runs-w2.jsonl;
      build merges all runs*.jsonl and drops duplicate turns.

## Decision (Malek, 2026-11-16 16:00): the benchmark and the target stay

"we will not lower benchmarks to reach them". Benchmark C (tasks, history window, work folder, judge, rubric) and the
target (>= 4.0 with --replay-tools) are fixed. Claude Opus 5.5 scores 2.95 under the same conditions, so 4.0 means
an agent better than the source model here; the gains must come from the agent side, the same for every model and
recorded in each result:
- [x] A1. Find the subject first: agent prompt v2 that makes the agent state what the request refers to (from its
      history, then by searching the work folder and recent files) before acting, and check every item of the ask
      before the final message. Run on the 35B coder (cheap, local) once the Spark is free; Claude only if it helps.
      16:10: `bench-c --agent v2` added (v1 = unchanged settings and files); work/queue12.sh runs it on the 35B.
- [x] A2. Self-check: the agent re-reads the request and its own result and continues if something is missing
      (no access to the judge or the reference: selecting with the benchmark's judge would be gaming it).
      16:55: harness self_check (once, when the agent first says it is done) = `--agent v3`; queue12 runs v2 then v3.
- [~] A3. Distillation: Claude trajectories through this harness on train-split tasks (with A1/A2), filtered by a
      separate checker, as LoRA data for the best local coder. Needs Malek's OK for the API spend first.
      20:50: Malek OK'd the spend through `claude -p` on his plan. forge/claude_p_backend.py (claude-p:<model>:
      the CLI is only the model, its tools off, empty folder, no settings/MCP; the harness runs every call in its
      sandbox + replay) and forge/distill.py (bench C task builder on train segments, subagent segments excluded,
      0 prompt overlap with the benchmark; Claude-as-checker against the recorded outcome; loss only on the new
      turns via loss_from). queue15: 100 tasks, agent v2. Then `--build` (overall >= 4) and a LoRA on the 27B.

## Target and loop (Malek, 2026-11-15 23:00: "loop until 4 is reached")

Target: benchmark C with --replay-tools, overall >= 4.0 (1-5 judge scale, n=20) for the coder agent, with the
offline subset and benchmark B reported next to it. Without replayed tools the ceiling is ~3.2 (12/20 tasks need
apps), which is why the target uses the replay run.
Loop after T3/O1, one change per iteration, keep everything else fixed: (1) 2x the token budget, (2) a second epoch
(--max-tokens 0 plus resume), (3) rank 32. After each: retrain, re-run T3's benchmarks, record. Stop when the target
is reached, or after 3 iterations, or when two iterations in a row improve bench C by < 0.1 (diminishing returns).
Then write the summary for Malek at the top of docs/RESULTS.md (what reached what, honestly, with CIs), restart the
best servers, delete the watcher cron job. E and F below are optional comparisons: run them only if the GPU is idle.

- [ ] E. (optional) Old model rerun with the same settings (Ollama, Spark servers stay stopped): on the Mac
      `python -m forge bench-c data/snapshots/v0002 --model qwen3-coder-next-abliterated-128k` then bench-b (same
      model). Unload it afterwards (`curl sync://ollama/api/generate -d '{"model":"...","keep_alive":0}'`). Record, commit.
- [ ] F. (optional) Coder comparison, one server at a time: `svc.sh start af-coder` (Huihui, 8082): bench-c and bench-b with
      --model llamacpp:8082@qwen36-ablit; `svc.sh stop af-coder`. Then `svc.sh start af-dense`,
      `nvsync open malekhassan 8083`, bench-c and bench-b with --model llamacpp:8083@qwen38-ablit; `svc.sh stop af-dense`.
      Record; commit.
- [ ] G. Then `svc.sh start af-operator <best coder>` and leave a short summary at the top of docs/RESULTS.md for Malek
      (what is best per role, by how much, what failed). Delete the watcher cron job.
