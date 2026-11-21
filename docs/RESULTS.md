# stem results log

## Summary for Malek · 2026-11-21 05:45 · the "loop until 4" run is finished, target not reached

**Best coder: the 27B + LoRA v3, benchmark C (replay) 2.90 (2.27-3.53)**, level with the untuned 27B at 2.85
(2.28-3.42) and with Claude Opus 5.5 run through the same harness at 2.95. The target was 4.0; nothing reached it,
and no LoRA is measurably better than the untuned model (all CIs overlap).

| Coder (Qwen3.8-27B-heretic) | Bench C replay | Offline (8) | vs untuned (better/worse tasks) |
|---|---|---|---|
| untuned | 2.85 (2.28-3.42) | 3.13 | - |
| LoRA v1 (36 claude -p attempts) | 2.30 (1.75-2.85) | 2.38 | 2 / 11 |
| LoRA v2 (67 attempts, 19 with thinking) | 2.45 (1.99-2.91) | 2.38 | 1 / 6 |
| **LoRA v3 = v2 + second epoch** (loop 1) | **2.90** (2.27-3.53) | 3.00 | 4 / 5 |
| LoRA v4 = v3 + rank 32 (loop 2) | 2.55 (1.89-3.21) | 2.88 | 4 / 9 |

What the loop did: change (1) "2x token budget" did not apply (v2 already used every example); (2) a second epoch
gave +0.45; (3) rank 32 gave -0.35 although its eval loss was slightly lower (0.721 vs 0.725). With the planned
changes used up, the loop stops after 2 iterations. Honest reading: with 67 examples and n=20 tasks the differences
between v2, v3, v4 and the base are mostly noise (one task moving 1->5 shifts the mean by 0.2). The limit is the
data, not the knobs: more checked attempts would help most, but the free-only rule (20.11) rules out the paid API
worker that produced them, so new data has to come from free sources (claude -p in-harness runs, Malek's own
sessions).

Running now on the Spark: af-coder-27t with adapters/coder27-v3 (port 8088) and af-operator (holo4, port 8081),
link daemon; judge unloaded; 71 GB available. Every run had 0 infra errors and no truncated generations.

## 2026-11-21 · 27B + LoRA v4 (v3 + rank 32, alpha 64) · benchmark C with replay

Loop iteration 2 (change: rank 16 -> 32, alpha 32 -> 64 to keep the ratio; data, 2 epochs, seed as v3). 61 steps,
eval loss 0.806 -> 0.721 (v3 0.725). **Overall 2.55 (1.89-3.21)** against 2.90 for v3 (3 tasks better, 7 worse)
and 2.85 untuned (4 better, 9 worse); offline 2.88 (1.68-4.07). Lower eval loss did not carry over to the agent
benchmark. 20/20 graded, 0 infra errors, read_session_logs 0; 259 s/task.

## 2026-11-21 · 27B + LoRA v3 (v2's data, 2 epochs) · benchmark C with replay

Loop iteration 1 of 3 (change: a second epoch, everything else as v2). **Overall 2.90 (2.27-3.53)** against 2.85 for
the untuned 27B (4 tasks better, 5 worse) and 2.45 for v2 (7 better, 3 worse); offline subset 3.00 (2.02-3.98) vs
3.13 untuned and 2.38 for v2. +0.45 over v2, the first LoRA that matches the base, but the CIs overlap almost
completely, so "as good as untuned", not better. Biggest gains over v2 are on tasks v2 dropped (#529 2->5, #125 2->5,
#2 2->4, #50 3->4). Training: 61 steps, eval loss flat (~0.725) over the second epoch while train loss kept falling.
20/20 graded, 0 infra errors, read_session_logs 0, no truncated generations in the server log; 442 s/task.

## 2026-11-20 · 27B + LoRA v2 (67 checked Claude attempts, 19 with thinking) · benchmark C with replay

**Overall 2.45 (1.99-2.91)** against 2.85 for the untuned 27B (1 task better, 6 worse) and 2.30 for v1 (8 better,
5 worse); offline subset 2.38 vs 3.13 untuned. v2 is +0.15 over v1, within noise, and still below the base: LoRA
training on this data has not beaten the untuned 27B yet. Understood drops most (3.50 -> 2.75). One attempt (#1226,
Chrome tab grouping) searched ~/.claude memory/session logs instead of doing the task (read_session_logs = 1, graded
1). Run: 18 rows on 20.11 14:28-17:15, the 2 rows hit by the coder outage that afternoon (#1226 connection reset,
#1053 timeout) rerun 17:30-17:45 with the coder healthy; no infra errors left. 404 s/task.

## 2026-11-17 · 27B + LoRA v1 from Claude's in-harness attempts (A3) · benchmark C with replay

36 training examples (39 attempts the checker scored >= 4 on train-split turns, claude -p, no thinking), 18 steps,
max-len 16384 with the oldest history trimmed and logits only where labels follow; eval loss 0.880 -> 0.822.

**Overall 2.30 (1.75-2.85)** against 2.85 for the untuned 27B: 2 tasks better, 11 worse. The transcripts show the
same failure as the 35B's LoRA v1: turns end mid-work with a text reply that announces the next step ("I'll go back
to the Financial Planning page and grab the PDF's download link directly.") and no call. The claude -p data has no
thinking, so the chat template put an empty <think></think> before every trained reply, and the model learned to
skip planning. Fix for v2: empty think blocks are no longer training targets, and the API attempts carry Claude's
summarised thinking as reasoning_content. One attempt read ~/.claude/projects (read_session_logs = 1).

## 2026-11-16 · 27B speed: n-gram speculative decoding · llama.cpp 75fd7f9, Spark GB10

No small model shares the 27B's vocabulary (248k) and the Heretic has no MTP head, so a draft model is out.
llama.cpp's n-gram speculation drafts from text already in the context, needs no memory, and keeps the output
(same token count and text on the fresh prompt). work/spec_bench.py, 700-token cap, sampling as in the benchmark:

| Server | fresh text (tok/s) | copy-heavy edit (tok/s) | draft accepted |
|---|---|---|---|
| plain | 12.2 | 12.1 | - |
| --spec-type ngram-map-k | 12.1 | 18.8 | 254/254 |
| --spec-type ngram-mod | 12.0 | **124.0** | 642/647 |

ngram-mod is on for af-coder-27 and af-coder-27t from 22:40 (svc.sh). Agent turns repeat file contents and tool
output a lot, so benchmark runs should shorten; new text stays at ~12 tok/s.

## 2026-11-16 · agent-side instructions A1/A2 on benchmark C with replay · 35B coder (Qwen3.6 heretic, port 8084)

The benchmark is unchanged (Malek: "we will not lower benchmarks"); only the agent's own instructions differ, the
same for every model, recorded as settings.agent.

| Agent | Understood | Approach | Correctness | Completeness | Overall | s/task |
|---|---|---|---|---|---|---|
| v1 (as before) | 2.85 | 2.45 | 2.20 | 1.90 | 2.35 | 101 |
| v2: say what the request refers to first, check every item | 3.00 | 2.55 | 2.25 | 2.05 | 2.45 | 93 |
| v3: v2 + one self-check before done | 2.25 | 2.05 | 1.75 | 1.70 | 1.90 | 145 |
| v4: v2 + a self-check quoting the current message | 3.20 | 2.45 | 2.20 | 2.10 | 2.45 (1.93-2.97) | 121 |
| 27B v1 (for comparison) | 3.50 | 2.90 | 2.70 | 2.40 | 2.85 (2.28-3.42) | 566 |
| 27B v2 | 3.05 | 2.65 | 2.25 | 2.00 | 2.50 (1.94-3.06) | 434 |

v2 is +0.10, within noise. v3 is worse, and the transcripts show why: "read the user's request again" made the
35B check the session's first request instead of the message it was answering, and its final message became a
checklist of the whole session ("Locate course content about OS03: Done") in place of the answer. 7 tasks fell to
1-2 that way. v4 quotes the current message in the self-check and asks for the full answer as the final message.
v4 fixes the drop but adds nothing over v2 (4 tasks better, 4 worse; one final message is again a self-review).
The self-check is dropped; v2 goes on to the 27B.

On the 27B, v2 scores **2.50** (1.94-3.06) against 2.85 for v1 (1 task better, 5 worse; offline 2.38 vs 3.13).
Two of the drops ran out of the 15 steps mid-work, and their last progress note ("Uploading now...") counted as the
final message: the forced final turn only fires when the last reply has no text. That gap affects every model the
same way and is left as it is so earlier scores stay comparable (noted for a harness v3). One drop is judge noise:
"A + the thread through all three" got 5 (v1) and 1 (v2) for nearly the same answer ("Locked in: Approach A, a
Driveway mode..."). Single tasks swing by up to 4 points, so differences of +-0.3 at n=20 are within noise. The 27B
keeps v1.

## 2026-11-16 · Qwen3.8-27B heretic on benchmark C with replay · Q4_K_M, llama.cpp, Spark port 8087

Malek switched the coder base to the dense 27B (about 9x the active parameters of the 35B-A3B). Same harness,
sampling, judge and tasks as the 35B coder's 2.35.

**Overall 2.85 (2.28-3.42)**, offline subset 3.13, against 2.35 for the 35B and 2.95 for Claude. Per task against
the 35B: 10 better, 4 worse, 6 the same (two-sided sign test p = 0.18, so likely but not yet proven at n=20).
Against Claude: 6 better, 6 worse. It is level with the 35B on the bare follow-ups ("Continue", "continue with
the build": 1 for every model) and ahead where the request carries its context (5s on "is the site done", "I
submitted", the grant remarks). Understood 3.50 vs 2.85: the larger model reads the request in a long history
better, which is what the 35B lacked.

Cost: 566 s per task against 101 s (decode ~12 tok/s; every token passes all 27B weights). Fine for the coder,
too slow for the operator. The no-replay run was stopped after its start to free the Spark for A1/A2.

## 2026-11-16 · Claude ceiling on benchmark C with replay · Claude Opus 5.5, effort high, Anthropic API

The model the sessions came from, through the same harness, replayed tools, judge and 20 tasks (forge
anthropic_backend: thinking blocks replayed, prompt caching, max 16k tokens per reply). It is a reference, never a
candidate. One task first failed on a backend bug (a turn whose calls got only some results; fixed in f55705a) and
was rerun.

**Overall 2.95 (2.31-3.59)**, offline subset 3.00. The 35B coder's 2.35 is 0.6 below it; the target 4.0 is above
Claude's whole confidence interval.

- Claude scores 4-5 on 10 of 20 tasks: the ones whose request carries its own context ("the numbers on hoopergpt are
  pure cap", "I have 120 credits...", "I submitted").
- It scores 1 on 5 tasks, all bare follow-ups whose subject is outside the window the harness hands over:
  "Continue", "continue with the build", "why did you stop", "Yes, do it. Upload this version now...", "is there
  anything now in my github...". The judge's reasons: it acted on an unrelated project. Three of them are 1 for the
  35B coder as well.

So under this harness 4.0 is not reachable by any model, including the one that did the original work: the limit
is what context reaches the agent (the session history window and the working directory), not the model's size.
Without those 5 tasks Claude averages 3.60 and the 35B coder 2.60.

## 2026-11-16 · queue4: operator and coder baselines · snapshot v0002 · harness 2

Operator: Holo4-35B-A3B-heretic Q4 (llama.cpp, Spark port 8081). Coder: llmfan46 Qwen3.6-35B-A3B-uncensored-heretic
Q4_K_M (port 8084). Sampling per model card (Qwen3.x: temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5),
max 4096 tokens per reply, seed 0. Judge for C stays greedy. Infrastructure errors are excluded and rerun later.

**Benchmark C** (20 held-out turns, judge scores 1-5, no replay of app/web tools, so the ceiling is about 3.2):

| Model | Graded | Infra | Understood | Approach | Correctness | Completeness | Overall (95% CI) | s/task |
|---|---|---|---|---|---|---|---|---|
| coder Qwen3.6-heretic | 20 | 0 | 2.65 | 2.10 | 2.00 | 1.70 | **2.20** (1.70-2.71) | 124 |
| operator Holo4-heretic | 20 | 0 | 2.50 | 1.70 | 1.80 | 1.65 | **1.90** (1.37-2.43) | 57 |
| earlier: qwen3-coder-next (v1 harness) | | | | | | | 1.63 | |
| null control (empty answer) | | | | | | | 1.00 | |

The coder's row includes the one task that first hit an infrastructure error (orphan tool message, fixed in
a252aa4); with 19 tasks it was 2.26.

**Benchmark C with replayed app/web tools** (`--replay-tools`: app and web tools answer from the recording, so a 4 or
5 is reachable; "offline" = the 8 tasks that need no such tool). This is the score the training target (>= 4.0) uses.

| Model | Graded | Infra | Understood | Approach | Correctness | Completeness | Overall (95% CI) | Offline (8) | s/task |
|---|---|---|---|---|---|---|---|---|---|
| coder Qwen3.6-heretic | 20 | 0 | 2.85 | 2.45 | 2.20 | 1.90 | **2.35** (1.89-2.81) | 2.13 | 101 |
| operator Holo4-heretic | 20 | 0 | 2.35 | 1.90 | 1.50 | 1.40 | **1.85** (1.49-2.21) | 2.00 | 40 |
| coder Qwen3.6-heretic + LoRA v1 | 20 | 0 | 2.60 | 2.35 | 1.85 | 1.75 | **2.15** (1.69-2.61) | 2.00 | 139 |
| coder Qwen3.8-27B-heretic (dense) | 20 | 0 | 3.50 | 2.90 | 2.70 | 2.40 | **2.85** (2.28-3.42) | 3.13 | 566 |
| coder Qwen3.8-27B + LoRA v1 (36 claude -p attempts) | 20 | 0 | 2.80 | 2.80 | 2.35 | 1.90 | **2.30** (1.75-2.85) | 2.38 | 438 |
| coder Qwen3.8-27B + LoRA v2 (67 attempts, 19 with thinking) | 20 | 0 | 2.75 | 2.35 | 1.95 | 1.95 | **2.45** (1.99-2.91) | 2.38 | 404 |
| coder Qwen3.8-27B + LoRA v3 (v2 data, 2 epochs) | 20 | 0 | 3.35 | 3.00 | 2.50 | 2.40 | **2.90** (2.27-3.53) | 3.00 | 442 |
| coder Qwen3.8-27B + LoRA v4 (v3 + rank 32) | 20 | 0 | 3.15 | 2.85 | 2.55 | 2.30 | **2.55** (1.89-3.21) | 2.88 | 259 |
| reference: Claude Opus 5.5, effort high (not a candidate) | 20 | 0 | 3.45 | 3.15 | 2.80 | 2.70 | **2.95** (2.31-3.59) | 3.00 | 71 |

Two of the coder's replay tasks first failed on harness bugs (orphan tool message; a request over the 64k
context) and were rerun after a252aa4. Replay lifts the tasks that need app/web tools to 2.50 (12 tasks), but the
distance to 4.0 is large. The judge's reasons: in all 5 tasks scored 1 the coder acted on the wrong thing (a study
plan, an iOS build, an unrelated repo instead of what was asked), and most 2-3s are partial answers that leave out
items the user wanted. Reading the request in a long history and finishing all of it is what training has to fix.

The operator gains nothing from replay: 1.85 against 1.90 without it (the difference is noise; external-tool tasks
1.75 over 12). It is fast (40 s per task) but its answers are shallow (correctness 1.50, completeness 1.40). The
coder stays the better base for the C target; the operator's job is clicks (D 53.5%) and handoffs.

**Benchmark B** (repo tasks, pass = the files Claude changed are changed and >= 75% similar):

| Setup | Graded | Infra | Checks passed | Mean (95% CI) | Handoffs | s/task |
|---|---|---|---|---|---|---|
| coder alone | 11 | 1 | 0/68 | 0.00 (0-0.26) | | 362 |
| operator + coder | 12 | 0 | 0/72 | 0.00 (0-0.24) | 3 | 167 |
| coder + coder | 11 | 1 | 0/68 | 0.00 (0-0.26) | 8 | 381 |

Reading B: not a harness bug (check() reads the files directly, so new files count). The agents edit other files
or none: the coder copies patterns from the long history (writes a brainstorming spec, builds a PDF with reportlab)
instead of changing the file Claude changed; only 2 of 11 solo runs called done. In the duo, 7 of 12 runs ended at
step 0 because Holo4 answered in prose (sample code, a question back, a tool call written as text) and the harness
ends a run on the first reply without a tool call. B is strict file matching and every model so far scores 0, so it
does not separate models yet. Candidate harness v3: nudge once on a reply without tool call or done, and parse
textual tool calls.

**Benchmark D** (43 clicks on the user's own screens, pass = within max(30 px, 3% width) of Claude's click):

| Model | Items | Infra | Accuracy (95% CI) | s/click |
|---|---|---|---|---|
| operator Holo4-heretic, asked in 0-1000 coordinates | 43 | 0 | **53.5%** (38.9-67.5) | 4.1 |
| operator Holo4-heretic, asked in pixels (first run) | 43 | 0 | 0% (scored as pixels) | 4.2 |
| coder Qwen3.6-heretic, asked in pixels, 4096-token cap | 43 | 0 | 0% scored as pixels; **34.9%** (15/43) re-read as 0-1000 (post hoc) | 42.5 |

Reading D: Holo4 and Qwen3.6 both answer in their trained 0-1000 convention even when the prompt asks for pixels.
Qwen3.6 also left 22 of 43 answers empty: its thinking used the whole 4096-token cap (42 s per click), so the
coder is ten times slower than the operator and much less reliable at clicking. Its 34.9% is a post-hoc re-read of
the saved answers (empties count as misses), not a rerun; the convention fix now covers Qwen3.5/3.6/VL too. Fixed in b99617d
and the next commit (per-model convention, recorded in settings). The operator's computer-use tools must convert the same way.

## 2026-11-15 · benchmark B · qwen3-coder-next-abliterated-128k · snapshot v0002

Model record: Ollama digest 3e0033bef22e…, Q4_K_M, Ollama 0.24.0, temperature 0, seed 0.
Task: "only have one phone for now adapt" (hoopergpt, 2 Swift files).

| Run | Score | Steps | Seconds | What happened |
|---|---|---|---|---|
| 1 | 0/4 | 15 | 958 | Harness bug: tools were named bash/read_file while the history used Bash/Read. Every call returned "unknown tool". Fixed in commit "Harness: use Claude Code tool names". |
| 2 | 0/4 | 30 (cap) | 886 | Tools now used correctly. Model failure: first Read targeted a temp file from the original session outside the sandbox (refused), then the same `xcodebuild` command was repeated 29 times with no edit. |

Reading: tool use works end to end; the base model loops on long, noisy build output instead of acting.
This is one task, so it says nothing about the overall rate (95% interval for 0/1 is 0 to 0.79).

## 2026-11-15 · publish · snapshot v0002

Private HF dataset `malekhassan23/agent-forge-data`, revision 100981de096317e2126b23b761f7d6700f974a39.
6 files checked against the manifest (5 outputs + raw.tar.gz): 0 mismatches. Repo confirmed private.

## 2026-11-15 · benchmark B baseline (all tasks) · qwen3-coder-next-abliterated-128k · snapshot v0002

Single agent, same settings as above. **0/7 valid tasks** (95% CI 0-0.35); 5 of 12 tasks hit a Sync tunnel outage
(0 steps, Connection refused) and are infrastructure errors, excluded and queued for rerun (they had first been
scored as zeros; fixed in commit 0f9e8f7). Failure pattern in the 7 valid tasks: identical Bash command repeated
27-29 times (3 tasks), Read-only loops of 25-28 calls without an edit (2), SendMessage to a non-existent agent (1),
one 7-step run that ran out of wall time. Research note: llama.cpp issue #20164 (fixed in PR #20171) makes this
model loop on tools with several optional parameters; the serving stack may explain part of this.

## 2026-11-15 · link between Mac and Spark

Mac (launchd) <-> Spark (systemd user unit, set up through the Sync JupyterLab tunnel). p2p failed (VPN between
subnets), relay mode + iroh hole punching gave a direct UDP path. Bare round trips: min 10 ms, median 17-63 ms,
jitter episodes 0.9-1.1 s median. Throughput with KV-sized blobs under load: 0.6-2.4 Mbit/s.
End-to-end trigger tested: the Spark's job sent "latent-done" over the link and the Mac started benchmark C.

## 2026-11-15 · latent handoff benchmark (python -m latent bench) · Spark GB10, bf16, transformers 5.17

60 synthetic code-handoff tasks (operator reads a module + traceback, coder answers), seed 20261115, greedy.
"co-located" = sender + receiver time, no serialisation, no wire (operator and coder on the same GPU).

| Qwen3 | condition | accuracy (95% CI) | wire MB | co-located s |
|---|---|---|---|---|
| 0.6B | text_card | 1.000 (0.94-1.00) | 0.0001 | 0.28 |
| 0.6B | text_full | 0.633 (0.51-0.74) | 0.002 | 0.09 |
| 0.6B | kv_full | 0.633 (0.51-0.74) | 70.8 | 0.11 |
| 0.6B | latentmas | 0.933 (0.84-0.97) | 75.8 | 0.59 |
| 0.6B | kvcomm | 0.233 (0.14-0.35) | 22.8 | 0.27 |
| 4B | text_card | 0.983 (0.91-1.00) | 0.0001 | 1.91 |
| 4B | text_full | 1.000 (0.94-1.00) | 0.002 | 0.25 |
| 4B | kv_full / kv_q8 | 1.000 (0.94-1.00) | 91.0 / 46.2 | 0.30 |
| 4B | latentmas | 1.000 (0.94-1.00) | 97.4 | 1.45 |
| 4B | kvcomm | 0.133 (0.07-0.24) | 27.8 | 0.87 |
| 8B | text_card | 0.983 (0.91-1.00) | 0.0001 | 4.16 |
| 8B | text_full | 1.000 (0.94-1.00) | 0.002 | 0.42 |
| 8B | kv_full / kv_q8 | 1.000 (0.94-1.00) | 91.0 / 46.2 | 0.49 |
| 8B | latentmas | 1.000 (0.94-1.00) | 97.4 | 2.56 |
| 8B | kvcomm | 0.383 (0.27-0.51) | 27.8 | 2.01 |

Reading:
- Co-located KV handoff is 6.4x (4B) and 8.5x (8B) faster than the text card at equal or better accuracy: it
  removes the sender's decode. kv_full equals text_full in answers (agreement 0.95-1.00), as it must.
- At ~600-token contexts KV does not beat forwarding the raw text (prefill is cheap); a context-length sweep
  (8k-32k, the size of real operator sessions) is needed before claiming more.
- Confound in 0.6B: text_card and latentmas senders saw the question. kv_goal ablation added (commit b399d4d), not run yet.
- 4B/8B are at ceiling on these tasks, so latentmas vs kv_full is not separable; harder tasks needed.
- Our KVComm reimplementation does not reproduce the paper (13-38% vs near-parity reported at ~30% of layers).
- Across machines the KV blobs (46-97 MB) lose to text at the measured Mac<->Spark bandwidth.

## 2026-11-15 · latent long-context sweep · Qwen3-8B, Spark GB10, bf16, n=30 per length, seed 20261115

Same synthetic handoff tasks, padded with distractor code so the facts sit mid-file. Receiver TTFT = time from
handoff to the coder's first token; co-located = sender + receiver, same GPU, no wire.

| ctx chars | condition | accuracy (95% CI) | wire | sender s | receiver TTFT s | co-located s |
|---|---|---|---|---|---|---|
| 8k | text_full | 1.000 (0.89-1.00) | 8 KB | 0 | 0.84 | 1.11 |
| 8k | text_card | 1.000 (0.89-1.00) | 0.1 KB | 4.28 | 0.09 | 4.64 |
| 8k | kv_full / kv_q8 | 1.000 / 1.000 | 415 / 211 MB | 0.83 | 0.10 | 1.20 |
| 8k | kv_goal | 1.000 | 418 MB | 0.84 | 0.10 | 1.21 |
| 8k | latentmas | 1.000 | 421 MB | 3.22 | 0.10 | 3.59 |
| 32k | text_full | 1.000 (0.89-1.00) | 32 KB | 0 | 4.41 | 4.80 |
| 32k | text_card | 1.000 (0.89-1.00) | 0.2 KB | 10.67 | 0.10 | 11.07 |
| 32k | kv_full / kv_q8 | 1.000 / 1.000 | 1.71 / 0.87 GB | 4.26 | 0.25 | 4.93 |
| 32k | kv_goal | 1.000 | 1.71 GB | 4.21 | 0.22 | 4.82 |
| 32k | latentmas | 1.000 | 1.71 GB | 6.66 | 0.23 | 7.27 |
| 64k | text_full | 1.000 (0.89-1.00) | 64 KB | 0 | 8.48 | 8.89 |
| 64k | text_card | 0.967 (0.83-0.99) | 0.2 KB | 14.22 | 0.09 | 14.58 |
| 64k | kv_full / kv_q8 | 1.000 / 1.000 | 3.46 / 1.76 GB | 8.60 | 0.29 | 9.29 |
| 64k | kv_goal | 1.000 | 3.47 GB | 8.42 | 0.28 | 9.10 |
| 64k | latentmas | 1.000 | 3.47 GB | 11.14 | 0.27 | 11.81 |
| 110k | text_full | 1.000 (0.89-1.00) | 110 KB | 0 | 28.26* | 28.77* |
| 110k | text_card | 0.967 (0.83-0.99) | 0.2 KB | 24.47 | 0.09 | 24.90 |
| 110k | kv_full / kv_q8 | 1.000 / 1.000 | 5.99 / 3.04 GB | 16.96 | 0.49 | 17.94 |
| 110k | kv_goal | 1.000 | 6.00 GB | 16.92 | 0.48 | 17.90 |
| 110k | latentmas | 1.000 | 6.00 GB | 20.33 | 0.49 | 21.57 |

110k characters is about 40k tokens. kv_full and text_full gave identical answers at every length (agreement 1.00).
\* At 110k the same prefill took 28 s as text_full but 17 s as the kv sender. The run hit a CUDA OOM-and-retry at
20:22 under memory pressure (101/121 GB), so the 110k text_full timing is suspect; do not read it as a KV win.

Reading:
- KV handoff moves the prefill from the coder to the operator. On one GPU, end to end, it costs the same as
  forwarding the raw text (same compute). The win is the coder's start: if the operator has already read the
  context, the coder's first token comes in 0.25-0.49 s instead of 4.4-28 s (17x at 32k, 30x at 64k).
- int8 halves the bytes with no accuracy loss at any length.
- kv_goal (operator knows the goal) equals kv_full, so the sender seeing the question was not a confound.
- Every condition is at or near ceiling (text_card 0.967 at 64k and 110k). These tasks cannot separate methods;
  harder multi-hop tasks added (--level hard), pilot queued.
- Across machines it is out of the question at 2.4 Mbit/s (0.4-6 GB per handoff). Latent handoff only makes
  sense with operator and coder on the same box.

## 2026-11-16 · coder LoRA v1 (STATUS T2) · Qwen3.6-35B-A3B-heretic bf16 base, Spark GB10

LoRA rank 16 / alpha 32 on every linear layer of the language model except the MoE router (21.2M trainable
parameters; the fused MoE experts are not nn.Linear and stay frozen), lr 1e-4 cosine, 8 examples per step,
max-len 8192, one pass over 779 coder examples (5.5M tokens) from snapshot v0002 train. 98 steps in 2 h 54 min at
~555 tok/s (flash-linear-attention kernels; 124 tok/s without). Held-out eval loss (32 examples):

| step | 0 | 50 | 98 |
|---|---|---|---|
| eval loss | 1.064 | 0.935 | 0.928 |

The loss gain did not carry over. Benchmark C with replay: **2.15** (1.69-2.61) against 2.35 untuned; per task 5
better, 9 worse, 6 the same; offline 2.00 against 2.13. Within noise of the untuned coder, and not better. Imitating
Claude's logged turns (with its thinking blanked out) at 8k tokens does not teach the model to read the request
and finish it. Benchmark C without replay agrees: **1.90** (1.50-2.30) against 2.20 untuned (understood 2.35,
completeness 1.45). Benchmark B: 0/72 checks, as for every model so far. Malek then decided to switch the coder base to the dense Qwen3.8-27B (12:50).

## 2026-11-16 · latent hard tasks at n=200 (STATUS D2) · Qwen3-8B, Spark GB10, bf16, seed 7, short context

Same task generator as the pilot below, 50 tasks per kind. Kinds 1-3 are the ones that separate (kind 0 is
trivial, 49-50/50 everywhere). Raw file on the Spark: ~/agent-forge/results/latent/Qwen_Qwen3-8B_n200_s7_hard.json.

| condition | kinds 1-3 (95% CI) | k1 | k2 | k3 | all 4 kinds | bytes sent | sender s | Mac to Spark e2e s |
|---|---|---|---|---|---|---|---|---|
| kv_full = kv_q8 | 0.273 (0.21-0.35) | 21/50 | 9/50 | 11/50 | 0.450 | 127 MB / 64 MB | 0.22 | 11.0 / 5.9 |
| text_full | 0.293 (0.23-0.37) | 23/50 | 9/50 | 12/50 | 0.465 | 3 KB | 0 | 0.59 |
| latentmas | 0.387 (0.31-0.47) | 23/50 | 21/50 | 14/50 | 0.535 | 135 MB | 2.16 | 13.6 |
| kv_goal | 0.420 (0.34-0.50) | 26/50 | 25/50 | 12/50 | 0.560 | 132 MB | 0.24 | 11.4 |
| text_card | **0.647 (0.57-0.72)** | 33/50 | 34/50 | 30/50 | 0.735 | 207 B | 5.17 | 5.5 |

Reading: the pilot's ordering holds and is now outside the noise. A written task card, where the operator works
the answer out before handing off, beats every latent handoff (0.65 against at most 0.42, CIs do not overlap), and
it is 207 bytes instead of 130 MB. Handing over the raw KV cache is the same as handing over the raw text (0.965
agreement), so it buys nothing but transfer time. Giving the operator the goal before prefill (kv_goal) or letting
it think in latent space (latentmas) helps the value kind (k2) but not the arithmetic kind (k3). Decision for the
link: the operator sends the coder a text task card (as forge.duo already does); latent handoff is not worth it on
this hardware or across machines. Caveat as before: the text_card sender saw the question, so part of its lead is
that it answers the question itself; kv_goal is the fair "knows the goal" comparison and still loses.

## 2026-11-15 · latent hard-task pilot (--level hard) · Qwen3-8B, Spark GB10, bf16, n=40, short context

Multi-hop tasks with decoys (overridden constants, a legacy module with the same function names, handled errors
before the crash). 10 tasks per kind: 0 crashing function, 1 violated constant, 2 its effective value, 3 how far over.

| condition | accuracy (95% CI) | k0 | k1 | k2 | k3 | sender s | receiver TTFT s |
|---|---|---|---|---|---|---|---|
| none | 0.000 (0.00-0.09) | 0/10 | 0/10 | 0/10 | 0/10 | 0 | 0.08 |
| text_full | 0.525 (0.38-0.67) | 10/10 | 4/10 | 4/10 | 3/10 | 0 | 0.24 |
| kv_full = kv_q8 | 0.525 (0.38-0.67) | 10/10 | 4/10 | 4/10 | 3/10 | 0.23 | 0.09 |
| kv_goal | 0.575 (0.42-0.71) | 10/10 | 3/10 | 6/10 | 4/10 | 0.24 | 0.09 |
| latentmas | 0.600 (0.45-0.74) | 10/10 | 4/10 | 6/10 | 4/10 | 2.45 | 0.09 |
| text_card | 0.675 (0.52-0.80) | 10/10 | 3/10 | 7/10 | 7/10 | 4.94 | 0.09 |

Reading: productive. The tasks leave the ceiling and the floor, and an ordering appears: the more the operator
computes before handing off (goal-aware prefill < latent thoughts < a written note), the better the coder does on
the value and arithmetic kinds. Writing the note lets the operator work the answer out. n=40 is too small to call
(CIs overlap); kind 0 is still trivial. Next: n=200 on kinds 1-3 when the Spark is free (STATUS item D2).
