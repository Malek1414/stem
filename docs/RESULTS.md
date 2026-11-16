# agent-forge results log

## 2026-11-16 · queue4: operator and coder baselines · snapshot v0002 · harness 2

Operator: Holo4-35B-A3B-heretic Q4 (llama.cpp, Spark port 8081). Coder: llmfan46 Qwen3.6-35B-A3B-uncensored-heretic
Q4_K_M (port 8084). Sampling per model card (Qwen3.x: temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5),
max 4096 tokens per reply, seed 0. Judge for C stays greedy. Infrastructure errors are excluded and rerun later.

**Benchmark C** (20 held-out turns, judge scores 1-5, no replay of app/web tools, so the ceiling is about 3.2):

| Model | Graded | Infra | Understood | Approach | Correctness | Completeness | Overall (95% CI) | s/task |
|---|---|---|---|---|---|---|---|---|
| coder Qwen3.6-heretic | 19 | 1 | 2.74 | 2.16 | 2.05 | 1.74 | **2.26** (1.75-2.78) | 129 |
| operator Holo4-heretic | 20 | 0 | 2.50 | 1.70 | 1.80 | 1.65 | **1.90** (1.37-2.43) | 57 |
| earlier: qwen3-coder-next (v1 harness) | | | | | | | 1.63 | |
| null control (empty answer) | | | | | | | 1.00 | |

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
| coder Qwen3.6-heretic | rerun running (first run had no token cap: 11k+ tokens per item, stopped) | | | |

Reading D: Holo4 answers in its trained 0-1000 convention even when the prompt asks for pixels. Fixed in b99617d
(per-model convention, recorded in settings). The operator's computer-use tools must convert the same way.

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
