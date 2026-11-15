# agent-forge results log

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
