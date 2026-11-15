# Open-weight coder candidates for one DGX Spark (GB10, 128 GB, ~273 GB/s)

Research date: 2026-11-15. Budget: coder weights <= ~70 GB quantized, license must allow LoRA/SFT.

## Read this first: how far to trust the numbers

1. **Benchmarks changed during 2026.** Late-2026 model cards mostly stop reporting SWE-bench Verified and Terminal-Bench 2.0. They report **SWE-bench Pro**, **Terminal-Bench 2.1**, **DeepSWE v1.1**, NL2Repo, and Terminal-Bench 3.0/4.0 instead. swebench.com's own leaderboard JSON stops at 2026-02-26. So the 2025/early-2026 models (Verified, TB 2.0) cannot be compared directly with the Aug-Oct 2026 models (Pro, TB 2.1).
2. **Scaffold matters a lot.** One model, Qwen3-Coder-Next, gets 25.8 to 36.2 on TB 2.0 depending on the harness, and 38.7 vs 42.7 on SWE-bench Pro. Vendors also "correct problematic tasks" in SWE-bench Pro and re-run the baselines themselves. Treat every vendor number as an upper bound.
3. **Independent numbers are much lower.** gpt-oss-120b: 62.4 Verified (vendor) vs 26.0 on swebench.com bash-only. SWE-rebench (fresh tasks, decontaminated) scores Qwen3.6-27B at 31.2%, while Qwen reports 77.2 on Verified.
4. "V" = vendor-reported. "I" = independent (swebench.com bash-only/mini-SWE-agent, SWE-rebench, Scale SWE-bench Pro leaderboard, Artificial Analysis, a third party's uniform harness). "S" = secondary source (press or aggregator citing someone else). [UNVERIFIED] = seen only in a search snippet, not on a fetched page.
5. Sizes are decimal GB, summed from Hugging Face file listings (unsloth GGUF repos, official FP8/NVFP4 repos). The decode ceilings (tok/s) are my own arithmetic: 273 GB/s divided by the bytes of active weights read per token. They are upper bounds, not measurements.

---

## Summary table (models that fit, or nearly fit, the budget)

| Model | Total / active | Arch | Vision | License | Q4 GGUF | FP8 / NVFP4 | Key coding scores | Decode ceiling* |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.8-27B** (Aug 2026) | 27B dense | hybrid Gated DeltaNet + attn, MTP | yes | Apache-2.0 | 16.6 (UD-Q4_K_M) / 17.7 (UD-Q4_K_XL) | FP8 30.9 | SWE-Pro 61.7 V, TB2.1 73.0 V, LCB v6 90.3 V, AA Agentic 51 S | ~16 (Q4), ~9 (FP8), more with MTP |
| **Nemotron 3.5 Super VL 120B-A12B** (Oct 2026) | 120B / 12B | Mamba-2 + Transformer hybrid MoE, MTP | yes | OpenMDW-1.1 | 86.9 (UD-Q4_K_XL) | **NVFP4 75.0** (official, made for DGX Spark) | SWE-V 71.07, SWE-Pro 51.30, TB2.1 62.78, SWE-ML 67.0 (all NVFP4, V) | ~35 |
| **Qwen3.6-27B** (Apr 2026) | 27B dense | hybrid DeltaNet | yes | Apache-2.0 | 16.8 | FP8 30.9 | SWE-V 77.2 V, SWE-Pro 53.5 V, TB2.0 59.3 V; **SWE-rebench 31.2% I** | ~16 / ~9 |
| **Qwen3.6-35B-A3B** (Apr 2026) | 35B / 3B | hybrid DeltaNet MoE | yes | Apache-2.0 | 22.1 | FP8 37.5 | SWE-V 73.4 V, SWE-Pro 49.5 V, TB2.0 51.5 V; SWE-rebench 24.7% I; SWE-V 64.4 under KAT's harness (I-ish) | ~90+ |
| **KAT-Coder-V2.5-Dev** (Jul 2026) | 35B / 3B (Qwen3.6-35B-A3B fine-tune) | same | (base has vision) | Apache-2.0 | n/a (no unsloth repo) | BF16 69.3 | SWE-V 69.40, SWE-Pro 45.96, TB2.1 41.02 (uniform Claude Code harness, V) | ~90+ |
| **Muse Glimmer-30B** (Meta, Aug 2026) | 29.6B dense | dense transformer, local/global attn | yes | Apache-2.0 | 15.9 (UD-Q4_K_XL) | BF16 59.6 (Q8 29.6) | SWE-V 76.0 V, SWE-Pro 51.2 V, TB2.1 51.7 V | ~17 |
| **Qwen3-Coder-Next** (current) (Feb 2026) | 80B / 3B | hybrid DeltaNet MoE | no | Apache-2.0 | 48.5 | FP8 80.4 | SWE-V 70.6/71.1/71.3 V, SWE-Pro 42.7 V, TB2.0 36.2 V, Aider 66.2 V; SWE-rebench 54.4% (Feb window) S | ~90+ |
| Qwen3.5-27B (Feb 2026) | 27B dense | hybrid | yes | Apache-2.0 | 16.7 | — | SWE-V 72.4 V (75.0 in the 3.6 card), TB2 41.6 V; SWE-rebench: N/A in current window | ~16 |
| Qwen3.5-35B-A3B | 35B / 3B | hybrid MoE | yes | Apache-2.0 | ~22 | — | SWE-V 69.2 V, TB2 40.5 V; SWE-rebench 17.1% I | ~90+ |
| Qwen3.5-122B-A10B | 122B / 10B | hybrid MoE | yes | Apache-2.0 | 76.5 (Q4_K_M), MXFP4 74.7, Q3_K_M 56.4 | FP8 127.2 | SWE-V 72.0 V, TB2 49.4 V, BFCL-V4 72.2 V, TAU2 79.5 V | ~40 |
| Nemotron 3.5 Bolt 74B-A9B (Oct 2026) | 74B / 9B | Mamba-2 hybrid LatentMoE | yes | OpenMDW-1.1 | official native-2-bit "24GB" GGUF 22.0 | BF16 149.8 | TB2.1 53.93 (BF16) / 50.98 (IQ2) V | ~40+ |
| GLM-4.7-Flash (Jan 2026) | 30B / 3B | MoE | no | MIT | 18.3 | BF16 62.4 | SWE-V 59.2 V, tau2 79.5 V, LCB v6 64.0 V | ~90+ |
| gpt-oss-120b (Aug 2025) | 117B / 5.1B | MoE, MXFP4 native | no | Apache-2.0 | ~63 (MXFP4 native) | — | SWE-V 62.4 V / **26.0 I** (bash-only); SWE-Pro **16.2 I** (Scale); Aider 44.4 V / 41.8 I | ~60 |
| gpt-oss-20b | 21B / 3.6B | MoE | no | Apache-2.0 | ~11.6 | — | SWE-V 60.7 V; Aider 34.2 V | fast |
| Devstral Small 2 24B (Dec 2025) | 24B dense | Mistral3 | yes | Apache-2.0 | 14.3 | FP8 repo | SWE-V 68.0 V / **56.4 I** (bash-only); TB2 22.5 V | ~19 |
| Mistral Small 4 119B-A6.5B (2026) | 119B / 6.5B | MoE | yes | Apache-2.0 | — | NVFP4 70.8 | no SWE/TB numbers in text on the card | ~50 |
| Qwen3-Coder-30B-A3B (Jul 2025) | 30.5B / 3.3B | MoE | no | Apache-2.0 | 18.6 | — | SWE-V 51.6 I (OpenHands, swebench.com); 31.8 under KAT's Claude Code harness | ~90+ |
| GLM-4.5-Air (Jul 2025) | 106B / 12B | MoE | no | MIT | 73.0 (Q4_K_M), 67.7 (UD-Q4_K_XL) | — | SWE-V 57.6 V | ~35 |
| Seed-OSS-36B (Aug 2025) | 36B dense | dense GQA | no | Apache-2.0 | [not checked] | — | SWE-V 56 V (OpenHands), LCB v6 67.4 V | ~13 |
| Kimi-Dev-72B (Jun 2025) | 72B dense | Qwen2.5-72B fine-tune | no | MIT | [not checked] ~44 est. | — | SWE-V 60.4 V | ~6 |

*Decode ceiling = my arithmetic (273 GB/s / active-weight bytes at Q4 or NVFP4). Not measured. Real throughput is usually 50-70% of this, less with long context, more with MTP/speculative decoding.

### Over budget (do not fit <= 70 GB; listed so you know why they are out)

| Model | Size | Why it is out | Notable scores |
|---|---|---|---|
| Qwen3.8-Flash-Next (Aug 2026) | 125B/6B + 51B n-gram embedding + 4B MTP = 180B stored | FP8 repo 185 GB; UD-Q4_K_XL 111.3 GB, UD-Q2_K_XL 79.6 GB. License qwen-community-1.0 (fine-tuning allowed, but you need a separate license for commercial "MaaS / AI Work Assistant" coding products) | SWE-Pro 62.5 V, SWE-ML 81.0 V (mini-SWE-agent), DeepSWE 58.7 V |
| Mistral Medium 3.5 128B (Mar 2026) | dense 128B | Q4 ~75 GB, and dense 128B decodes at ~3-4 tok/s on Spark. Modified MIT (revenue cap) | SWE-V 77.6 V |
| Devstral 2 123B (Dec 2025) | dense 123B | Q4_K_M 74.9 GB, ~3-4 tok/s | SWE-V 72.2 V / 53.8 I (bash-only), TB2 32.6 V |
| Step-3.7-Flash (May 2026) | 198B / 11B | UD-Q4_K_XL 122 GB, UD-Q2_K_XL 65.8 GB | SWE-Pro 56.3 V, TB2.1 59.5 V |
| MiniMax-M2.5 (Feb 2026) | 230B / 10B | Q3_K_M 109.3 GB, UD-Q2_K_XL 85.9 GB, Q4_K_M 138.3 GB. **Even Q3 needs ~110 GB, so it does not fit next to an operator** | SWE-V **75.8 I** (bash-only, high), SWE-ML 68.3 I, SWE-rebench 54.6% S |
| MiniMax-M3 / M3.1-Flash | M3.1-Flash ~428B / 23B | far too big; license "minimax-community" | — |
| GLM-5.3-Flash (Aug 2026) | 320B / 18B | Q4 ~180 GB | TB2.1 84.3 V (Claude Code), DeepSWE 63.4 V |
| MiMo-V2.6-Flash (Sep 2026) | 309B / 15B | Q4 ~175 GB | TB2.1 87.6 V, DeepSWE 67.9 V |
| DeepSeek-V4.1-Flash (Sep 2026) | 552B backbone + 196B Engram | far too big | TB2.1 90.6 V, DeepSWE 74.2 V |
| Ling-3.1-flash-1019 | 560B / 25B | far too big | AA Intelligence v5: 30 |

---

## Per-model detail

### 1. Qwen3-Coder-Next (your current candidate)
- Repo: `Qwen/Qwen3-Coder-Next` (created 2026-01-30). Apache-2.0. 79.67B params total, ~3B active. Hybrid Gated DeltaNet + attention MoE (built on Qwen3-Next). Text only.
- Sizes: Q4_K_M 48.5 GB, UD-Q4_K_XL 49.6 GB, MXFP4_MOE 48.0 GB, IQ4_XS 42.7 GB; official FP8 80.4 GB.
- Scores (vendor tech report, arXiv 2603.00729, max 300 turns):

| Benchmark | Score | Scaffold | Type | Source |
|---|---|---|---|---|
| SWE-bench Verified | 70.6 | SWE-agent | V | https://arxiv.org/html/2603.00729v1 |
| SWE-bench Verified | 71.1 | mini-SWE-agent | V | https://arxiv.org/html/2603.00729v1 |
| SWE-bench Verified | 71.3 | OpenHands | V | https://arxiv.org/html/2603.00729v1 |
| SWE-bench Multilingual | 62.8 / 56.2 / 64.3 | SWE-agent / mini / OpenHands | V | https://arxiv.org/html/2603.00729v1 |
| SWE-bench Pro | 42.7 / 38.7 | SWE-agent / mini-SWE-agent | V | https://arxiv.org/html/2603.00729v1 |
| Terminal-Bench 2.0 | 34.2 / 36.2 / 30.9 / 25.8 | Terminus2-xml / Terminus2-json / Claude Code / Qwen Code | V | https://arxiv.org/html/2603.00729v1 |
| Aider-Polyglot | 66.2 | — | V | https://arxiv.org/html/2603.00729v1 |
| LiveCodeBench v6 | 58.93 | — | V | https://arxiv.org/html/2603.00729v1 |
| SWE-rebench (Feb 2026 window, 57 tasks) | 54.4% | SWE-rebench standard | I, via S | https://ai-radar.it/article/swe-rebench-gpt-5-4-qwen3-5-e-gemini-3-1-pro-a-confronto |
| AA Terminal-Bench Hard 18.2%, Coding Index 36.2 (or 22.9), TB2.1 38.2% | — | AA | [UNVERIFIED]: search snippet only; the fetched AA page shows N/A ("No publicly available API") | https://artificialanalysis.ai/models/qwen3-coder-next |

- Same table, peers (vendor-run by Qwen): GLM-4.7 74.2 / MiniMax-M2.1 74.8 / DeepSeek-V3.2 70.2 SWE-V (SWE-agent); TB2.0 best: GLM-4.7 44.9, MiniMax-M2.1 42.7 (Claude Code), Kimi-K2.5 49.4; Claude Opus 4.5 58.4.
- **Possible cause of your 0/7 (worth checking before you drop the model):** llama.cpp issue #20164 reports Qwen3-Coder-Next and Qwen3.5-35B-A3B looping on repeated, almost-correct tool calls when a tool has several optional parameters (around 30k tokens in, e.g. OpenCode `read` with offset/limit). Fixed by PR #20171 (autoparser). The reporter also made it go away by making the optional parameter required. Community reports also cite parser/template mismatches (vLLM `qwen3_coder` XML parser vs. Hermes JSON on llama.cpp; `<`/`>` in code arguments breaking the regex parser). Sources: https://github.com/ggml-org/llama.cpp/issues/20164 , https://huggingface.co/SergioPatricio/qwen3-coder-next-llamacpp-template , https://allanchan339.github.io/bug-fixes/2026/04/13/Qwen35-tool-calling.html
- Verdict: three newer Qwen releases beat it at a smaller memory footprint (see below). Its best independent number (SWE-rebench 54.4%, Feb) is respectable, but SWE-rebench has dropped it from the current window.

### 2. Qwen3.8-27B (strongest in-budget model on vendor numbers)
- Repo: `Qwen/Qwen3.8-27B` (created 2026-08-05), `Qwen/Qwen3.8-27B-FP8` (2026-08-13). Apache-2.0. 27B dense, hybrid (16 x [3 x Gated DeltaNet + 1 x Gated Attention]), MTP head, vision encoder, 262K native context (1M with YaRN). Thinking on by default with `reasoning_effort`.
- Sizes: UD-Q4_K_M 16.6 GB, UD-Q4_K_XL 17.7 GB, UD-Q6_K 22.0 GB, Q8_0 29.0 GB, FP8 30.9 GB, BF16 55.6 GB.
- Scores:

| Benchmark | Qwen3.8-27B | Qwen3.6-27B | Opus 4.6 Max (in Qwen's table) | Type | Source |
|---|---|---|---|---|---|
| Terminal-Bench 2.1 (Terminus) | 73.0 | 63.4 | 78.2 | V | https://huggingface.co/Qwen/Qwen3.8-27B |
| SWE-bench Pro (Claude Code harness, refined task set) | 61.7 | 53.5 | 53.4 | V | https://huggingface.co/Qwen/Qwen3.8-27B |
| SWE-bench Multilingual (mini-SWE-agent) | 73.8 | — | 77.5 | V | https://huggingface.co/Qwen/Qwen3.8-Flash-Next |
| DeepSWE 1.1 | 42.2 | 13.3 | — | V | https://huggingface.co/Qwen/Qwen3.8-27B |
| NL2Repo-Bench | 42.3 | 36.2 | 47.6 | V | https://huggingface.co/Qwen/Qwen3.8-27B |
| LiveCodeBench v6 | 90.3 | 83.9 | 88.8 | V | https://huggingface.co/Qwen/Qwen3.8-27B |
| Toolathlon Verified | 67.1 | — | — | V | https://huggingface.co/Qwen/Qwen3.8-Flash-Next |
| AA Intelligence Index (older version) xhigh / medium | 52 / 44; Agentic Index 51 | Qwen3.6-27B: 38 | — | I via S | https://www.qubrid.com/blog/qwen38-27b-benchmarks-official-and-independent-results |
| AA Intelligence Index v5 (current) | 20 (Nemotron 3.5 Super: 14) | — | — | I | https://artificialanalysis.ai/models/open-source |
| SWE-bench Verified | not reported by Qwen | — | — | — | — |

- Field reports (S): Simon Willison called it the best local model he has tested, but it "defaults to wildly overthinking" (22k reasoning tokens / 21 min for one SVG). Cline's team: "first time a local model has scored frontier model capability". It has a large KV cache per token. https://www.developersdigest.tech/blog/qwen-3-8-27b-local-agentic-coding-2026
- Caveat for Spark: dense 27B means every token reads the full weights, so expect roughly 10-15 tok/s at Q4/NVFP4 before MTP speculation. Not on SWE-rebench yet (current window ends 2026-07-01).

### 3. Qwen3.6-27B
- Repo `Qwen/Qwen3.6-27B` (2026-04-21) + FP8. Apache-2.0. Dense hybrid, vision. Q4_K_M 16.8 GB, FP8 30.9 GB.

| Benchmark | Score | Scaffold | Type | Source |
|---|---|---|---|---|
| SWE-bench Verified | 77.2 | internal (bash + file-edit), 200K | V | https://huggingface.co/Qwen/Qwen3.6-27B |
| SWE-bench Pro | 53.5 | internal, refined set | V | same |
| SWE-bench Multilingual | 71.3 | internal | V | same |
| Terminal-Bench 2.0 | 59.3 | Harbor/Terminus-2, 3h, avg of 5 | V | same |
| SkillsBench / NL2Repo / LCB v6 | 48.2 / 36.2 / 83.9 | OpenCode / Claude Code / — | V | same |
| SWE-bench Verified | 77.2 | (as listed by Meta) | S | https://huggingface.co/meta-models/Muse-Glimmer-30B |
| TB2.1 (terminus2) | 60.7 | Meta's run | I-ish (competitor-run) | https://huggingface.co/meta-models/Muse-Glimmer-30B |
| **SWE-rebench (15 May to 1 Jul 2026, 111 tasks)** | **31.2% ±1.68, pass@5 57.7%** | SWE-rebench | **I** | https://swe-rebench.com/ |

### 4. Qwen3.6-35B-A3B (fast MoE option, same family as Coder-Next but newer)
- Repo `Qwen/Qwen3.6-35B-A3B` (2026-04-15) + FP8. Apache-2.0. 35B/3B, 256 experts (8+1), hybrid DeltaNet, vision. UD-Q4_K_M 22.1 GB, MXFP4_MOE 21.7 GB, FP8 37.5 GB.

| Benchmark | Score | Scaffold | Type | Source |
|---|---|---|---|---|
| SWE-bench Verified | 73.4 | internal | V | https://huggingface.co/Qwen/Qwen3.6-35B-A3B |
| SWE-bench Pro | 49.5 | internal | V | same |
| SWE-bench Multilingual | 67.2 | internal | V | same |
| Terminal-Bench 2.0 | 51.5 | Terminus-2 | V | same |
| TAU3-Bench / MCPMark / LCB v6 | 67.2 / 37.0 / 80.4 | — | V | same |
| SWE-V / SWE-Pro / TB2.1 under KAT's uniform Claude Code harness | 64.4 / 40.63 / 32.02 | claude_code@2.1.195 | competitor-run | https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev |
| TB2.1 (NVIDIA's run) | 44.38 | NVIDIA harness | competitor-run | https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 |
| SWE-rebench (May-Jul 2026) | 24.7% ±0.79, pass@5 43.2% | SWE-rebench | I | https://swe-rebench.com/ |

### 5. KAT-Coder-V2.5-Dev (Kwaipilot)
- Repo `Kwaipilot/KAT-Coder-V2.5-Dev` (2026-07-23). Apache-2.0. SFT (127K examples) + RL on top of Qwen3.6-35B-A3B. BF16 69.3 GB (any Qwen3.6-35B-A3B quant recipe applies, so Q4 should be ~22 GB [estimate]).
- All models run in one in-house pipeline (Claude Code 2.1.195, 256K): SWE-V 69.40 (Qwen3.5-27B 68.60, Qwen3.6-35B-A3B 64.40, Gemma4-31B 60.60, Qwen3-Coder-30B 31.80); SWE-Pro 45.96; TB2.1 41.02 (average of Terminus-2 and Claude Code). V. https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev
- Why it matters: it shows an agentic-coding SFT+RL fine-tune of the 3B-active base gains about +5 SWE-V / +5 Pro / +9 TB2.1. That is the same kind of fine-tune you plan to do.

### 6. NVIDIA Nemotron 3.5 Super VL 120B-A12B
- Repos: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (HF release 10/15/2026), BF16 and Base-BF16 also published (the Base checkpoint is useful for fine-tuning). License OpenMDW-1.1 (Linux Foundation, permissive, no obligation to publish fine-tunes). Mamba-2/Transformer hybrid MoE, 512 experts per layer, MTP, C-RADIO vision, 1M context.
- NVFP4 weights 75.0 GB (decimal). The card's vLLM recipe "is intended for DGX Spark (GB10) only" and recommends the LiLiCorr DFlash drafter for Spark. UD-Q4_K_XL GGUF 86.9 GB, UD-Q2_K_XL 57.8 GB.

| Benchmark | BF16 | NVFP4 | Type | Source |
|---|---|---|---|---|
| SWE-bench Verified | 71.10 | 71.07 | V | https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 |
| SWE-bench Multilingual | 67.72 | 67.00 | V | same |
| SWE-bench Pro | 51.78 | 51.30 | V | same |
| Terminal-Bench 2.1 | 60.67 | 62.78 | V | same |
| LiveCodeBench v6 | 88.22 | 87.41 | V | same |
| tau2 retail / airline / telecom | 69.24 / 76.50 / 73.19 | 69.30 / 75.75 / 73.19 | V | same |
| DeepSWE / Terminal-Bench 4 | — | 18.2* / 1.5* | V (eval module) | same |
| AA Intelligence Index v5 | 14 | | I | https://artificialanalysis.ai/models/open-source |

- Scaffold for SWE/TB is not stated in the card text (only a harness-comparison image).
- Note: at 75 GB it sits slightly above the 70 GB target. With an operator model you have about 50 GB left minus KV cache and the OS.

### 7. NVIDIA Nemotron 3.5 Bolt 74B-A9B and Lightning 30B-A3B
- Bolt: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16` (Oct 2026). OpenMDW-1.1. The BF16 card says it is meant for "customization and post-training", which fits a LoRA plan. Official native-2-bit "24GB" GGUF is 22.0 GB. TB2.1 53.93 (BF16) / 50.98 (24GB IQ2) vs Qwen3.6-35B-A3B 44.38 (NVIDIA-run). V. https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 . No SWE-bench numbers on the card.
- Lightning 30B-A3B (Aug 2026): SWE-V 51.56, SWE-ML 39.33, TB2.1 24.58 (BF16). V. Too weak for the coder role. https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

### 8. Muse Glimmer-30B (Meta Superintelligence Labs)
- Repo `meta-models/Muse-Glimmer-30B` (created 2026-08-09). Apache-2.0. 29.6B dense (incl. 1.8B frozen perception encoder), 52 layers, 3:1 local:global attention, 131K context. UD-Q4_K_XL 15.9 GB, Q8_0 29.6 GB. Fine-tuning supported via TorchTitan (S: https://infoq.com/news/2026/08/meta-muse-glimmer).

| Benchmark | Glimmer-30B | Qwen3.6-27B | Gemma4-31B | Type | Source |
|---|---|---|---|---|---|
| SWE-bench Verified | 76.0 | 77.2 | 66.6 | V | https://huggingface.co/meta-models/Muse-Glimmer-30B |
| SWE-bench Pro | 51.2 | 50.2 | 36.9 | V | same |
| Terminal-Bench 2.1 (terminus2) | 51.7 | 60.7 | 43.4 | V | same |
| MCP-Atlas | 75.5 | 62.5 | 54.2 | V | same |

### 9. Qwen3.5 medium family (Feb 2026)
- `Qwen/Qwen3.5-27B`, `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.5-122B-A10B`. Apache-2.0, hybrid DeltaNet, vision.

| Benchmark | 122B-A10B | 27B | 35B-A3B | GPT-OSS-120B (in same table) | Type | Source |
|---|---|---|---|---|---|---|
| SWE-bench Verified | 72.0 | 72.4 | 69.2 | 62.0 | V | https://huggingface.co/Qwen/Qwen3.5-122B-A10B |
| Terminal Bench 2 | 49.4 | 41.6 | 40.5 | 18.7 | V | same |
| LiveCodeBench v6 | 78.9 | 80.7 | 74.6 | 82.7 | V | same |
| BFCL-V4 | 72.2 | 68.5 | 67.3 | — | V | same |
| TAU2-Bench | 79.5 | 79.0 | 81.2 | — | V | same |
| SWE-rebench Feb 2026 (Qwen3.5-397B for reference: 59.9%) | | | | | S | https://ai-radar.it/article/swe-rebench-gpt-5-4-qwen3-5-e-gemini-3-1-pro-a-confronto |
| SWE-rebench May-Jul 2026 | — | N/A | 17.1% | | I | https://swe-rebench.com/ |

- Note: the Qwen3.6 cards list Qwen3.5-27B at SWE-V 75.0 and Qwen3.5-35B-A3B at 70.0, so the same vendor re-scored its own model differently in a later card.
- 122B-A10B sizes: Q4_K_M 76.5, MXFP4_MOE 74.7, Q3_K_M 56.4, FP8 127.2 GB. Over budget at Q4.

### 10. GLM-4.7-Flash (Z.ai)
- `zai-org/GLM-4.7-Flash` (2026-01-19). MIT. 30B-A3B MoE. Q4_K_M 18.3 GB.
- SWE-V 59.2, tau2 79.5, LCB v6 64.0, vs gpt-oss-20b 34.0 SWE-V (Z.ai's figure for gpt-oss-20b, which is much lower than OpenAI's own 60.7). V. https://huggingface.co/zai-org/GLM-4.7-Flash
- Z.ai's newer releases (GLM-5.x, GLM-5.3-Flash 320B-A18B) are all far over budget. GLM-5.2 scores 62.9% on SWE-rebench (I), the best open model there, but it is huge.

### 11. gpt-oss-120b / gpt-oss-20b (OpenAI, Aug 2025)
- Apache-2.0. 117B/5.1B and 21B/3.6B, MXFP4-native (~63 GB / ~12 GB GGUF).

| Benchmark | 120b | 20b | Type | Source |
|---|---|---|---|---|
| SWE-bench Verified (high) | 62.4 | 60.7 | V | https://docs.api.nvidia.com/nim/reference/openai-gpt-oss-120b |
| Aider Polyglot | 44.4 | 34.2 | V | same |
| Tau-Bench Retail / Airline | 67.8 / 49.2 | 54.4 / 38.0 | V | same |
| SWE-bench Verified bash-only (mini-SWE-agent 1.7.0) | **26.0** | — | I | https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json |
| SWE-bench Pro (Scale public) | **16.20 ±2.67** | — | I | https://labs.scale.com/leaderboard/swe_bench_pro_public |
| Aider polyglot leaderboard (high) | 41.8 | — | I | https://aider.chat/docs/leaderboards/ |

- Verdict: large gap between vendor and independent scores. Not recommended as the coder.

### 12. Devstral 2 / Devstral Small 2 (Mistral, Dec 2025)
- `mistralai/Devstral-Small-2-24B-Instruct-2512` (Apache-2.0, 24B dense, vision, 256K) and `mistralai/Devstral-2-123B-Instruct-2512`.

| Benchmark | Devstral 2 (123B) | Small 2 (24B) | Type | Source |
|---|---|---|---|---|
| SWE-bench Verified | 72.2 | 68.0 | V | https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512 |
| SWE-bench Multilingual | 61.3 | 55.7 | V | same |
| Terminal-Bench 2 | 32.6 | 22.5 | V | same |
| SWE-bench Verified bash-only | 53.8 | 56.4 | I | https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json |

- Mistral says Mistral Medium 3.5 128B (dense, SWE-V 77.6 V) now "supersedes" Devstral, but 128B dense is too slow on a Spark. https://huggingface.co/mistralai/Mistral-Medium-3.5-128B

### 13. Older / smaller coders (reference)
- Qwen3-Coder-30B-A3B-Instruct: SWE-V 51.6 (OpenHands, swebench.com, I). https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json
- GLM-4.5-Air (106B-A12B, MIT): SWE-V 57.6 V. https://huggingface.co/zai-org/GLM-4.5-Air
- Seed-OSS-36B-Instruct (Apache-2.0): SWE-V 56 (OpenHands) / 47 (Agentless), LCB v6 67.4, TAU1-Retail 70.4. V. https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct
- Kimi-Dev-72B (MIT, Qwen2.5-72B fine-tune): SWE-V 60.4 V. https://huggingface.co/moonshotai/Kimi-Dev-72B
- KAT-Dev-32B / KAT-Dev-72B-Exp (Oct 2025): replaced by KAT-Coder-V2.5-Dev; not re-checked.
- Qwen-AgentWorld-35B-A3B: an environment/world-simulator model, not a coder. https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B
- Not checked in depth: IBM Granite, Meituan LongCat (flagship 560B, too big), SWE-smith/SWE-Dev/OpenHands-LM (2025, superseded).

---

## Independent leaderboards: state as of 2026-11-15

- **SWE-rebench** (https://swe-rebench.com/): current window 15/05/2026 to 01/07/2026, 111 tasks / 65 repos. Open models: GLM-5.2 62.9%, MiniMax M3 47.2%, MiMo V2.5 Pro 46.5%, DeepSeek-V4 Pro 40.2%, **Qwen3.6-27B 31.2%**, Qwen3.6-35B-A3B 24.7%, Qwen3.5-35B-A3B 17.1%. Closed: Fable 5 64.5%, Grok 4.5 63.8%, Opus 5 63.4%, GPT-5.6 Sol 62.3%, Sonnet 5 56.8%. Qwen3-Coder-Next, Devstral, gpt-oss, Nemotron, and GLM-4.7-Flash show N/A in this window. February 2026 window (S, ai-radar.it): Claude Opus 4.6 65.3, GLM-5 62.8, DeepSeek-V3.2 60.9, Qwen3.5-397B 59.9, Step-3.5-Flash 59.6, MiniMax M2.5 54.6, **Qwen3-Coder-Next 54.4**.
- **swebench.com bash-only (mini-SWE-agent)**: last entry 2026-02-26. Open models: MiniMax M2.5 (high) 75.8, GLM 5 (high) 72.8, Kimi K2.5 70.8, DeepSeek V3.2 70.0, Devstral Small 2512 56.4, Devstral 2512 53.8, gpt-oss-120b 26.0. Closed: Claude 4.5 Opus (high) 76.8, Gemini 3 Flash 75.8, Claude 4.6 Opus 75.6. Multilingual board: GLM 5 69.7, MiniMax M2.5 68.3.
- **Scale SWE-bench Pro public** (https://labs.scale.com/leaderboard/swe_bench_pro_public): Muse Spark 1.1 61.5, gpt-5.4 xHigh 59.1, Opus 4.6 51.9. Open: qwen3-coder-480b 38.7, minimax-2.1 36.8, gpt-oss-120b 16.2, glm-4.6 9.7. Does not include the newer models.
- **tbench.ai**: now shows Terminal-Bench 4.1 by default. Rows load client-side and could not be fetched, so no independent TB 2.0/2.1 rows here.
- **Aider polyglot**: last updated 2025-11-20. Stale for 2026 models.
- **Artificial Analysis**: now on Intelligence Index v5 (TB 4.0, FrontierCode-AA, etc.). Open models under 130B: Qwen3.8-27B (xhigh) 20, Nemotron 3.5 Super VL 14. No v5 score for Qwen3-Coder-Next ("No publicly available API").

---

## Frontier reference (closed models, Nov 2026)

| Model | SWE-bench Verified | SWE-bench Pro | Terminal-Bench 2.1 | Other | Source / type |
|---|---|---|---|---|---|
| Claude Fable 5 (Jun 9 2026) | 95.0 (V; Vals AI run also 95.0) | 80.3 (V) | 84.3 V / 80.5 Vals / 84.6 AA | SWE-rebench 64.5% I | https://the-decoder.com/anthropic-releases-claude-fable-5-and-mythos-5-with-major-gains-in-coding-and-science/ (Pro); https://benchlm.ai/md/models/claude-fable.md (S aggregator: Verified, TB2.1); https://swe-rebench.com/ |
| Claude Opus 5 | — | — | 89.1 (V, in Google table) | DeepSWE v1.1 74.0; SWE-rebench 63.4% I | https://www.vellum.ai/md/blog/gemini-3-8-flash-benchmarks-explained (S); https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash (TB2.1 89.1, DeepSWE 74.0 in DeepSeek's table) |
| GPT-5.6 Sol (Jul 2026) | — ("coming soon" on aggregator) | 64.6 (S) | 88.8 standard / 91.9 ultra (S); 85.8 Vals | DeepSWE 72.7; SWE-rebench 62.3% I | https://benchlm.ai/compare/claude-fable-5-vs-gpt-5-6-sol (S); https://swe-rebench.com/ |
| Gemini 3.8 Flash (Sep 2 2026) | — | — | 89.4 (V) | DeepSWE v1.1 73.7 | https://www.vellum.ai/md/blog/gemini-3-8-flash-benchmarks-explained (S) |
| Gemini 3.1 Pro (Feb 2026, latest Pro) | — | 54.2 (as quoted by Anthropic) | — | Scale SWE-Pro 46.1 (mini-SWE-agent, I) | https://the-decoder.com/anthropic-releases-claude-fable-5-and-mythos-5-with-major-gains-in-coding-and-science/ ; https://labs.scale.com/leaderboard/swe_bench_pro_public |
| Aider polyglot | not reported for 2026 frontier models (leaderboard frozen Nov 2025; best there gpt-5 high 88.0%) | | | | https://aider.chat/docs/leaderboards/ |

Caveat: official vendor pages (anthropic.com, openai.com) returned 403 to the fetcher, so the frontier numbers come from press and aggregators that cite the vendors. Treat them as S. Vendor frontier claims can disagree by a few points between sources (e.g. Fable 5 SWE-V 93.9 vs 95.0, SWE-Pro 80.0 vs 80.3).

---

## Caveats specific to the decision

1. Qwen's 2026 numbers come from an "internal agent scaffold" or the Claude Code harness on refined task sets. SWE-rebench puts Qwen3.6-27B about 31 points below Fable 5, while Qwen's Verified number suggests the gap is only about 4 points. Plan on your internal benchmark being the real decider.
2. Dense 27-30B models (Qwen3.8-27B, Qwen3.6-27B, Glimmer) are bandwidth-bound on GB10: roughly 10-16 tok/s at 4-bit before MTP/DFlash speculation. The 3B-active MoEs (Qwen3.6-35B-A3B, KAT-Coder-V2.5, Coder-Next) are 4-6x faster per token. Nemotron 3.5 Super (12B active, NVFP4, Spark-tuned recipe plus drafter) sits in between.
3. Licenses: Apache-2.0 (Qwen3.x open models, Glimmer, KAT, Devstral Small 2, gpt-oss), MIT (GLM), OpenMDW-1.1 (Nemotron; permissive, OSI review pending) all allow LoRA/SFT and redistribution of fine-tunes. qwen-community-1.0 (Qwen3.8-Flash-Next only) allows fine-tuning but requires a separate license for external "Model-as-a-Service / AI Work Assistant (coding)" products.
4. Your Coder-Next looping matches a known llama.cpp tool-call parsing bug (#20164, fixed in PR #20171) and known qwen3_coder parser fragility. Test the replacement candidates on the same, updated serving stack. Otherwise the replacement may "fail" for the same infrastructure reason.
