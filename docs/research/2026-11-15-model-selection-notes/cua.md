# Open-weight computer-use / GUI-agent models for a DGX Spark operator (research as of 2026-11-15)

Scope: open-weight GUI / computer-use specialist models and the general VLMs used as their bases. Target: a low-latency macOS operator (screenshot to click/type plus tool calls) on one DGX Spark (GB10, 128 GB, ~273 GB/s). It must be uncensored (an abliterated variant exists or abliteration is feasible), the license must allow fine-tuning, and the quantized weights should ideally be 40 GB or less.

## How to read this file

- Every number has a source key such as `[OSW-V]`. The keys resolve to URLs in the source key table at the end.
- **LB-verified** means the number is in the official OSWorld-Verified spreadsheet (`osworld_verified_results.xlsx`), which the OSWorld team runs or verifies. **Vendor** means a model card, blog or paper by the model's authors. **Aggregator** means a third-party leaderboard site (llm-stats, benchlm). **[UNVERIFIED]** means I could not confirm the number on a primary page I actually saw.
- The OSWorld step budget is given wherever the source states it. Many 2026 vendor numbers (Holo, Qwen, Nex) come from the vendor's own harness, often with large step budgets. H Company's Holo4 harness used up to 300 or 500 steps for OSWorld 2.0. Vendor numbers are therefore not directly comparable to LB 15/50/100-step rows.
- **Single model vs. framework:** every row in this file is a single end-to-end model unless it is tagged **[framework]**. A framework row means a planner (usually a big closed model) plus a grounder (for example "GTA1 w/ GPT-5" or "Agent S3 w/ Opus 4.5"). Those rows do not represent what the open model can do alone.
- Size estimates marked "est." are computed by me, not sourced: Q4_K_M ≈ 0.6 bytes per parameter and FP8 ≈ 1.05 bytes per parameter. All other sizes are actual file sizes from the HF API tree listing.
- **Decode-speed intuition (my estimate, not measured).** At 273 GB/s, a ~3B-active MoE at Q4 reads about 1.7–2 GB per token, so the upper bound is about 120–150 tok/s. A 27B dense model at Q4 reads about 17 GB per token, so the upper bound is about 15 tok/s. A 9B dense model at Q4 tops out around 45 tok/s. Prefill of screenshots (image tokens) also matters and is compute-bound on GB10.

---

## TL;DR ranking for this use case

| # | Model | Arch / active | License | Q4 / FP8 size | OSWorld-Verified | Abliterated on HF |
|---|---|---|---|---|---|---|
| 1 | **Holo4-35B-A3B** (H Company, 2026-09-24) | Qwen3.6 MoE, 35.1B total / ~3B active | Apache-2.0 | 21.3 GB Q4 GGUF (+0.9 GB mmproj); 37.6 GB FP8; 22.5 GB NVFP4 | 80.8% vendor (own harness) | Yes, Heretic GGUF only (Q4_K_M, Q6_K, GSQ-RCO quants) |
| 2 | **Holo3-35B-A3B** (2026-03-23) | Qwen3.5 MoE 35B / 3B | Apache-2.0 | ~21 GB Q4 est.; FP8 ~37 GB est. | **82.56% LB-verified (100 steps)** (a second LB run gave 78.15%); 77.8% vendor | No direct one. A Holo-3.1 Heretic exists, and the Mantis LoRA fine-tune is on HF |
| 3 | **Holo-3.1-35B-A3B** (2026-05-29) | Qwen3.6 MoE 35B / 3B | Apache-2.0 | 21.3 GB Q4; 37.6 GB FP8 | 80.0% vendor | **Yes, full weights** (darkc0de Heretic, F32) plus GGUF (mradermacher) |
| 4 | **Qwen3.8-27B** (base of Holo4-27B) | dense 27B | Apache-2.0 | ~17 GB Q4 est.; FP8 repo exists | 84.3% vendor | Yes, many (for example 0bserverx Heretic, full weights) |
| 5 | **Mage-VL-Cowork** (Microsoft, 2026-10-30) | dense 4B (llava_onevision2) | Apache-2.0 | 9.48 GB BF16 (so ~3 GB Q4 est.) | 66.0% vendor (50 steps) | No. Abliteration is feasible (standard transformer) |
| (alt) | Holo4-27B | dense 27B (Qwen3.8) | **CC-BY-NC-4.0** | 16.9 GB Q4; 35.9 GB FP8 | 85.2% vendor | Yes, exl3 only (groxaxo) |
| (alt) | UI-Venus-2-9B | dense 9B (Qwen3.5-9B) | **license pending** per its own card | ~5.6 GB Q4 est. | 70.8% vendor | No |
| (alt) | Nex-N2.5-mini | Qwen3.5-MoE 35B / ~3B | Apache-2.0 | ~21 GB Q4 est. | 71.2% vendor | No |
| (alt) | CUA-Master-27B | dense 27B (Qwen3.5-27B) | **no license in card** | ~17 GB Q4 est. | 80.2% vendor; **MacOSWorld 43.0, MacAgentBench 59.9** | No |

---

## 1. H Company: Holo family

### 1a. Holo4-35B-A3B (top pick)

- HF: https://huggingface.co/Hcompany/Holo4-35B-A3B (also `-FP8`, `-NVFP4`, `-GGUF`)
- Released 2026-09-24 on HF; blog date 2026-09-28 `[HOLO4-BLOG]`.
- Base: Qwen/Qwen3.6-35B-A3B. Architecture: Qwen3.6 MoE, 35,107,181,936 parameters (HF safetensors metadata), about 3B active. Context 262,144 `[HOLO4-35]`.
- License: Apache-2.0 `[HOLO4-35]`.
- Sizes from the HF API: BF16 70.2 GB; FP8 37.6 GB; NVFP4 22.5 GB; GGUF `q4_k_m.gguf` 21.3 GB plus `mmproj.f16.gguf` 0.9 GB.
- Harness: hai-agents (screenshots plus tool calls, code execution, function calling) `[HOLO4-35]`. Holo4 includes native function calling and MCP tool use.
- Abliterated / uncensored variants (all GGUF; no full-precision abliterated safetensors found):
  - https://huggingface.co/jduraes/Holo4-35B-A3B-heretic-Q4_K_M (21.2 GB + mmproj)
  - https://huggingface.co/jduraes/Holo4-35B-A3B-heretic-Q6_K (28.5 GB)
  - https://huggingface.co/lavendercomputing/Holo4-35B-A3B-Heretic-GSQ-RCO-GGUF (IQ2–IQ4_XS, 10–17 GB, plus an MTP draft head)
  - For fine-tuning you would run Heretic or abliteration yourself on the BF16 weights. This is feasible: it is the standard Qwen3.6 MoE, which Heretic already supports.

| Benchmark | Score | Type | Source |
|---|---|---|---|
| OSWorld (H harness, mean of 2–4 runs) | 80.8% ($0.05/task) | vendor | `[HOLO4-TBL]`, `[HOLO4-BLOG]` |
| OSWorld 2.0 (score / success) | 30.9% / 12.3% | vendor, single run | `[HOLO4-TBL]` |
| AndroidWorld | 77.6% | vendor | `[HOLO4-TBL]` |
| AutomationBench (business API/MCP) | 34.5% (held-out 120 tasks: 31.7%) | vendor | `[HOLO4-TBL]` |
| ALE-CLI (score / pass) | 30.9% / 13.5% | vendor | `[HOLO4-TBL]` |
| Agentic Task Factory, Web / MCP / Desktop | 70.8 / 85.4 / 64.8% | vendor, internal | `[HOLO4-TBL]` |
| ScreenSpot-Pro, ScreenSpot-v2, OSWorld-G, WebArena, WAA, BFCL, tau2 | not reported | n/a | `[HOLO4-BLOG]` (the blog says these are not mentioned) |
| macOS benchmark | not reported | n/a | n/a |
| OSWorld-Verified LB | **not on the LB** as of 2026-11-15 | n/a | `[OSW-V]` |

### 1b. Holo4-27B (stronger but non-commercial)

- HF: https://huggingface.co/Hcompany/Holo4-27B. Base: Qwen/Qwen3.8-27B (dense, 27,356,728,560 parameters).
- License: **CC-BY-NC-4.0** `[HOLO4-27]`. Fine-tuning is allowed for non-commercial use only.
- Sizes: BF16 54.7 GB; FP8 35.9 GB; Q4_K_M GGUF 16.9 GB (+0.9 GB mmproj).
- Abliterated: https://huggingface.co/groxaxo/Holo4-27B-abliterated-exl3-4bpw and `-6bpw` (EXL3 only).

| Benchmark | Score | Type | Source |
|---|---|---|---|
| OSWorld (H harness) | 85.2% ($0.08/task) | vendor | `[HOLO4-TBL]` |
| OSWorld 2.0 (score / success) | 61.7% / 41.5% | vendor | `[HOLO4-TBL]` |
| AndroidWorld | 85.1% | vendor | `[HOLO4-TBL]` |
| AutomationBench | 45.4% (held-out: 49.3%) | vendor | `[HOLO4-TBL]` |
| ALE-CLI | 44.1% / 19.4% | vendor | `[HOLO4-TBL]` |
| ATF Web / MCP / Desktop | 80.2 / 89.4 / 72.0% | vendor | `[HOLO4-TBL]` |

Latency caveat: this is a 27B dense model, so expect roughly 6–9x slower decode than the 3B-active MoE on the Spark (my estimate).

### 1c. Holotron4-30B-A3B

- HF: https://huggingface.co/Hcompany/Holotron4-30B-A3B. Base: NVIDIA Nemotron 3 Nano Omni 30B-A3B (hybrid NemotronH). About 33.0B parameters.
- License: NVIDIA Open Model Agreement `[HOLOTRON4]`. It is "other", not OSI, though the agreement generally permits derivatives.
- OSWorld: 76.3 (base 21.0). OSWorld 2.0: 7.9. AutomationBench: 35.6 (vendor) `[HOLOTRON4]`, `[HOLO4-BLOG]`.
- No abliterated variant found. Abliterating a hybrid Mamba/attention model is less standard than a plain transformer.

### 1d. Holo-3.1 family (35B-A3B, 9B, 4B, 0.8B)

- HF: https://huggingface.co/Hcompany/Holo-3.1-35B-A3B, released 2026-05-29. Bases: Qwen3.6-35B-A3B and Qwen3.5-9B/4B/0.8B. License: Apache-2.0 `[HOLO31]`.
- Adds mobile support, native function calling and local quants. Sizes: 35B-A3B FP8 37.6 GB; Q4_K_M 21.3 GB. 9B BF16 is 18.8 GB.
- Abliterated:
  - **Full weights:** https://huggingface.co/darkc0de/Holo-3.1-35B-A3B-heretic (F32, 140 GB)
  - GGUF: https://huggingface.co/mradermacher/Holo-3.1-35B-A3B-heretic-GGUF (Q4_K_M 21.2 GB, Q8_0 36.9 GB)
  - 9B: https://huggingface.co/mradermacher/Holo-3.1-9B-abliterated-rdo-GGUF
  - 4B: https://huggingface.co/mradermacher/Holo-3.1-4B-uncensored-heretic-GGUF

| Benchmark (Holo3.1-35B-A3B) | Score | Type | Source |
|---|---|---|---|
| OSWorld (H internal impl.) | 80.0% | vendor | `[HOLO31-TBL]` |
| AndroidWorld | 79.3% | vendor | `[HOLO31-TBL]` |
| ScreenSpot-Pro | 71.5% | vendor | `[HOLO31-TBL]` |
| OSWorld-G | 78.8% | vendor | `[HOLO31-TBL]` |
| H Corporate (E-com / Biz / Collab / Multi-app) | 97.8 / 90.1 / 75.3 / 65.5% | vendor | `[HOLO31-TBL]` |

Scores for the 9B and 4B models are not in the table I saw.

### 1e. Holo3-35B-A3B (highest LB-verified open specialist)

- HF: https://huggingface.co/Hcompany/Holo3-35B-A3B, released 2026-03-23. Base: Qwen3.5-35B-A3B. 35B total / 3B active MoE. Apache-2.0 `[HOLO3]`.
- Abliterated: none found for Holo3 itself. https://huggingface.co/cabal-ai/mantis is an Apache-2.0 LoRA SFT on Holo3 for browser/workflow rollouts, not an abliteration.

| Benchmark | Score | Type | Source |
|---|---|---|---|
| **OSWorld-Verified, 100 steps** | **82.56%** (run 1); **78.15%** (run 2) | **LB-verified**, single model, no a11y tree, no coding action | `[OSW-V]` |
| OSWorld-Verified | 77.8% | vendor | `[HOLO3]`, `[HOLO3-TBL]` |
| WebArena | 64.8% | vendor | `[HOLO3-TBL]` |
| ScreenSpot-Pro | 67.6% | vendor | `[HOLO3-TBL]` |
| OSWorld-G | 77.3% | vendor | `[HOLO3-TBL]` |
| AndroidWorld | 67.2% | vendor (in Holo3.1 table) | `[HOLO31-TBL]` |
| Holo3-122B-A10B (not released on HF; ref.) | OSW-V 78.9, SS-Pro 72.2, OSWorld-G 79.4 | vendor | `[HOLO3-TBL]` |

### 1f. Holo2 / Holo1.5 (older)

- Holo2-30B-A3B is **CC-BY-NC-4.0**. Holo2-8B and Holo2-4B are Apache-2.0. Holo1.5-7B is Apache-2.0. Holo2-235B-A22B is too big.
- No abliterated variants found.

| Model | OSWorld | WebVoyager | WebArena | AndroidWorld | ScreenSpot-Pro | ScreenSpot-v2 | OSWorld-G | Source |
|---|---|---|---|---|---|---|---|---|
| Holo2-30B-A3B | 37.4 | 83.0 | 46.3 | 71.6 | 66.1 | 94.9 | 76.1 | `[HOLO2]` vendor |
| Holo2-8B | 39.9 | 80.2 | 42.2 | 60.4 | 58.9 | 93.2 | 70.1 | `[HOLO2]` vendor |
| Holo1.5-7B | 6.4 | 65.9 | 23.4 | 32.7 | 57.9 | 93.3 | 66.2 | `[HOLO2]`, `[HOLO15]` vendor |
| Holo1.5-72B | n/a | n/a | n/a | n/a | 63.3 | 94.4 | 71.8 | `[HOLO15]` |

---

## 2. Qwen general VLMs (bases of most 2026 CUA models)

### 2a. Qwen3.8-27B (dense)

- HF: https://huggingface.co/Qwen/Qwen3.8-27B, released 2026-08-05. Apache-2.0. 27B dense, 262K context `[Q38-27]`. An FP8 repo exists.
- Abliterated: many. Examples:
  - https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored (full weights)
  - `-GGUF` version of the same
  - https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF
- I have not checked whether the vision tower survives in every one of these uploads. Check each before use.

| Benchmark | Score | Type | Source |
|---|---|---|---|
| OSWorld-Verified | 84.3 | vendor | `[Q38-27]` |
| WebArena-Verified (OSWorld scaffold) | 64.8 | vendor | `[Q38-27]` |
| AndroidWorld | 81.9 | vendor | `[Q38-27]` |
| ClawEval-MM pass@3 / avg | 57.4 / 56.9 | vendor | `[Q38-27]` |
| OSWorld 2.0 (binary / partial) | 19.4 / 48.0 | vendor | `[Q38-FN]` |
| Toolathlon Verified | 67.1 | vendor | `[Q38-FN]` |
| Terminal-Bench 2.1 | 73.0 | vendor | `[Q38-27]` |

### 2b. Qwen3.8-Flash-Next (reference; too big for the ≤40 GB budget)

- HF: https://huggingface.co/Qwen/Qwen3.8-Flash-Next. 125B total, **6B active**, plus 51B n-gram embedding and a 4B MTP head. License: `qwen-community-1.0` (not Apache) `[Q38-FN]`.
- AndroidWorld 84.5; OSWorld 2.0 19.4 / 52.3; Toolathlon 73.5 `[Q38-FN]`.
- Many abliterated variants exist, for example https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated.
- Q4 is likely about 90–100 GB (est.), so it does not fit the operator budget.

### 2c. Qwen3.6-35B-A3B / Qwen3.6-27B

- Qwen3.6-35B-A3B: https://huggingface.co/Qwen/Qwen3.6-35B-A3B (2026-04-15, Apache-2.0, 35B / 3B active).
  - Its card reports no OSWorld score. It reports TAU3-Bench 67.2, MCPMark 37.0, MCP-Atlas 62.8 and Tool Decathlon 26.9 `[Q36-35]`.
  - H Company measured AutomationBench held-out at 13.1% `[HOLO4-TBL]`.
  - Abliterated: https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive, plus many Heretic variants.
- Qwen3.6-27B: AndroidWorld 70.3 `[Q36-27]`; OSWorld-Verified 63.9 (as reported in the Qwen3.8 card) `[Q38-27]`. Abliterated: https://huggingface.co/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive.

### 2d. Qwen3.5 (natively multimodal, Feb 2026)

| Model | OSWorld-Verified | ScreenSpot-Pro | AndroidWorld | BFCL-V4 | TAU2 | Source |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B (Apache-2.0) | 54.5 | 68.6 | 71.1 | 67.3 | 81.2 | `[Q35-35]` vendor |
| Qwen3.5-27B | 56.2 | 70.3 | 64.2 | 68.5 | 79.0 | `[Q35-35]` vendor |
| Qwen3.5-122B-A10B | 58.0 | 70.4 | 66.4 | 72.2 | 79.5 | `[Q35-35]` vendor |
| Qwen3.5-9B | 41.8 | 65.2 | 57.8 | 66.1 | 79.1 | `[Q35-9]` vendor |
| Qwen3.5-4B | 35.6 | 60.3 | 58.6 | 50.3 | 79.9 | `[Q35-9]` vendor |

H Company measured Qwen3.5-35B-A3B at OSWorld 54.2, ScreenSpot-Pro 64.5 and OSWorld-G 76.9 `[HOLO3-TBL]`.

Abliterated:
- https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
- https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
- `huihui-ai`-derived `Huihui-Qwen3.5-*-abliterated` variants

### 2e. Qwen3-VL (2025 baselines)

| Model | OSWorld | ScreenSpot-Pro | ScreenSpot-v2 | OSWorld-G | AndroidWorld | WebArena | Source |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-30B-A3B-Thinking | 36.6 | 49.9 | 91.8 | 65.8 | 62.9 | 45.0 | `[HOLO2]` (H-measured) |
| Qwen3-VL-8B-Thinking | 28.8 | 38.5 | 91.5 | 56.0 | 52.6 | 31.9 | `[HOLO2]` (H-measured) |
| Qwen3-VL-30B-A3B (Qwen-reported) | 30.6 (OSW-V) | 60.5 | n/a | n/a | 55.0 | n/a | `[Q35-9]` |
| Qwen3-VL-32B-Thinking, WindowsAgentArena | 42.9 | n/a | n/a | n/a | n/a | n/a | `[EVOCUA]` (cites Qwen) |
| Qwen3-VL-32B-Instruct, WindowsAgentArena | 30.9 | n/a | n/a | n/a | n/a | n/a | `[EVOCUA]` |
| qwen3-vl-flash (closed API) | 41.57 (LB, 100 steps) | n/a | n/a | n/a | n/a | n/a | `[OSW-V]` |

- macOS (MacArena, ICML 2026): Qwen3-VL-4B scored 24.23% overall (39.74% on the macOSWorld subset, 12.24% on the native subset). Qwen3-VL-2B scored 11.40% `[MACARENA]`.
- Abliterated:
  - https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated
  - https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated
  - https://huggingface.co/coder3101/Qwen3-VL-32B-Instruct-heretic-v2

### 2f. Qwen-CUA (xlangai; reference only, 397B-A17B, ~807 GB BF16)

- HF: https://huggingface.co/xlangai/Qwen-CUA (2026-08-25, Apache-2.0). Architecture `qwen3_5_moe`, 397B total / 17B active `[QWEN-CUA]`.
- Results chart `[QWEN-CUA-IMG]` (values read from the bar chart; the colour-to-series mapping is my reading):

| Benchmark | Qwen-CUA | Qwen-CUA-Max | GPT-5.5 | Opus 4.8 |
|---|---|---|---|---|
| OSWorld-Verified | 86.2 | 87.6 | 78.7 | 83.4 |
| OSWorld 2.0 (partial / binary) | 48.4 / 18.5 | n/a | n/a | 54.8 / 20.3 |
| **MacAgentBench (macOS) pass@1** | 69.2 | n/a | 66.7 | 58.4 |
| WebArena | 64.16 | n/a | 68.90 | 65.60 |

Muse-Spark-1.1 scored 80.8 on OSWorld-Verified in the same chart.

Too large for the Spark. It is still useful as a macOS reference and as a teacher model.

---

## 3. Other 2026 open CUA specialists

### 3a. Mage-VL-Cowork (Microsoft, 4B; smallest strong option)

- HF: https://huggingface.co/microsoft/Mage-VL-Cowork (2026-10-30), Apache-2.0.
- `llava_onevision2` architecture, 4B parameters, 9.48 GB BF16. Base: Mage-VL-4B.
- Hybrid action space: `computer_use` (GUI) plus `bash` `[MAGE]`.
- Abliterated: none found. Abliteration is feasible (standard decoder LM).

| Benchmark | Score | Type | Source |
|---|---|---|---|
| OSWorld-Verified, 50 steps, 361 tasks, own harness | 66.0% (OS tasks 87.5%) | vendor | `[MAGE]` |
| ScreenSpot-v2 | 87.2 | vendor (stage-1 grounding) | `[MAGE]` |
| OSWorld-G | 56.9 | vendor | `[MAGE]` |

### 3b. CUA-Master-27B (CUHK / THU / JD, 2026-10-29)

- HF: https://huggingface.co/yxl66666/CUAMaster-27B. Base: Qwen3.5-27B (dense, 27.36B). BF16 54.7 GB. **No license field on the card** `[CUAMASTER]`.
- This is the only open model I found with **macOS numbers**.

| Benchmark | Score | Type | Source |
|---|---|---|---|
| OSWorld-Verified (GUI) | 80.2 (8 min/task) | vendor | `[CUAMASTER]` |
| OSWorld 2.0 | 36.9 | vendor | `[CUAMASTER]` |
| WindowsAgentArena | 56.2 | vendor | `[CUAMASTER]` |
| **MacOSWorld** | **43.0** | vendor | `[CUAMASTER]` |
| **MacAgentBench** | **59.9** | vendor | `[CUAMASTER]` |
| Agents' Last Exam (full-pass) | 25.0 | vendor | `[CUAMASTER]` |

Abliterated: none found.

### 3c. UI-Venus-2-9B (Ant / inclusionAI, 2026-08)

- HF: https://huggingface.co/inclusionAI/UI-Venus-2-9B. Base: Qwen3.5-9B (dense 9B).
- **License pending.** The card says Apache-2.0 has deliberately not been carried over `[VENUS2]`. Treat it as not cleared for fine-tuning until that changes.

| Benchmark | Score | Source |
|---|---|---|
| OSWorld-Verified | 70.8 | `[VENUS2]` vendor |
| AndroidWorld | 80.2 | `[VENUS2]` |
| Online-Mind2Web | 74.0 | `[VENUS2]` |
| WebVoyager | 90.8 | `[VENUS2]` |
| ScreenSpot-Pro | 73.0 | `[VENUS2]` |
| OSWorld-G-R | 78.5 | `[VENUS2]` |
| (UI-Venus-2-27B, not released) | OSW-V 80.5, SS-Pro 74.1 | `[VENUS2]` |

The same card cites "Qwen-UI-Agent-27B" (OSW-V 79.5, SS-Pro 76.6). I found no HF repo for it, so it is [UNVERIFIED as open].

UI-Venus-1.5-8B and 2B (Apache-2.0) also exist on HF. UI-Venus-1.5-30B-A3B scored SS-Pro 69.6 and AndroidWorld 77.6 `[VENUS2]`.

No abliterated variants found.

### 3d. Nex-N2.5-mini (nex-agi, 2026-09-08)

- HF: https://huggingface.co/nex-agi/Nex-N2.5-mini. Apache-2.0. `Qwen3_5MoeForConditionalGeneration`: 35.1B parameters, 256 experts, top-8, hidden 2048. This matches the Qwen3.5/3.6-35B-A3B shape, so about 3B active.
- Native function calling (`qwen3_coder` parser) `[NEX]`.
- Scores: OSWorld-Verified 71.2; OSWorld-2 30.5; WebArena-Verified 63.4; OSWorld-G 82.9. All vendor, from the NexCUA harness `[NEX]`.
- Abliterated: none found.

### 3e. Fara1.5 (Microsoft, 2026-05/07) and Fara-7B

- HF:
  - https://huggingface.co/microsoft/Fara1.5-27B (Qwen3.5-27B base)
  - https://huggingface.co/microsoft/Fara1.5-9B
  - https://huggingface.co/microsoft/Fara1.5-4B
  - All MIT `[FARA15]`.
- Fara1.5 is **web-browser only**, trained at 1440×900. It is also trained to **stop at "critical points"** (logins, payments, sending messages). That is a refusal-like behaviour you would have to remove.

| Model | WebVoyager | Online-Mind2Web | WebTailBench | Source |
|---|---|---|---|---|
| Fara1.5-27B | 89.3 | 72.3 | 40.2 | `[FARA15]` |
| Fara1.5-9B | 86.6 | 63.4 | 32.3 | `[FARA15]` |
| Fara1.5-4B | 80.8 | 57.3 | 27.4 | `[FARA15]` |
| Fara-7B (Qwen2.5-VL-7B base, MIT) | 73.5 | 34.1 | 38.4 | `[FARA7]` |

Abliterated:
- Fara-7B: https://huggingface.co/huihui-ai/Huihui-Fara-7B-abliterated and https://huggingface.co/josephmayo/Fara-7B-Abliterated-v2
- Fara1.5: none found.

### 3f. GUI-Owl-1.5 / Mobile-Agent-v3 (Alibaba Tongyi, mPLUG, 2026-02)

- HF: https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct and `-32B-Instruct` (also Thinking, 4B and 2B versions). MIT `[GUIOWL15]`.

| Model | OSW-V | AndroidWorld | OSWorld-MCP | WindowsAA | WebArena | Online-M2W | Source |
|---|---|---|---|---|---|---|---|
| GUI-Owl-1.5-32B-Instruct | 56.5 (vendor); **55.44 LB-verified, 50 steps** | 69.4 | 47.6 | 44.8 | n/a | n/a | `[GUIOWL15]`, `[OSW-V]` |
| GUI-Owl-1.5-8B-Instruct | 52.3 | 69.0 | 41.8 | 31.7 | 45.7 | 41.7 | `[GUIOWL15]` |
| GUI-Owl-1.5-8B-Thinking | 52.9 | 71.6 | 38.8 | 35.1 | 46.7 | 48.6 | `[GUIOWL15]` |
| GUI-Owl-7B (2025) | 32.11 LB (15 steps) | n/a | n/a | n/a | n/a | n/a | `[OSW-V]` |
| Mobile-Agent-v3 w/ GUI-Owl-32B **[framework]** | 38.91 LB (50 steps) | n/a | n/a | n/a | n/a | n/a | `[OSW-V]` |

- ScreenSpot-Pro and ScreenSpot-v2 scores are in the tech report (arXiv 2602.16855). I did not collect them: [UNVERIFIED].
- No abliterated variants. mPLUG/ToolCUA-8B (Qwen3-VL-8B base, MIT) reports 46.85% on feasible OSWorld-MCP tasks `[TOOLCUA]`.

### 3g. EvoCUA (Meituan, 2026-01)

- HF: https://huggingface.co/meituan/EvoCUA-32B-20260105 and `-8B-20260105`. Apache-2.0. Base: Qwen3-VL-32B/8B-Thinking `[EVOCUA]`.

| Model | OSWorld-Verified | WindowsAgentArena | Source |
|---|---|---|---|
| EvoCUA-32B-20260105 | **56.73 LB-verified (50 steps)** | 56.48 (vendor) | `[OSW-V]`, `[EVOCUA]` |
| EvoCUA-8B-20260105 | **46.06 LB-verified (50 steps)** | n/a | `[OSW-V]` |

No abliterated variants.

### 3h. MAI-UI (Alibaba Tongyi, 2025-12)

- HF: https://huggingface.co/Tongyi-MAI/MAI-UI-8B and `-2B`. Apache-2.0. The family spans 2B to 235B-A22B; only 2B and 8B are on HF.
- Family-best numbers: ScreenSpot-Pro 73.5, OSWorld-G 70.9, AndroidWorld 76.7, MobileWorld 41.7 `[MAI-UI]`. The card does not say clearly which size produced them, so do not attribute them to the 8B: [UNVERIFIED for 8B].
- No abliterated variants.

### 3i. Tencent Hy4-preview (reference; too big)

- HF: https://huggingface.co/tencent/Hy4-preview. 770B total / 49B active MoE. Apache-2.0 `[HY4]`. OSWorld score: not extracted.

### 3j. NorKa (MindStudio / Beo AI)

- A news item (2026-11-06) claims NorKa Pro (397B-A17B) 88, Mini (35B) 85 and Nano (9B) 78.9 on OSWorld-Verified, with weights "on Hugging Face" `[NORKA]`.
- **I found no HF repo, base model or license.** All [UNVERIFIED].

---

## 4. 2025-era open models (for context)

| Model | HF repo | Params / base | License | OSWorld-Verified (LB) | ScreenSpot-Pro | Other | Abliterated |
|---|---|---|---|---|---|---|---|
| UI-TARS-1.5-7B | ByteDance-Seed/UI-TARS-1.5-7B | 7B (Qwen2.5-VL) | Apache-2.0 | 29.6 (100 steps), 29.4 (50), 25.7 (15) `[OSW-V]`; vendor OSWorld 27.5 `[UITARS]` | 49.6 `[UITARS]` | macOSWorld (UI-TARS-7B-DPO) 4.8% `[MACOSWORLD]`; MacArena 21.14% `[MACARENA]` | None found |
| UI-TARS-2 | **not released on HF** (no ByteDance-Seed repo; the LB lists it among closed models) | n/a | n/a | 53.1 (100 steps) `[OSW-V]` | n/a | WAA 50.6 (cited) `[EVOCUA]` | n/a |
| OpenCUA-7B / 32B / 72B | xlangai/OpenCUA-* | Qwen2.5-VL | MIT | 7B: 28.8 (50 steps); 32B: 35.7 (100); 72B: 46.1 (100) `[OSW-V]` | 50.0 / 55.3 / 60.8 `[OPENCUA]` | ScreenSpot-v2 92.3 / 93.4 / 92.9; OSWorld-G 55.3 / 59.6 / 59.2 `[OPENCUA]` | None found |
| GTA1-7B / 32B (grounders) | Salesforce/GTA1-* | Qwen2.5-VL | MIT | Only as **[framework]** w/ GPT-5: 63.41 LB (100 steps) `[OSW-V]`; GTA1-32B w/ GPT-5 63.4 vendor `[GTA1]` | 7B 50.1–55.5, 32B 63.6 `[GTA1]` | WAA (GTA1-32B w/ o3) 51.2 `[GTA1]` | None found |
| autoglm-os-9b | not on HF as `-os` (only AutoGLM-Phone-9B) | 9B | n/a | 48.03 (50 steps) `[OSW-V]` | n/a | n/a | n/a |
| DART-GUI-7B | dart-gui/dart-gui-7b | 7B | n/a | 40.47 (30 steps) `[OSW-V]` | n/a | n/a | n/a |
| ScaleCUA-3B / 7B / 32B | OpenGVLab/ScaleCUA-* | Qwen2.5-VL | Apache-2.0 | not on LB | n/a | n/a | None found |
| Kimi-VL-A3B | moonshotai | 16B / 3B MoE | MIT-ish | 10.3 (100 steps) `[OSW-V]` | n/a | n/a | n/a |
| GLM-4.6V (106B-A12B) / -Flash (9B) | zai-org/GLM-4.6V | MIT | MIT | no GUI numbers in card (I grepped the card) | n/a | n/a | Not checked |

Agent S3 (Simular) is a **framework**. Its best LB entry is "agent s3 w/ Opus 4.5 + GPT-5 bBoN (N=10)" at 72.58 (100 steps, multiple rollouts). With GPT-5 bBoN N=10 it scored 69.9 `[OSW-V]`. Its grounder is UI-TARS-1.5-7B per the Agent S docs [UNVERIFIED here]. CoACT-1 **[framework]** scored 60.76 at 150 steps `[OSW-V]`.

Not found / not checked:
- UltraCUA and InfiGUI: no HF hits for those names.
- Step-GUI: StepFun's GUI model on HF is `stepfun-ai/GELab-Zero-4B-preview` (mobile). Scores not collected.

---

## 5. Frontier and large-model reference rows

### 5a. Official OSWorld-Verified LB (osworld-v1.xlang.ai, spreadsheet fetched 2026-11-15) `[OSW-V]`

| Entry | Type | Steps | Success | Date |
|---|---|---|---|---|
| Intelligence-Indeed Agent | [framework] (+coding action) | 100 | 90.19 | 2026-07-25 |
| **claude-fable-5 [1m]** | General, closed | 100 | **85.96** | 2026-08-01 |
| Pointer Agent w/ Opus 4.7 | [framework] | 100 | 83.64 | 2026-05-21 |
| claude-opus-5 [1m] | General, closed | 100 | 83.39 | 2026-08-01 |
| Coasty CUA v1 | General (+coding), closed | 100 | 82.81 | 2026-07-01 |
| **Holo3-35B-A3B** | **Specialized, open** | 100 | **82.56** (also 78.15) | 2026-04-20 |
| Muse Spark 1.1 (Meta) | General, closed | 100 | 80.67 | 2026-07-09 |
| MiniMax M3 | General | 100 | 75.19 | 2026-06-08 |
| Qwen 3.7 Plus | General, closed | 100 | 73.30 | 2026-05-25 |
| Kimi K2.6 (open weights, ~1T) | General | 100 | 73.06 | 2026-04-20 |
| claude-sonnet-4-6 | General, closed | 100 | 72.11 | 2026-03-08 |
| Kimi K2.5 | General | 100 | 63.30 | 2026-01-31 |
| claude-sonnet-4-5 | General, closed | 100 / 50 / 15 | 62.88 / 58.08 / 42.88 | 2025-10-31 |
| OpenAI computer-use-preview | Specialized, closed | 100 / 50 / 15 | 31.4 / 31.3 / 26.0 | 2025-07-28 |

Notes:
- The LB has **no OpenAI GPT-5.x or Gemini 3.x rows**.
- No Holo4, Holo-3.1, Qwen3.8 or CUA-Master rows.
- The LB HTML says the table loads dynamically. I parsed the xlsx it fetches.

### 5b. Vendor / aggregator frontier numbers (not LB-verified)

| Model | OSWorld-Verified | Source |
|---|---|---|
| Qwen3.8 Max (closed) | 86.1 | `[HOLO4-TBL]` (as reported by Qwen), `[LLMSTATS]` |
| Claude Fable 5 | 85.0 (Kimi table, "max, w/ fallback"); 86.0 (Holo4 table, from LB) | `[KIMI-K3]`, `[HOLO4-TBL]` |
| Kimi K3 (max; open weights, too big) | 84.8 | `[KIMI-K3]` |
| Claude Opus 4.8 (max) | 83.4 | `[KIMI-K3]`, `[NEX]` |
| GPT-5.6 Sol (max) | 83.0 / 83.2 | `[KIMI-K3]`, `[NEX]` |
| Gemini 3.6 Flash | 83.0 | `[LLMSTATS]` aggregator, self-reported |
| GPT-5.5 (xhigh) | 78.7–79.0 | `[HOLO4-TBL]` (as reported by OpenAI), `[KIMI-K3]` |
| OSWorld 2.0: Opus 5.5 81.8; GPT-6 Astra 73.5; Muse Spark 1.3 66.9; Fable 5 66.1 | n/a | `[HOLO4-TBL]`, `[KIMI-K3]` |

### 5c. Frontier ScreenSpot-Pro (vendor-reported, via aggregator `[BENCHLM-SSP]`)

| Model | ScreenSpot-Pro |
|---|---|
| GPT-6 Astra | 92.7 |
| Claude Opus 4.8 | 87.9 |
| GPT-5.4 | 85.4 |
| Qwen3.8 Max | 84.5 |
| Gemini 3.1 Pro | 84.4 |
| Muse Spark | 84.1 |
| Qwen3.7 Plus | 79.0 |
| Nemotron 3 Nano Omni 30B-A3B | 57.8 |

The page warns that zoom/crop tools, planners and resolution change results. Best open-weight models for comparison:
- UI-Venus-2-9B: 73.0 `[VENUS2]`
- MAI-UI (family best): 73.5 `[MAI-UI]`
- Holo3-122B (unreleased): 72.2 `[HOLO3-TBL]`
- Holo3.1-35B-A3B: 71.5 `[HOLO31-TBL]`

### 5d. macOS-specific results

**macOSWorld** (arXiv 2506.04135, 202 tasks, 30 apps), overall success rate `[MACOSWORLD]`:
- Claude CUA: 37.1% (English 44.4%)
- OpenAI CUA: 33.8%
- Gemini 2.5 Pro: 18.5%
- GPT-4o: 5.3%
- **UI-TARS-7B-DPO: 4.8%**
- ShowUI: 1.1%
- Safety: both strong CUAs were about 70% deceivable by pop-up attacks.

**MacArena** (arXiv 2606.06560, ICML 2026), overall success rate `[MACARENA]`:
- OpenAI computer-use-preview: 31.83%
- Qwen3-VL-4B: 24.23%
- UI-TARS-1.5-7B: 21.14%
- Qwen3-VL-2B: 11.40%
- Rankings invert on the macOS-native tasks.

**MacAgentBench** pass@1:
- Qwen-CUA (397B): 69.2
- GPT-5.5: 66.7
- Opus 4.8: 58.4
- Qwen-3.7: 57.1
- Source: `[QWEN-CUA-IMG]`, read from the chart.

**CUA-Master-27B:** MacOSWorld 43.0; MacAgentBench 59.9 `[CUAMASTER]`.

None of the top Spark candidates (Holo4, Holo3.x, Mage-VL-Cowork) publishes a macOS score. Plan your own macOS eval, for example on macOSWorld or MacArena tasks.

---

## 6. Caveats

1. **Harness dependence.** Holo4, Holo3.1, Qwen3.8, Nex, CUA-Master and UI-Venus-2 scores are vendor-run in their own harnesses. H Company's harness changes included a shell on the desktop, a unified code-plus-click tool and raised step budgets. Only Holo3-35B-A3B (82.56 / 78.15), EvoCUA (56.73 / 46.06) and GUI-Owl-1.5-32B (55.44) are LB-verified among 2026 open specialists. Holo3's two LB rows differ by 4.4 points, which shows the run-to-run variance.
2. **"OSWorld" ≠ "OSWorld-Verified" ≠ "OSWorld 2.0".** Holo4 labels its column "OSWorld". H Company's Holo3.1 footnote says its OSWorld numbers come from an internal implementation and "differ slightly" from the official ones.
3. **Abliteration status.** Only GGUF/EXL3 abliterated builds exist for Holo4. Full-weight Heretic exists for Holo-3.1-35B-A3B (darkc0de) and for the Qwen bases. Removing refusals from a CUA can also remove the trained "stop and ask" safety behaviours (Fara is explicitly trained for them). You need your own guardrails in the harness.
4. **Licenses.**
   - Holo4-27B and Holo2-30B-A3B are **non-commercial** (CC-BY-NC).
   - UI-Venus-2-9B has its license **pending**.
   - CUA-Master-27B has **no license** listed.
   - Holotron4 uses the NVIDIA Open Model Agreement.
   - Qwen3.8-Flash-Next uses the qwen-community license.
   - Holo4-35B-A3B, Holo3, Holo3.1, Qwen3.x, Nex-N2.5-mini, Mage-VL-Cowork, MAI-UI and EvoCUA are Apache-2.0. GUI-Owl-1.5, OpenCUA, Fara and GTA1 are MIT.
5. **macOS gap.** Most training is Ubuntu, Windows, Android or web. The MacArena and macOSWorld results show large drops on macOS-native tasks. Budget time for macOS fine-tuning data.
6. **Not verified / not found:** NorKa (no HF repo), Qwen-UI-Agent-27B (no HF repo), UI-TARS-2 weights (not on HF), MAI-UI-8B-specific grounding numbers, GUI-Owl-1.5 ScreenSpot numbers, GLM-4.6V GUI numbers.

---

## Source key

| Key | URL |
|---|---|
| OSW-V | http://osworld-v1.xlang.ai/static/data/osworld_verified_results.xlsx (loaded by http://osworld-v1.xlang.ai/; os-world.github.io redirects there) |
| OSW-SELF | http://osworld-v1.xlang.ai/static/data/self_reported_results.xlsx |
| LLMSTATS | https://llm-stats.com/benchmarks/osworld-verified |
| BENCHLM-SSP | https://benchlm.ai/md/benchmarks/screenspot-pro.md |
| HOLO4-35 | https://huggingface.co/Hcompany/Holo4-35B-A3B |
| HOLO4-27 | https://huggingface.co/Hcompany/Holo4-27B |
| HOLO4-TBL | https://assets.hcompanyprod.fr/holo4/hf/holo4-benchmark-table.png (embedded in both Holo4 cards) |
| HOLO4-BLOG | https://hcompany.ai/newsroom/holo4 |
| HOLOTRON4 | https://huggingface.co/Hcompany/Holotron4-30B-A3B |
| HOLO31 | https://huggingface.co/Hcompany/Holo-3.1-35B-A3B |
| HOLO31-TBL | https://cdn-uploads.huggingface.co/production/uploads/69ce2739f4b9146a31e99a2f/eKbMPsY32AF4PCNzumeZQ.png |
| HOLO3 | https://huggingface.co/Hcompany/Holo3-35B-A3B |
| HOLO3-TBL | https://huggingface.co/Hcompany/Holo3-35B-A3B/resolve/main/benchmark_table_light.png |
| HOLO2 | https://huggingface.co/Hcompany/Holo2-30B-A3B |
| HOLO15 | https://huggingface.co/Hcompany/Holo1.5-7B |
| Q38-27 | https://huggingface.co/Qwen/Qwen3.8-27B |
| Q38-FN | https://huggingface.co/Qwen/Qwen3.8-Flash-Next |
| Q36-35 | https://huggingface.co/Qwen/Qwen3.6-35B-A3B |
| Q36-27 | https://huggingface.co/Qwen/Qwen3.6-27B |
| Q35-35 | https://huggingface.co/Qwen/Qwen3.5-35B-A3B |
| Q35-9 | https://huggingface.co/Qwen/Qwen3.5-9B |
| QWEN-CUA | https://huggingface.co/xlangai/Qwen-CUA |
| QWEN-CUA-IMG | https://raw.githubusercontent.com/xlang-ai/Qwen-CUA/main/assets/readme/main-results.png |
| MAGE | https://huggingface.co/microsoft/Mage-VL-Cowork |
| CUAMASTER | https://huggingface.co/yxl66666/CUAMaster-27B |
| VENUS2 | https://huggingface.co/inclusionAI/UI-Venus-2-9B |
| NEX | https://huggingface.co/nex-agi/Nex-N2.5-mini |
| FARA15 | https://huggingface.co/microsoft/Fara1.5-27B |
| FARA7 | https://huggingface.co/microsoft/Fara-7B |
| GUIOWL15 | https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct |
| TOOLCUA | https://huggingface.co/mPLUG/ToolCUA-8B |
| EVOCUA | https://huggingface.co/meituan/EvoCUA-32B-20260105 |
| MAI-UI | https://huggingface.co/Tongyi-MAI/MAI-UI-8B |
| HY4 | https://huggingface.co/tencent/Hy4-preview |
| NORKA | https://af.net/realtime/norka-releases-open-weight-model-family-scoring-88-on-osworld/ |
| UITARS | https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B |
| OPENCUA | https://huggingface.co/xlangai/OpenCUA-7B |
| GTA1 | https://huggingface.co/Salesforce/GTA1-32B |
| KIMI-K3 | https://huggingface.co/moonshotai/Kimi-K3 |
| KIMI-K26 | https://huggingface.co/moonshotai/Kimi-K2.6 |
| MACOSWORLD | https://arxiv.org/html/2506.04135 |
| MACARENA | https://arxiv.org/html/2606.06560v1 |

Model sizes, creation dates and parameter counts came from the HF API: `https://huggingface.co/api/models/<repo>?expand[]=safetensors` and `/tree/main`. Abliterated variants came from HF API search: `https://huggingface.co/api/models?search=<name>`.
