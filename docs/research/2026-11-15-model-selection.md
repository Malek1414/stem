# Model selection for the coder + operator pair on one DGX Spark

Date: 2026-11-15. Hardware target: one NVIDIA DGX Spark (GB10, 128 GB unified, about 119.7 GiB usable, about 273 GB/s). Budget: weights plus KV headroom of about 110 GB for both agents.

Working notes with every URL (one file per research track): `2026-11-15-model-selection-notes/{coders,cua,vlm_bases,spark_and_abliteration}.md`.

## How to read the numbers

- **V** means the vendor reported it on its own model card or blog, in its own harness. **3P** means another lab ran it, usually NVIDIA re-running competitors in NeMo Gym. A competitor running your model is independent, but not neutral. **I** means an independent leaderboard (official OSWorld-Verified spreadsheet, swebench.com bash-only, SWE-rebench, Scale SWE-bench Pro). **S** means a secondary source: press or an aggregator citing a vendor. **[UNVERIFIED]** means it was only seen in a search snippet.
- I re-checked these myself on 2026-11-15:
  - The Qwen3.8-27B model card (SWE-Pro 61.7, TB2.1 73.0, LCB v6 90.3, OSWorld-Verified 84.3, AndroidWorld 81.9, Apache-2.0, no SWE-V / ScreenSpot / BFCL / tau2 rows).
  - The Holo4-35B-A3B card (Apache-2.0, base Qwen3.6-35B-A3B, hai-agents harness).
  - The OSWorld-Verified leaderboard spreadsheet (`osworld_verified_results.xlsx`): Holo3-35B-A3B 82.56 / 78.15, claude-fable-5 85.96, claude-opus-5 83.39, Kimi K2.6 73.06, GUI-Owl-1.5-32B 55.44, EvoCUA-32B 56.73, UI-TARS-1.5-7B 29.6.
  - Every `config.json` used for the KV maths below.
  - Everything else was collected by four research subagents that fetched the pages. The URLs are inline.
- **The benchmarks moved during 2026.** Cards from August to October 2026 mostly report SWE-bench Pro and Terminal-Bench 2.1 instead of SWE-bench Verified and Terminal-Bench 2.0. The swebench.com leaderboard stopped updating on 2026-02-26 and the Aider leaderboard on 2025-11-20. So 2025/early-2026 models and late-2026 models are often not directly comparable. Compare inside one table where possible, for example a vendor's own comparison column or NVIDIA's 3P column.
- **Vendor numbers overstate independent ones, sometimes by a lot:**
  - gpt-oss-120b: SWE-V 62.4 V vs 26.0 I (swebench.com bash-only).
  - Qwen3.5-122B: OSWorld-Verified 58.0 V vs 51.5 3P (NVIDIA).
  - Qwen3.6-27B: SWE-V 77.2 V, but 31.2% on SWE-rebench I. These are different task sets, so this is not a like-for-like gap.
- **OSWorld is not one benchmark.** "OSWorld" (H Company's harness, up to hundreds of steps), "OSWorld-Verified" (the official leaderboard at 15, 50 or 100 steps) and "OSWorld 2.0" are different. Framework entries (a closed planner plus an open grounder) are excluded from the open-model rankings.
- **Decode ceilings are my arithmetic.** I compute 273 GB/s divided by the bytes of active weights per token. That is an upper bound. Measured Spark numbers are given where they exist.

---

## 0. The verdict on Qwen3-Coder-Next first

`Qwen/Qwen3-Coder-Next` (2026-01-30, Apache-2.0, 79.7B total / about 3B active, hybrid Gated DeltaNet MoE, text only). Q4_K_M is 48.5 GB and FP8 is 80.4 GB.

| Benchmark | Score | Scaffold | Type | Source |
|---|---|---|---|---|
| SWE-bench Verified | 70.6 / 71.1 / 71.3 | SWE-agent / mini-SWE-agent / OpenHands | V | https://arxiv.org/html/2603.00729v1 |
| SWE-bench Pro | 42.7 / 38.7 | SWE-agent / mini-SWE-agent | V | same |
| SWE-bench Multilingual | 62.8 / 56.2 / 64.3 | SWE-agent / mini / OpenHands | V | same |
| Terminal-Bench 2.0 | 36.2 best (Terminus2-json); **25.8 in Qwen Code** | | V | same |
| Aider Polyglot | 66.2 | | V | same |
| LiveCodeBench v6 | 58.9 | | V | same |
| SWE-rebench (Feb 2026 window, 57 tasks) | 54.4% | SWE-rebench | I via S | https://ai-radar.it/article/swe-rebench-gpt-5-4-qwen3-5-e-gemini-3-1-pro-a-confronto |
| Spark decode | 42.7 tg FP8 vLLM (37.2 at 32K); 74.8 tg vLLM 4-bit; 59.2 tg Ollama Q4; NVFP4 only 14.1 | | measured | https://forums.developer.nvidia.com/t/how-to-run-qwen3-coder-next-on-spark/359571 , https://forums.developer.nvidia.com/t/measured-inference-benchmarks-on-a-single-dgx-spark-same-harness-across-ollama-llama-cpp-and-vllm-notes-data-published/379766 |

How it compares:

- **Qwen3.6-35B-A3B** has the same 3B-active speed class and needs less than half the memory (22 GB at Q4). On Qwen's own cards it beats Coder-Next on every agentic coding row: SWE-V 73.4, SWE-Pro 49.5, TB2.0 51.5 vs 36.2.
- **Qwen3.8-27B** is far ahead on SWE-Pro (61.7 vs 42.7) but decodes about 4 to 6 times slower on the Spark.

**Our 0/7 may partly be the serving stack.** llama.cpp issue #20164 reports Qwen3-Coder-Next (and Qwen3.5-35B-A3B) looping on repeated, almost-correct tool calls when a tool has several optional parameters, for example OpenCode `read` with offset/limit at about 30K tokens. It was fixed by PR #20171. There are also known `qwen3_coder` XML-parser and chat-template mismatches.

- Issue: https://github.com/ggml-org/llama.cpp/issues/20164
- Template fix: https://huggingface.co/SergioPatricio/qwen3-coder-next-llamacpp-template
- Write-up: https://allanchan339.github.io/bug-fixes/2026/04/13/Qwen35-tool-calling.html

**Action:** re-run the 0/7 replay on a current llama.cpp or vLLM build with the correct tool parser before blaming the model. Run every replacement on that same fixed stack, because the abliterated GGUF chat template is another variable.

---

## 1. Ranked tables per role

### 1a. Coder (direction B specialist; must fit at or below about 70 GB)

| # | Model (HF repo) | Release | Total / active, arch | Vision | License | Q4 / FP8 / NVFP4 (GB) | Coding scores | Tool calling | Spark speed | Abliterated |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Qwen3.8-27B** (`Qwen/Qwen3.8-27B`) | 2026-08-05 | 27B dense; hybrid GDN (16 of 64 layers full attention), MTP | yes | Apache-2.0 | 17.7 (UD-Q4_K_XL) / 30.9 / n.a. | SWE-Pro **61.7** V (Claude Code harness, refined set); TB2.1 **73.0** V; LCB v6 90.3 V; SWE-Multilingual 73.8 V (mini-SWE-agent); no SWE-V reported. AA Agentic Index 51 S | Toolathlon-Verified 67.1 V; BFCL/tau2 not reported | No direct measurement. Same-shape Qwen3.5-27B int4: **8.7 tg** plain, 36 to 71 with DFlash speculation (real-world 15 to 20) | yes: https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated , https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored |
| 2 | **Qwen3.6-35B-A3B** (`Qwen/Qwen3.6-35B-A3B`) | 2026-04-15 | 35B / 3B MoE, hybrid GDN | yes | Apache-2.0 | 22.1 / 37.5 / 22.5 (3rd party) | SWE-V **73.4** V (internal bash+edit scaffold); SWE-Pro 49.5 V; TB2.0 51.5 V; TB2.1 44.4 3P (NVIDIA); **SWE-rebench 24.7% I** | TAU3 67.2 V; MCP-Atlas 62.8 V / 51.2 3P | Qwen3.5-35B-A3B FP8 on an SM121 build: **48.6 to 52.9 tg**; 68 to 82 with MTP | yes: https://huggingface.co/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated , https://huggingface.co/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 |
| 2b | KAT-Coder-V2.5-Dev (`Kwaipilot/KAT-Coder-V2.5-Dev`), SFT+RL of #2 | 2026-07-23 | same as #2 | (base has vision) | Apache-2.0 | about 22 est. / BF16 69.3 | Uniform Claude Code harness: SWE-V 69.4, SWE-Pro 46.0, TB2.1 41.0, vs its base 64.4 / 40.6 / 32.0 in the same harness (V) | n.r. | as #2 | not found |
| 3 | **Qwen3.6-27B** (`Qwen/Qwen3.6-27B`) | 2026-04-21 | 27B dense hybrid | yes | Apache-2.0 | 16.8 / 30.9 | SWE-V 77.2 V; SWE-Pro 53.5 V; TB2.0 59.3 V; **SWE-rebench 31.2% I (best open model under 100B there)** | n.r. | as #1 | yes: https://huggingface.co/huihui-ai/Huihui-Qwen3.6-27B-abliterated |
| 4 | **Nemotron 3.5 Super VL 120B-A12B** (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`) | 2026-10-15 | 120B / 12B, Mamba-2 + attention hybrid MoE (8 of 88 layers attention), MTP | yes | OpenMDW-1.1 | 86.9 / n.a. / **75.0** (official, DGX Spark recipe on the card) | SWE-V 71.1 V; SWE-Pro 51.3 V; TB2.1 62.8 V; SWE-Multilingual 67.0 V | tau2 retail/airline/telecom 69.3 / 75.8 / 73.2 V | Previous-gen Nemotron-3-Super NVFP4: **23.5 tg** flat to 100K (https://vllm.ai/blog/2026-06-01-vllm-dgx-spark) | yes: https://huggingface.co/armin1/Nemotron-3.5-Super-VL-120B-A12B-NVFP4-abliterated |
| 5 | Muse Glimmer-30B (`meta-models/Muse-Glimmer-30B`) | 2026-08-09 | 29.6B dense | yes | Apache-2.0 | 15.9 / Q8 29.6 | SWE-V 76.0 V; SWE-Pro 51.2 V; TB2.1 51.7 V | MCP-Atlas 75.5 V | dense, about 17 ceiling | not checked |
| ref | Qwen3-Coder-Next (current) | 2026-01-30 | 80B / 3B | no | Apache-2.0 | 48.5 / 80.4 | see section 0 | | 42.7 to 74.8 tg | yes (huihui, Heretic) |
| out | Qwen3.5-122B-A10B | 2026-02-24 | 122B / 10B | yes | Apache-2.0 | 77.0 / 127 | SWE-V 72.0 V / 71.9 3P; TB2.0 49.4 V | BFCL-V4 72.2 V; TAU2 79.5 V | 38.4 tg INT4 + MTP | yes (GGUF, HauhauCS, heretic int4) |
| out | Devstral Small 2 24B | 2025-12 | 24B dense | yes | Apache-2.0 | 14.3 | SWE-V 68.0 V / **56.4 I**; TB2 22.5 V | | about 15 est. | yes (AliBilge mirror) |
| out | gpt-oss-120b | 2025-08 | 117B / 5.1B | no | Apache-2.0 | about 63 (MXFP4) | SWE-V 62.4 V / **26.0 I**; SWE-Pro **16.2 I**; Aider 41.8 I | | 58.7 tg llama.cpp | yes, but high KL (0.53 to 0.92) |
| out | MiniMax-M2.5 | 2026-02 | 230B / 10B | no | | Q3 109 GB | SWE-V 75.8 I | | | yes | Does not fit beside an operator |

Sources for the coder rows:
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next (Qwen3.8-27B Multilingual and Toolathlon)
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- https://huggingface.co/Qwen/Qwen3.6-27B
- https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 (3P TB2.1)
- https://huggingface.co/meta-models/Muse-Glimmer-30B
- https://swe-rebench.com/
- https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json
- https://labs.scale.com/leaderboard/swe_bench_pro_public
- https://forums.developer.nvidia.com/t/dgx-spark-13-49-tok-s-with-qwen3-5-35b-native-sm121-kernel-build-guide/365083
- https://forums.developer.nvidia.com/t/qwen3-5-27b-optimisation-thread-starting-at-30-t-s-tp-1/366009
- https://www.qubrid.com/blog/qwen38-27b-benchmarks-official-and-independent-results (AA, S)

Field notes (S, https://www.developersdigest.tech/blog/qwen-3-8-27b-local-agentic-coding-2026):
- Qwen3.8-27B "defaults to wildly overthinking": in one report it spent 22K reasoning tokens on a single task. Cap `reasoning_effort` for the coder.

### 1b. Operator (computer use; ideally 40 GB or less and low active parameters)

| # | Model (HF repo) | Release | Total / active, base | License | Q4 / FP8 / NVFP4 (GB) | GUI scores | Tool calling | Abliterated |
|---|---|---|---|---|---|---|---|---|
| 1 | **Holo4-35B-A3B** (`Hcompany/Holo4-35B-A3B`) | 2026-09-24 | 35.1B / about 3B MoE; Qwen3.6-35B-A3B | Apache-2.0 | 21.3 (+0.9 mmproj) / 37.6 / 22.5 | OSWorld 80.8 V (H harness, mean of 2 to 4 runs); OSWorld 2.0 30.9 V; AndroidWorld 77.6 V; AutomationBench 34.5 V. **Not on the OSWorld-Verified LB yet.** No ScreenSpot numbers. | native function calling and MCP (card) | GGUF only: https://huggingface.co/jduraes/Holo4-35B-A3B-heretic-Q4_K_M , https://huggingface.co/lavendercomputing/Holo4-35B-A3B-Heretic-GSQ-RCO-GGUF . We would abliterate BF16 (70 GB) ourselves; Heretic supports this architecture |
| 2 | **Holo-3.1-35B-A3B** (`Hcompany/Holo-3.1-35B-A3B`) | 2026-05-29 | 35B / 3B; Qwen3.6-35B-A3B | Apache-2.0 | 21.3 / 37.6 | OSWorld 80.0 V; **ScreenSpot-Pro 71.5 V**; OSWorld-G 78.8 V; AndroidWorld 79.3 V | native function calling | **full weights:** https://huggingface.co/darkc0de/Holo-3.1-35B-A3B-heretic ; GGUF https://huggingface.co/mradermacher/Holo-3.1-35B-A3B-heretic-GGUF |
| 3 | **Holo3-35B-A3B** (`Hcompany/Holo3-35B-A3B`) | 2026-03-23 | 35B / 3B; Qwen3.5-35B-A3B | Apache-2.0 | about 21 est. / about 37 est. | **OSWorld-Verified 82.56 / 78.15 LB (100 steps, single model, two runs)**; 77.8 V; ScreenSpot-Pro 67.6 V; OSWorld-G 77.3 V; WebArena 64.8 V | n.r. | none for Holo3 itself |
| 4 | Mage-VL-Cowork (`microsoft/Mage-VL-Cowork`) | 2026-10-30 | 4B dense (llava_onevision2) | Apache-2.0 | about 3 est. / BF16 9.5 | OSWorld-Verified 66.0 V (50 steps, own harness); ScreenSpot-v2 87.2 V; OSWorld-G 56.9 V | GUI + bash action space | none; feasible |
| 5 | Qwen3.8-27B as operator | 2026-08-05 | 27B dense | Apache-2.0 | 17.7 / 30.9 | OSWorld-Verified **84.3 V** (not reproduced); AndroidWorld 81.9 V; WebArena-Verified 64.8 V | see 1a | yes (see 1a) |
| alt | GUI-Owl-1.5-32B / 8B (`mPLUG/GUI-Owl-1.5-*`) | 2026-02 | dense | MIT | about 19 / about 5 est. | 32B: **55.44 LB (50 steps)**; 8B: 52.3 V; AndroidWorld 69.0 to 71.6 V | | none |
| alt | EvoCUA-32B / 8B (`meituan/EvoCUA-*-20260105`) | 2026-01 | dense, Qwen3-VL | Apache-2.0 | | **56.73 / 46.06 LB (50 steps)** | | none |
| excl. | Holo4-27B | 2026-09 | 27B dense | **CC-BY-NC-4.0** | 16.9 / 35.9 | OSWorld 85.2 V | | EXL3 only | Non-commercial |
| excl. | CUA-Master-27B (`yxl66666/CUAMaster-27B`) | 2026-10-29 | 27B dense | **no license on card** | | OSWorld-Verified 80.2 V; **MacOSWorld 43.0 V; MacAgentBench 59.9 V** (only open model with macOS numbers) | | none | Licence blocks it; use it as a reference only |
| excl. | UI-Venus-2-9B | 2026-08 | 9B dense | **licence pending** | | OSWorld-Verified 70.8 V; ScreenSpot-Pro 73.0 V; AndroidWorld 80.2 V | | none | |
| excl. | Fara1.5-27B/9B/4B | 2026 | | MIT | | web only (WebVoyager 89.3, Online-Mind2Web 72.3 V); trained to stop at "critical points" | | Fara-7B only | |
| legacy | UI-TARS-1.5-7B / OpenCUA-7B/32B/72B | 2025 | Qwen2.5-VL | Apache / MIT | | 29.6 / 28.8 (50 steps) / 35.7 / 46.1 LB; UI-TARS-1.5-7B MacArena 21.1% | | none | Superseded. UI-TARS-2 weights not released |

Sources for the operator rows:
- Holo4: https://huggingface.co/Hcompany/Holo4-35B-A3B , https://hcompany.ai/newsroom/holo4 , https://assets.hcompanyprod.fr/holo4/hf/holo4-benchmark-table.png
- Holo-3.1: https://huggingface.co/Hcompany/Holo-3.1-35B-A3B
- Holo3: https://huggingface.co/Hcompany/Holo3-35B-A3B
- Mage-VL-Cowork: https://huggingface.co/microsoft/Mage-VL-Cowork
- CUA-Master: https://huggingface.co/yxl66666/CUAMaster-27B
- UI-Venus-2: https://huggingface.co/inclusionAI/UI-Venus-2-9B
- GUI-Owl-1.5: https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct
- EvoCUA: https://huggingface.co/meituan/EvoCUA-32B-20260105
- Fara1.5: https://huggingface.co/microsoft/Fara1.5-27B
- Official LB: http://osworld-v1.xlang.ai/static/data/osworld_verified_results.xlsx

Spark speed for operators:
- No Holo, UI-TARS or GUI-Owl numbers have been measured on the Spark.
- Proxies:
  - Qwen3.5-35B-A3B (same shape as Holo3/3.1/4): 48.6 to 52.9 tg FP8, and 68 to 82 with MTP (https://forums.developer.nvidia.com/t/qwen3-5-35b-a3b-optimizations-on-single-spark/366326).
  - Qwen3-VL-30B-A3B FP8 TTFT with one image: 766 ms in vLLM (https://github.com/mrdbourke/gpu-benchmarking/blob/main/benchmark_analysis/README.md).
- Screenshot prefill is compute-bound on GB10. Expect image TTFT, not decode, to dominate per-step latency for small outputs like a click.

macOS: almost no model has a macOS score. MacArena (arXiv 2606.06560) and macOSWorld (arXiv 2506.04135) show large drops on macOS-native tasks:

| Model | macOSWorld | MacArena | MacAgentBench |
|---|---|---|---|
| UI-TARS-7B-DPO | 4.8% | | |
| UI-TARS-1.5-7B | | 21.1% | |
| Qwen3-VL-4B | | 24.2% | |
| Claude CUA | 37.1% | | |
| OpenAI CUA | | 31.8% | |
| Qwen-CUA (397B, reference) | | | 69.2 (V, read from chart) |

Sources: https://arxiv.org/html/2606.06560v1 , https://arxiv.org/html/2506.04135 , https://raw.githubusercontent.com/xlang-ai/Qwen-CUA/main/assets/readme/main-results.png

### 1c. Shared base (direction A: vision plus coding plus GUI in one model, two LoRAs)

| # | Model | Total / active | Fits as (GB) | Coding | GUI | Tool calling | Spark decode | bf16 LoRA trainable on one Spark? | Abliterated |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Qwen3.8-27B** | 27B dense | FP8 30.9 / Q4 17.7 | SWE-Pro 61.7 V; TB2.1 73.0 V | OSWorld-Verified 84.3 V; AndroidWorld 81.9 V | Toolathlon 67.1 V | about 9 to 16 ceiling; 8.7 measured on the 3.5-27B int4 sibling without speculation | yes (55.6 GB BF16) | yes, full weights |
| 2 | **Qwen3.6-35B-A3B** | 35B / 3B | FP8 37.5 / Q4 22.1 | SWE-V 73.4 V; SWE-Pro 49.5 V; TB2.0 51.5 V | Not on the 3.6 card. Qwen3.5-35B-A3B predecessor: ScreenSpot-Pro 68.6, OSWorld-Verified 54.5, AndroidWorld 71.1 V. **Holo3/3.1/4 prove this base reaches 80+ OSWorld after GUI training** | TAU3 67.2 V; 3.5-35B: BFCL-V4 67.3, TAU2 81.2 V | about 50 tg FP8; 68 to 82 with MTP | yes (Unsloth: about 74 GB for bf16 LoRA) | yes, full weights |
| 3 | **Nemotron 3.5 Super VL 120B-A12B** | 120B / 12B | NVFP4 75.0 | SWE-V 71.1 V; SWE-Pro 51.3 V; TB2.1 62.8 V | OSWorld-Verified 51.1 BF16 / 52.3 NVFP4 V; WebVoyager 77.7 V | tau2 69 to 76 V | about 23 tg (prev gen) | **no** (about 240 GB BF16) | yes, NVFP4-abliterated |
| 4 | Qwen3.5-122B-A10B | 122B / 10B | Q4 77.0 / GPTQ-Int4 78.9 | SWE-V 72.0 V / 71.9 3P; TB2.0 49.4 V | ScreenSpot-Pro 70.4 V; OSWorld-Verified 58.0 V / **51.5 3P** | BFCL-V4 72.2; TAU2 79.5 V | 38.4 tg INT4 + MTP | **no** | yes |
| excl. | Qwen3.8-Flash-Next | 125B / 6B + 51B n-gram embeddings | NVIDIA FP4-Local 102 GB | SWE-V 78.7 3P; TB2.1 85.8 3P; SWE-Pro 62.5 V | OSWorld-Verified 55.8 3P; AndroidWorld 84.5 V | tau2 71.6 / 80.8 / 90.4 3P | about 80 ceiling | no | yes | **Excluded because:** 102 GB leaves no room for KV at our budget; needs a community vLLM container; `qwen-community-1.0` licence requires a separate licence for commercial "AI Work Assistant" coding products |
| excl. | Gemma 4 31B / 26B-A4B | | 18.3 / 17.0 | SWE-V 52.0 / 17.4 (as measured by Qwen) | no GUI numbers | | | | yes | Weak agentic coder |
| excl. | GLM-4.6V (106B-A12B) | | Q4 65.5 | none | OSWorld 37.2 V | | 23 tg on 2x Spark | | GGUF heretic | Outdated |

Sources for the shared-base rows:
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- https://huggingface.co/Qwen/Qwen3.5-122B-A10B (Qwen3.5 family GUI and tool numbers)
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 (also the 3P column for Qwen3.5-122B and Qwen3.8-Flash-Next)
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/LICENSE
- https://unsloth.ai/docs/models/qwen3.6.md
- vLLM LoRA support matrix: https://docs.vllm.ai/en/latest/models/supported_models/ . It marks LoRA as supported for `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration` and `NemotronH_Omni_Reasoning_V3`. On VL models, LoRA applies to the language backbone.

### 1d. Frontier reference rows (closed models, for the gap)

| Benchmark | Best closed | Best open that fits (this report) | Source |
|---|---|---|---|
| OSWorld-Verified (LB, 100 steps) | claude-fable-5 **85.96**; claude-opus-5 83.39; Muse Spark 1.1 80.67 | Holo3-35B-A3B **82.56** / 78.15 (LB) | xlsx above (checked by me) |
| OSWorld-Verified (vendor) | Qwen3.8 Max 86.1; Opus 4.8 83.4; GPT-5.6 Sol 83.0; Gemini 3.6 Flash 83.0 (aggregator) | Qwen3.8-27B 84.3 V | https://huggingface.co/moonshotai/Kimi-K3 , https://llm-stats.com/benchmarks/osworld-verified |
| ScreenSpot-Pro | GPT-6 Astra 92.7; Opus 4.8 87.9; Gemini 3.1 Pro 84.4 (aggregator, vendor-reported) | Holo-3.1-35B-A3B 71.5 V; UI-Venus-2-9B 73.0 V | https://benchlm.ai/md/benchmarks/screenspot-pro.md |
| SWE-bench Verified | Claude Fable 5 about 95.0 (S) | Qwen3.6-27B 77.2 V; Qwen3.6-35B-A3B 73.4 V | https://benchlm.ai/md/models/claude-fable.md |
| SWE-bench Pro | Fable 5 80.3 (S); GPT-5.6 Sol 64.6 (S) | Qwen3.8-27B 61.7 V | https://the-decoder.com/anthropic-releases-claude-fable-5-and-mythos-5-with-major-gains-in-coding-and-science/ , https://benchlm.ai/compare/claude-fable-5-vs-gpt-5-6-sol |
| Terminal-Bench 2.1 | Gemini 3.8 Flash 89.4; Opus 5 89.1; GPT-5.6 Sol 88.8; Fable 5 84.3 V / 80.5 Vals (all S) | Qwen3.8-27B 73.0 V; Nemotron 3.5 Super 62.8 V | https://www.vellum.ai/md/blog/gemini-3-8-flash-benchmarks-explained |
| SWE-rebench (May to Jul 2026) | Fable 5 64.5%; Opus 5 63.4%; GPT-5.6 Sol 62.3% | Qwen3.6-27B 31.2%; Qwen3.6-35B-A3B 24.7% (I) | https://swe-rebench.com/ |
| macOS (MacAgentBench) | GPT-5.5 66.7; Opus 4.8 58.4 (read from Qwen-CUA chart) | CUA-Master-27B 59.9 V (no licence) | Qwen-CUA chart above |
| Aider polyglot | leaderboard frozen 2025-11-20 (gpt-5 high 88.0) | Qwen3-Coder-Next 66.2 V | https://aider.chat/docs/leaderboards/ |
| BFCL v4 / tau2 | not collected for 2026 frontier models | Qwen3.5-122B BFCL-V4 72.2 V, TAU2 79.5 V | |

**Reading:**
- **GUI:** the open-to-frontier gap is small on vendor numbers. Holo3 is about 3 points behind Fable 5 on the official leaderboard.
- **Coding:** the gap is large on independent numbers. SWE-rebench shows about 33 points between Qwen3.6-27B and Fable 5. Vendor SWE-V suggests the gap is under 20 points.
- **Frontier coding numbers are secondary (S):** vendor pages returned 403 to the fetcher.

---

## 2. Memory budgets (top 3 per direction)

### Assumptions

KV bytes per token at BF16 (FP8 KV halves these), computed from each `config.json`:

| Model | Full-attention layers | KV heads × head_dim | KV per token |
|---|---|---|---|
| Qwen3.8-27B | 16 | 4 × 256 | 64 KiB |
| Qwen3.6-35B-A3B / Holo4-35B-A3B | 10 | 2 × 256 | 20 KiB |
| Qwen3-Coder-Next | 12 | 2 × 256 | 24 KiB |
| Qwen3.5-122B-A10B | 12 | 2 × 256 | 24 KiB |
| Nemotron 3.5 Super VL | 8 of 88 | 2 × 128 | 8 KiB |

Fixed allowances:
- Recurrent (DeltaNet/Mamba) state is a few MB per sequence. It is ignored.
- LoRA adapters: about 0.5 to 1.5 GB each (my estimate for rank 32 to 64 on all linear layers).
- Runtime, CUDA graphs and activations: about 4 GB per serving process (estimate).
- Usable pool: about 119.7 GiB. Keep about 10 GB for the OS and harness.

The context column reads `concurrent sequences × context length`.

### Direction A: one base, two LoRAs, one serving process

| Pairing | Base weights | LoRAs | KV | Runtime | **Total** | Decode (single stream) |
|---|---|---|---|---|---|---|
| **A1: Qwen3.8-27B FP8** | 30.9 | 2 × about 1 | coder 1 × 256K (16 GiB) + operator 1 × 64K (4 GiB) = 20 | 4 | **about 57 GB** | about 9 ceiling FP8 (about 16 at Q4); needs MTP or DFlash to be usable |
| **A2: Qwen3.6-35B-A3B FP8** | 37.5 | 2 × about 1 | 1 × 256K (5 GiB) + 1 × 64K (1.3 GiB) = 6.3 | 4 | **about 50 GB** | about 50 tg measured on the 3.5 sibling; 68 to 82 with MTP |
| **A3: Nemotron 3.5 Super VL NVFP4** | 75.0 | 2 × about 1.5 | 1 × 256K (2 GiB) + 1 × 64K (0.5 GiB) = 2.5 | 4 | **about 85 GB** | about 23 tg (prev gen); cannot be LoRA-trained in bf16 on the Spark |

Notes:
- **A2 leaves about 60 GB free.** That is room for bf16 LoRA training on the same box (Unsloth reports about 74 GB for bf16 LoRA on 35B-A3B, so serving and training at once will not fit).
- **A2 also has a ready-made operator path.** Holo4/Holo-3.1 are full fine-tunes of this exact base. We could extract their delta from Qwen3.6-35B-A3B as a low-rank adapter (an approach such as mergekit's LoRA extraction), or train our own operator LoRA. Rank-truncation loss is unmeasured (open question 3).

### Direction B: two specialists, two serving processes

| Pairing | Coder weights | Operator weights | KV | Runtime | **Total** | Notes |
|---|---|---|---|---|---|---|
| **B1: Qwen3.8-27B FP8 + Holo4-35B-A3B FP8** | 30.9 | 37.6 | 16 GiB (256K) + 1.3 GiB (64K) = 17 | 8 | **about 94 GB** | Strongest coder plus strongest Apache operator. Coder decode is slow (about 9 tg; use Q4 at 17.7 GB for about 16, or MTP). |
| **B2: Qwen3.6-35B-A3B (or KAT-Coder-V2.5-Dev) FP8 + Holo4-35B-A3B FP8** | 37.5 | 37.6 | 5 + 1.3 = 6.3 | 8 | **about 90 GB** | Both 3B-active, so both decode fast (about 50 tg each solo, less when decoding at the same time because they share the 273 GB/s). Same architecture and serving stack, and it collapses into A2 later. At Q4 for both, the total is about 52 GB. |
| **B3: Nemotron 3.5 Super VL NVFP4 + Holo4-35B-A3B Q4** | 75.0 | 21.3 (+0.9) | 2 + 1.3 = 3.3 | 8 | **about 108 GB** | At the limit. Only use it if the Nemotron coder wins clearly on our replay. |
| control: Qwen3-Coder-Next Q4 + Holo4-35B-A3B FP8 | 48.5 | 37.6 | 6 (256K) + 1.3 | 8 | about 101 GB | Keeps the current coder, re-tested on a fixed tool-call stack. |

Spark practicalities, all measured or reported in the forum threads cited in `spark_and_abliteration.md`:

1. **Memory setting.** `--gpu-memory-utilization` is a fraction of the total pool (about 119.7 GiB), not of free memory. The two processes must sum below about 0.95. Start the big model first, read actual residency, then size the second (https://dev.to/ric03uec/two-qwen3-models-on-one-dgx-spark-the-residency-math-for-local-llm-coding-5bpj).
2. **Engine images.** Use an SM121 build: eugr `spark-vllm-docker` or `vllm/vllm-openai:cu130-nightly`. The stock image gave 13 vs 49 tg on Qwen3.5-35B-A3B.
3. **Quantization format.** Prefer FP8, INT4 AutoRound/AWQ or MXFP4 GGUF over NVFP4 unless a tuned recipe exists. Qwen3-Coder-Next NVFP4 gave 14.1 tg vs 69 for INT4 AutoRound.
4. **Prefix caching.** Enable `--enable-prefix-caching` explicitly for the hybrid GDN models. Without it, prefill at 32K depth fell from 2162 to 193 tok/s.

---

## 3. Recommendation: benchmark shortlist

### Direction A (shared base)

1. **Qwen3.6-35B-A3B**, the main bet.
   - The only base that is fast (3B active), trainable with bf16 LoRA on the Spark, Apache-2.0, already abliterated in full weights, and supported by vLLM multi-LoRA.
   - Its GUI ceiling is proven by its own fine-tunes: Holo-3.1 scores 80.0 OSWorld V and 71.5 ScreenSpot-Pro V; Holo4 scores 80.8 V.
   - Coding is decent: SWE-V 73.4 V, TB2.0 51.5 V, SWE-rebench 24.7% I. KAT-Coder-V2.5 shows +5 SWE-V and +9 TB2.1 from an agentic SFT+RL on this exact base.
2. **Qwen3.8-27B**, the quality ceiling.
   - Best numbers on both sides: SWE-Pro 61.7 V, TB2.1 73.0 V, OSWorld-Verified 84.3 V.
   - The costs: dense decode (about 9 to 16 tok/s before speculation), an unreproduced OSWorld score and overthinking.
   - Benchmark it with MTP or DFlash on and `reasoning_effort` capped.
3. **Nemotron 3.5 Super VL 120B-A12B NVFP4**, the outside option.
   - Spark-specific recipe, SWE-V 71.1 V, TB2.1 62.8 V, OSWorld-Verified about 52 V.
   - It cannot be LoRA-trained in bf16 on one Spark, so it only makes sense if we train elsewhere.

### Direction B (two specialists)

**Operators.** These three share an architecture, so one serving setup covers all of them.

1. **Holo4-35B-A3B**: newest, 80.8 V.
2. **Holo-3.1-35B-A3B**: 80.0 V, ScreenSpot-Pro 71.5 V, and a full-weight Heretic build already exists.
3. **Holo3-35B-A3B**: the only one verified on the official leaderboard, at 82.56 / 78.15.

Add **Mage-VL-Cowork 4B** (66.0 V) as the latency floor.

**Coders:**
1. **Qwen3.6-35B-A3B and KAT-Coder-V2.5-Dev**: fast.
2. **Qwen3.8-27B**: strongest.
3. **Qwen3.6-27B**: best independent SWE-rebench score among open models that fit, 31.2%.

Plus the **Qwen3-Coder-Next control** on a fixed tool-call stack, to separate model failure from parser failure.

**Pairing to try first:** B2 (Qwen3.6-35B-A3B-family coder + Holo4-35B-A3B). It fits at about 90 GB in FP8, both agents decode at about 50 tok/s, and if direction A wins later it is the same base.

---

## 4. Open questions

1. **Does the 0/7 come from the model or the stack?** Re-run Qwen3-Coder-Next with llama.cpp after PR #20171 (or vLLM with the `qwen3_coder` parser) and the original, non-abliterated chat template. Until then, any comparison against it is confounded.
2. **Does direction A actually give KV sharing?**
   - A LoRA on q/k/v projections changes the cache. To my knowledge vLLM keys its prefix-cache blocks per LoRA adapter, so the two adapters share weights but not KV [UNVERIFIED; check vLLM's block-hash code].
   - Sharing KV across agents needs adapters that leave k/v projections untouched, or a KV-transfer method.
   - The hybrid GDN layers carry a recurrent state, not KV. The earlier agent-communication note found no KV-sharing work tested on hybrid models.
3. **Holo4 as a LoRA.** How much of Holo4's GUI gain survives when its delta from Qwen3.6-35B-A3B is extracted as a rank-64/128 LoRA? This decides whether A2 inherits a strong operator for free.
4. **macOS.** No candidate except the unlicensed CUA-Master-27B has a macOS score, and macOS-native tasks drop sharply in MacArena. We need our own macOS eval slice (macOSWorld or MacArena tasks plus our replay) before trusting any OSWorld number.
5. **Abliteration damage to tool calls and grounding.**
   - Evidence of real damage: an abliterated Llama-3.1-8B broke the tool protocol (124 of 150 tool failures vs 35 for stock; arXiv 2605.19722). Crude uncensors of Qwen3.8-27B corrupted literals in tool calls (Level1Techs).
   - Low-KL Heretic and ArliAI "Derestricted" builds stayed close to stock.
   - The effect on click-coordinate accuracy is unmeasured.
   - Plan: abliterate the BF16 weights ourselves with Heretic, then run a tool-call and grounding regression before switching.
6. **Unreproduced headline numbers.**
   - Qwen3.8-27B OSWorld-Verified 84.3 V vs NVIDIA's 55.8 for the larger Qwen3.8-Flash-Next.
   - Holo4 is not on the official leaderboard.
   - Every Qwen 2026 SWE number uses an internal or Claude Code harness on "refined" task sets.
7. **Running both models at once.** Both models share 273 GB/s. When coder and operator decode simultaneously, each slows down, and no Spark measurement of that exists. Measure it with the B2 pairing.
8. **Licences to confirm before shipping.**
   - Nemotron's OpenMDW-1.1 derivative clauses were not quoted.
   - Qwen3.8-Flash-Next's `qwen-community-1.0` restricts commercial coding assistants.
   - UI-Venus-2 and CUA-Master have no usable licence yet.
