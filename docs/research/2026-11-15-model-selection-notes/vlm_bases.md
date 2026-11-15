# Open-weight VLM generalists as a shared base for the DGX Spark (operator + coder LoRAs)

Research date: 2026-11-15. Scope: open-weight vision-language models that can serve both as a GUI/computer-use operator and as an agentic coder from ONE base with two LoRA adapters, on one DGX Spark (GB10, 128 GB unified, ~273 GB/s).

Conventions
- **V** = vendor-reported (model card of the releasing lab). **3P** = third-party measured (e.g. NVIDIA re-ran competitors in NeMo Gym; Artificial Analysis). **[UNVERIFIED]** = not seen on a page I actually opened.
- Sizes are summed from the Hugging Face file listings (API `?blobs=true`) on 2026-11-15, in GB (1e9 bytes). GGUF sizes include all shards and exclude the mmproj (vision projector, about 0.9 to 1.8 GB extra).
- "Decode ceiling" = 273 GB/s / bytes of active weights at that precision. This is my arithmetic, an upper bound that ignores KV reads, attention, vision and overheads. MTP/speculative decoding can push real numbers above it, and overheads pull them below.
- Benchmark versions matter: Qwen3.5/3.6 report Terminal-Bench **2.0**; Qwen3.8 and NVIDIA report Terminal-Bench **2.1**. Qwen3.8 reports **OSWorld-Verified** (27B) and **OSWorld 2.0** (Flash-Next), which are different benchmarks.

---

## TL;DR ranking for "one shared base" on a Spark

| # | Model | Total / active | Vision | License | Fits? (quant size) | Coding (best numbers) | GUI (best numbers) | Abliterated |
|---|---|---|---|---|---|---|---|---|
| 1 | **Qwen3.8-27B** (Aug 2026) | 27B dense (hybrid GDN) | yes | Apache-2.0 | FP8 30.9 GB; Q4_K_XL 17.7 GB | SWE-Pro 61.7, TB2.1 73.0, LCB v6 90.3 (V) | OSWorld-Verified 84.3, AndroidWorld 81.9 (V) | yes, huihui-ai full weights + many GGUF |
| 2 | **Qwen3.8-Flash-Next** (Aug 2026) | 125B + 51B n-gram emb + 4B MTP, **6B active** | yes | qwen-community-1.0 (restrictive for coding products) | NVIDIA FP4-Local 102 GB (96 GiB, Spark-validated); UD-IQ4_XS 93.7 GB | SWE-V 78.7, TB2.1 85.8 (3P NVIDIA); SWE-Pro 62.5 (V) | OSWorld-Verified 55.8 (3P NVIDIA); AndroidWorld 84.5 (V) | yes, huihui-ai full weights |
| 3 | **Qwen3.6-35B-A3B** (Apr 2026) | 35B / 3B (hybrid GDN MoE) | yes | Apache-2.0 | FP8 37.5 GB; Q4_K_XL 22.4 GB | SWE-V 73.4, TB2.0 51.5 (V) | no GUI agent rows on the 3.6 card; 3.5-35B-A3B: ScreenSpot-Pro 68.6, OSWorld-V 54.5 (V) | yes, huihui-ai full weights |
| 4 | **Qwen3.5-122B-A10B** (Feb 2026) | 122B / 10B (hybrid GDN MoE) | yes | Apache-2.0 | Q4_K_XL 77.0 GB; GPTQ-Int4 78.9 GB; FP8 127 GB (does not fit) | SWE-V 72.0 (V) / 71.9 (3P NVIDIA); TB2.0 49.4 (V) | ScreenSpot-Pro 70.4, OSWorld-V 58.0 (V) / 51.5 (3P NVIDIA) | GGUF and HauhauCS full weights |
| 5 | **Nemotron 3.5 Super VL 120B-A12B** (Oct 2026) | 120B / 12B (Mamba-2/Transformer hybrid MoE) | yes | OpenMDW-1.1 | NVFP4 75.0 GB (Spark recipe on the card) | SWE-V 71.1, SWE-Pro 51.78, TB2.1 60.67 (V) | OSWorld-Verified 51.14 BF16 / 52.3 NVFP4 (V) | yes, armin1 NVFP4-abliterated + heretic GGUF |

Honourable mentions: Qwen3.6-27B (dense, SWE-V 77.2, TB2.0 59.3, AndroidWorld 70.3, Qwen3.8 card lists its OSWorld-Verified as 63.9); Gemma 4 31B / 26B-A4B (Apache, small, weak on agentic coding: SWE-V 52.0 / 17.4 per the Qwen card); Mistral Small 4 119B-A6.5B (Apache, NVFP4 70.8 GB, no GUI or SWE numbers published); GLM-4.6V (MIT, OSWorld 37.2, now outdated).

Doesn't fit / reference only: Qwen3.5-397B-A17B, Step-3.7-Flash (NVFP4 129 GB), MiniMax-M3.1-Flash (252 GB BF16), GLM-5.3-Flash (FP4-Local 177.5 GB), DeepSeek-V4-Flash-Vision-Exp (305B), Kimi K2.5/K2.6/K3/K3.1, Mistral Large 4 (1T), Mistral Medium 3.5 (128B dense: fits at 4-bit but decode ceiling is about 4 tok/s).

---

## 1. Qwen3.8-27B

- Repo: https://huggingface.co/Qwen/Qwen3.8-27B (FP8: https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- Release: repo created 2026-08-05 (HF API); the card's citation says August 2026.
- Params/arch: 27B dense, 64 layers, layout `16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))`, MTP, 262,144 native context (extensible to about 1M). `architectures: Qwen3_5ForConditionalGeneration` (same class as Qwen3.5/3.6, so existing vLLM/llama.cpp/Unsloth paths apply). Source: model card and config.json.
- Vision: yes (native VLM, image and video).
- License: Apache-2.0.
- Sizes: BF16 55.6 GB; official FP8 30.9 GB; unsloth GGUF: Q4_0 17.4, UD-Q4_K_XL 17.7, UD-Q6_K_XL 25.3, Q8_0 29.0 GB, mmproj 0.9 GB. https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
- Decode ceiling: FP8 about 273/28 ≈ 10 tok/s; 4-bit about 273/15.5 ≈ 18 tok/s, before the MTP speedup. **It is dense, so it is the slowest of the top candidates per token.**
- Abliterated: https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated (full weights, image-text-to-text, Apache-2.0), GGUF https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF, plus heretic variants such as https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF

| Benchmark | Score | Type | Scaffold / note | Source |
|---|---|---|---|---|
| SWE-bench Verified | not reported | | | https://huggingface.co/Qwen/Qwen3.8-27B |
| SWE-bench Pro | 61.7 (Qwen3.6-27B 53.5, Opus 4.6 Max 53.4) | V | Claude Code harness, temp 1.0, 256K ctx | same |
| Terminal-Bench 2.1 (Terminus) | 73.0 (Opus 4.6 Max 78.2) | V | | same |
| LiveCodeBench v6 | 90.3 | V | | same |
| QwenSWEBench (internal) | 79.0 | V | avg@3, 8h timeout | same |
| OSWorld-Verified | **84.3** (Qwen3.6-27B 63.9, Opus 4.6 Max 72.7) | V | internal scaffold, not independently reproduced | same |
| AndroidWorld | 81.9 | V | | same |
| WebArena-Verified | 64.8 | V | OSWorld scaffold | same |
| SWE-MM (SWE-bench Multimodal dev) | 38.6 | V | | same |
| ScreenSpot-Pro / ScreenSpot-v2 / OSWorld-G | not reported | | | same |
| BFCL / tau2 | not reported | | | same |
| Artificial Analysis Intelligence Index (xhigh) | 52 | 3P | AA | https://www.qubrid.com/blog/qwen38-27b-benchmarks-official-and-independent-results |
| AA Agentic Index | 51 | 3P | AA | same |

Caveat: the OSWorld-Verified 84.3 beats the Opus 4.6 Max number in the same table, and NVIDIA's independent run of the larger Qwen3.8-Flash-Next scored only 55.8 on OSWorld-Verified (section 2). Treat 84.3 as scaffold-dependent until it is reproduced. Independent coverage at launch was limited to AA and a few others (same Qubrid page).

## 2. Qwen3.8-Flash-Next

- Repo: https://huggingface.co/Qwen/Qwen3.8-Flash-Next (FP8: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8; NVIDIA NVFP4 "FP4-Local": https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
- Release: repo created 2026-08-24 (HF API).
- Params/arch: "125B with 6B activated, plus 51B n-gram embedding and 4B MTP" (about 180B in safetensors). 48 layers, `12 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Qwen Sparse Attention -> MoE))`, 512 experts (10 routed + 1 shared), gated residual streams, n-gram (PLE) embeddings. `architectures: Qwen4ExpForConditionalGeneration` (model_type `qwen4_exp`). Source: model card and config.json.
- Vision: yes.
- License: **qwen-community-1.0**. It allows use, modification and fine-tuning, but (a) products with more than 100M MAU or US$20M monthly revenue must display the model name, and (b) **a licensee running a "Model as a Service" or "AI Work Assistant" business (a product "primarily designed for AI-assisted coding or office productivity") needs a separate license for commercial use. Internal use is exempt.** Source: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/LICENSE. This matters if the operator/coder agent becomes a commercial product.
- Sizes: FP8 185.5 GB (does not fit); **NVIDIA FP4-Local 102.2 GB (card: "approximately 96 GiB on disk")**, validated on DGX Spark with vLLM container `eugr/spark-vllm-b12x:latest`, "1M tokens of FP8 KV Cache with PLE offload" (the n-gram table can go to disk: `VLLM_PLE_TABLE_MEMORY="disk"`). Upstream vLLM support for this NVFP4 format "has not yet landed". unsloth GGUF: UD-IQ4_XS 93.7, UD-Q3_K_XL 89.5, UD-Q4_K_XL 111.3 GB. https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
- Decode ceiling: 6B active at FP4 is about 3.4 GB, so roughly 80 tok/s upper bound (n-gram lookup is a gather, not a GEMM).
- FP4-Local accuracy vs BF16: GPQA-D 91.5 vs 92.2; ToolEvalBench 88.7 vs 89.4 (V, NVIDIA card).
- Abliterated: https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated (full weights), GGUF https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF

| Benchmark | Score | Type | Note | Source |
|---|---|---|---|---|
| SWE-bench Verified | 78.7 | 3P (NVIDIA NeMo Gym) | | https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 |
| SWE-bench Pro | 62.5 (V); 60.37 (3P NVIDIA) | V / 3P | V: Claude Code harness | https://huggingface.co/Qwen/Qwen3.8-Flash-Next ; NVIDIA card above |
| SWE-bench Multilingual | 81.0 (V); 77.10 (3P) | V / 3P | V: mini-SWE-agent | same two |
| Terminal-Bench 2.1 | 85.80 | 3P NVIDIA | | NVIDIA card |
| LiveCodeBench v6 | 91.9 (V); 94.1 (3P) | V / 3P | | both |
| tau2-bench retail / airline / telecom | 71.6 / 80.8 / 90.4 | 3P NVIDIA | | NVIDIA card |
| Toolathlon Verified | 73.5 | V | | Qwen card |
| OSWorld-Verified | 55.8 | 3P NVIDIA | | NVIDIA card |
| OSWorld 2.0 (binary / partial) | 19.4 / 52.3 | V | new benchmark, not OSWorld-Verified | Qwen card |
| AndroidWorld | 84.5 | V | | Qwen card |
| WebVoyager [offline] | 90.39 | 3P NVIDIA | | NVIDIA card |
| ScreenSpot-Pro / v2, BFCL | not reported | | | |

## 3. Qwen3.6-35B-A3B

- Repo: https://huggingface.co/Qwen/Qwen3.6-35B-A3B (FP8: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8). Released 2026-04-15/17 (HF createdAt 2026-04-15; news on 04-17: https://cntechpost.com/2026/04/17/alibaba-open-sources-qwen3-6-35b-a3b-rival-larger-ai-models-coding/).
- Arch: 35B total / 3B active, 40 layers, `10 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Gated Attention -> MoE))`, 256 experts (8 routed + 1 shared), 262K context. Apache-2.0. Vision yes.
- Sizes: BF16 71.9 GB; FP8 37.5 GB; unsloth UD-Q4_K_XL 22.4, MXFP4_MOE 21.7, Q8_0 36.9 GB. https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
- Decode ceiling: FP8 about 273/3 ≈ 90 tok/s; 4-bit about 150 tok/s upper bound.
- Abliterated: https://huggingface.co/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated (full weights), GGUF https://huggingface.co/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF, heretic NVFP4 https://huggingface.co/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4

| Benchmark | Score | Type | Note | Source |
|---|---|---|---|---|
| SWE-bench Verified | 73.4 | V | internal scaffold (bash + file edit), temp 1.0, 200K ctx | https://huggingface.co/Qwen/Qwen3.6-35B-A3B |
| SWE-bench Pro | 49.5 | V | | same |
| SWE-bench Multilingual | 67.2 | V | | same |
| Terminal-Bench 2.0 | 51.5 | V | Harbor/Terminus-2, 3h, avg of 5 | same |
| Terminal-Bench 2.1 | 44.38 | 3P NVIDIA | | https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 |
| LiveCodeBench v6 | 80.4 | V | | Qwen card |
| TAU3-Bench | 67.2 | V | | Qwen card |
| MCP-Atlas | 62.8 (V); 51.2 (3P NVIDIA, public subset) | V / 3P | | both |
| MCPMark | 37.0 | V | | Qwen card |
| WebArena (vision) | 36.60 | 3P NVIDIA | | Bolt card |
| RefCOCO avg / ODInW13 (grounding) | 92.0 / 50.8 | V | | Qwen card |
| OSWorld / ScreenSpot-Pro / AndroidWorld | not on the 3.6 card | | see the Qwen3.5-35B-A3B row in section 6 | |

## 4. Qwen3.6-27B (dense)

- Repo: https://huggingface.co/Qwen/Qwen3.6-27B. HF createdAt 2026-04-21. Apache-2.0. 64 layers, `16 x (3 x GDN -> FFN, 1 x Gated Attn -> FFN)`.
- Sizes: FP8 30.9 GB; unsloth Q4_K_M 16.8, UD-Q4_K_XL 17.6 GB. https://huggingface.co/unsloth/Qwen3.6-27B-GGUF
- Abliterated: https://huggingface.co/huihui-ai/Huihui-Qwen3.6-27B-abliterated
- Superseded by Qwen3.8-27B (same architecture class and size), so only use it if 3.8 has a problem.

| Benchmark | Score | Type | Source |
|---|---|---|---|
| SWE-bench Verified | 77.2 | V | https://huggingface.co/Qwen/Qwen3.6-27B |
| SWE-bench Pro | 53.5 | V | same |
| Terminal-Bench 2.0 | 59.3 | V | same |
| LiveCodeBench v6 | 83.9 | V | same |
| AndroidWorld | 70.3 | V | same |
| OSWorld-Verified | 63.9 | V (from the Qwen3.8-27B card's comparison column) | https://huggingface.co/Qwen/Qwen3.8-27B |
| Terminal-Bench 2.1 | 63.4 | V (Qwen3.8 card) | same |

## 5. Qwen3.5-122B-A10B

- Repo: https://huggingface.co/Qwen/Qwen3.5-122B-A10B. Released 2026-02-24 (https://proxy-ga.blitzz.co/proxy/123456/github.com/QwenLM/Qwen3.6, also reported in https://codersera.com/blog/qwen-3-5-complete-guide-2026/). Apache-2.0. 48 layers, 256 experts (8 routed + 1 shared), 10B active.
- Sizes: FP8 127.2 GB (does not fit with KV); official GPTQ-Int4 78.9 GB (https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4); unsloth UD-Q4_K_XL 77.0, MXFP4_MOE 74.7, UD-IQ4_XS 60.2 GB (https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF).
- Decode ceiling at 4-bit: 10B ≈ 5.6 GB, so about 49 tok/s.
- Abliterated: GGUF https://huggingface.co/huihui-ai/Huihui-Qwen3.5-122B-A10B-abliterated-GGUF ; full-weight uncensored https://huggingface.co/HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive ; heretic https://huggingface.co/mradermacher/Qwen3.5-122B-A10B-heretic-GGUF

| Benchmark | Score | Type | Source |
|---|---|---|---|
| SWE-bench Verified | 72.0 (V); 71.9 (3P NVIDIA) | V / 3P | https://huggingface.co/Qwen/Qwen3.5-122B-A10B ; https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 |
| SWE-bench Pro | 44.9 | 3P NVIDIA | NVIDIA card |
| Terminal-Bench 2.0 | 49.4 | V | Qwen card |
| Terminal-Bench 2.1 | 48.70 | 3P NVIDIA | NVIDIA card |
| LiveCodeBench v6 | 78.9 (V); 76.7 (3P) | V / 3P | both |
| BFCL-V4 | 72.2 | V | Qwen card |
| TAU2-Bench | 79.5 (V); retail 55.4 / airline 63.5 / telecom 87.2 (3P) | V / 3P | both |
| ScreenSpot-Pro | 70.4 | V | Qwen card |
| OSWorld-Verified | 58.0 (V); **51.5 (3P NVIDIA)** | V / 3P | both |
| AndroidWorld | 66.4 | V | Qwen card |

## 6. Qwen3.5-35B-A3B / Qwen3.5-27B (Feb 2026, superseded by 3.6/3.8 but the only Qwen cards with ScreenSpot-Pro)

Source for all: https://huggingface.co/Qwen/Qwen3.5-122B-A10B (V).

| Benchmark | Qwen3.5-35B-A3B | Qwen3.5-27B |
|---|---|---|
| SWE-bench Verified | 69.2 (the 3.6 card later lists 70.0) | 72.4 (the 3.6 card later lists 75.0) |
| Terminal-Bench 2.0 | 40.5 | 41.6 |
| LiveCodeBench v6 | 74.6 | 80.7 |
| BFCL-V4 | 67.3 | 68.5 |
| TAU2-Bench | 81.2 | 79.0 |
| ScreenSpot-Pro | 68.6 | 70.3 |
| OSWorld-Verified | 54.5 | 56.2 |
| AndroidWorld | 71.1 | 64.2 |

Note: Qwen's own SWE-V numbers for the same model differ between cards (re-evaluation); see https://huggingface.co/Qwen/Qwen3.6-35B-A3B. Abliterated: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated , https://huggingface.co/huihui-ai/Huihui-Qwen3.5-27B-abliterated

## 7. Qwen3.5-397B-A17B (reference only, does not fit)

- https://huggingface.co/Qwen/Qwen3.5-397B-A17B, released 2026-02-16, Apache-2.0, 60 layers, 512 experts (10 + 1 shared). A 4-bit build is about 200+ GB [UNVERIFIED exact size; GPTQ-Int4 repo exists: https://huggingface.co/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4].
- V scores: SWE-V 76.4, SWE-Multilingual 69.3, TB2 52.5, LCB v6 83.6, BFCL-V4 72.9, TAU2 86.7, ScreenSpot-Pro 65.6, OSWorld-Verified 62.2, AndroidWorld 66.8 (same card). Note that the Qwen3.6/3.8 cards list its SWE-V as 76.2.
- Abliterated: https://huggingface.co/Rick02280/Huihui-Qwen3.5-397B-A17B-abliterated-GGUF

## 8. NVIDIA Nemotron 3.5 Super VL 120B-A12B

- Repos: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 , NVFP4 https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 . HF publication 10/15/2026 (card).
- Arch: "Mamba-2-Transformer hybrid Mixture of Experts (MoE) with Multi-Token Prediction", 512 routed experts per MoE layer, C-RADIO vision encoder, up to 1M context. `architectures: NemotronH_Omni_Reasoning_V3`.
- License: OpenMDW-1.1, "ready for commercial or non-commercial use" (card). Derivative/fine-tune clauses not quoted on the card [UNVERIFIED details; see openmdw.ai/license/1-1/].
- Sizes: BF16 about 124B params; **NVFP4 75.0 GB**; unsloth UD-Q4_K_XL 86.9 GB + mmproj 1.7 GB (https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF). The NVFP4 card has a "DGX Spark (GB10) only" vLLM recipe (`--max-num-seqs 1`, MTP 7 speculative tokens) but no tok/s figure.
- Decode ceiling at NVFP4: 12B ≈ 6.8 GB, so about 40 tok/s.
- Abliterated: https://huggingface.co/armin1/Nemotron-3.5-Super-VL-120B-A12B-NVFP4-abliterated ; heretic https://huggingface.co/spiritfather/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-heretic-ara-GGUF

| Benchmark | Score | Type | Source |
|---|---|---|---|
| SWE-bench Verified | 71.10 | V | BF16 card |
| SWE-bench Pro | 51.78 | V | same |
| SWE-bench Multilingual | 67.72 | V | same |
| Terminal-Bench 2.1 | 60.67 | V | same |
| LiveCodeBench v6 | 88.22 | V | same |
| tau2 retail / airline / telecom | 69.24 / 76.50 / 73.19 | V | same |
| OSWorld-Verified | 51.14 (BF16); 52.3 (NVFP4) | V | BF16 and NVFP4 cards |
| WebVoyager [offline] | 77.72 | V | BF16 card |
| ScreenSpot-Pro / v2, AndroidWorld, BFCL | not reported | | |

## 9. NVIDIA Nemotron 3.5 Bolt 74B-A9B (smaller sibling)

- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 (BF16 149.8 GB), plus a 24 GB IQ2-class build https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB (24.1 GB). Mamba-2/Transformer hybrid LatentMoE, C-RADIO v4-H, image and video, 256K context, OpenMDW-1.1. DGX Spark is listed as an inference target.
- V (BF16): Terminal-Bench 2.1 53.93, MCP-Atlas 55.60, WebArena (vision) 50.25, BrowseComp 70.00 (same card). No SWE-bench, OSWorld or ScreenSpot numbers.
- Abliterated: https://huggingface.co/BoldingBuilds/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-Abliterated-24GB-GGUF

## 10. Gemma 4 (31B dense, 26B-A4B MoE)

- https://huggingface.co/google/gemma-4-31B-it , https://huggingface.co/google/gemma-4-26B-A4B-it (HF createdAt 2026-03-11). Apache-2.0. 31B: 30.7B, 60 layers, 256K. 26B-A4B: 25.2B total / 3.8B active, 128 experts (8 active + 1 shared). Text and image (~550M vision encoder).
- Sizes: 31B BF16 62.5 GB, Q4_K_M 18.3 GB; 26B-A4B UD-Q4_K_XL 17.0 GB, MXFP4_MOE 16.6 GB (unsloth GGUF repos).
- V (Google card): LCB v6 80.0% / 77.1%; Tau2 (avg of 3) 76.9% / 68.2%. No SWE, Terminal, OSWorld or ScreenSpot numbers.
- From the Qwen3.6 card (V by Qwen, i.e. a competitor's measurement): SWE-V 52.0 (31B) / 17.4 (26B-A4B); Terminal-Bench 2.0 42.9 / 34.2. https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- 3P NVIDIA (26B-A4B): Terminal-Bench 2.1 37.22; WebArena (vision) 38.2. https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16
- Abliterated: https://huggingface.co/huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2 , https://huggingface.co/huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated
- Verdict: weak agentic coder, no GUI evidence. Not a good shared base.

## 11. GLM-4.6V / GLM-4.6V-Flash (Zhipu)

- https://huggingface.co/zai-org/GLM-4.6V (HF 2025-12-07), MIT, 106B / A12B MoE (glm4v_moe), 128K. Flash = 9B dense.
- Sizes: FP8 110.0 GB (too tight); unsloth UD-Q4_K_XL 65.5 GB, Q4_K_M 70.6 GB (+1.8 GB mmproj). https://huggingface.co/unsloth/GLM-4.6V-GGUF
- V (benchmark image https://raw.githubusercontent.com/zai-org/GLM-V/refs/heads/main/resources/bench_46v.jpeg): OSWorld 37.2 (Flash 21.1), AndroidWorld 57.0 (42.7), WebVoyager 81.0 (71.8), Design2Code 88.6 (69.8). No SWE-bench, Terminal-Bench or ScreenSpot-Pro numbers.
- Abliterated: only heretic GGUF for the 106B (https://huggingface.co/mradermacher/GLM-4.6V-heretic-GGUF); Flash: https://huggingface.co/AliBilge/Huihui-GLM-4.6V-Flash-abliterated
- Newer: GLM-5.3-Flash (https://huggingface.co/zai-org/GLM-5.3-Flash, MIT, 321B, Aug 2026; NVIDIA FP4-Local 177.5 GB, does not fit). 3P NVIDIA numbers: SWE-V 79.10, TB2.1 82.70, but OSWorld-Verified only 14.78 (NVIDIA Super VL card).

## 12. Mistral

- Mistral Small 4 119B-2603: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603 . 119B / 6.5B active, 128 experts (4 active), image input, 256K, Apache-2.0. NVFP4 70.8 GB (https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4). The card text only gives GPQA-D 71.2; no SWE, Terminal, OSWorld or ScreenSpot. Heretic: https://huggingface.co/darkc0de/Mistral-Small-4-119B-2603-heretic , NVFP4 https://huggingface.co/GulfCoastAI/Mistral-Small-4-119B-Heretic-NVFP4
- Mistral Medium 3.5 128B: https://huggingface.co/mistralai/Mistral-Medium-3.5-128B . Dense 128B, image input, Modified MIT (large-revenue exception). SWE-bench Verified 77.6% (V). Dense 128B at 4-bit is about 64 GB/token, so the decode ceiling is about 4 tok/s. Too slow.
- Mistral Large 4 1T-A52B (Sep 2026): 1,060 GB BF16. Does not fit.

## 13. Others checked (legacy or no fit)

- **Qwen3-VL-235B-A22B** (2025): ScreenSpot-Pro 62.0, OSWorld-Verified 38.1, AndroidWorld 63.7 (V, from the Qwen3.5 cards' comparison column, https://huggingface.co/Qwen/Qwen3.5-397B-A17B). Superseded by Qwen3.5+, which "outperform Qwen3-VL" per the Qwen3.6 card. Qwen3-VL-30B-A3B / 32B: abliterated exist (https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated , https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-32B-Instruct-abliterated). Individual scores not collected because they are superseded.
- **Ling-3.0-flash-VL** (inclusionAI, MIT): 124B / 5.5B active, KDA + Gated MLA hybrid (5:1), fp4 69.6 GB / int4 76.1 GB. Its benchmark image shows WebVoyager 90.83, Vision2Web 57.69, ClawEval-MM 59.9 (Qwen3.8-27B in the same image: 90.80 / 62.90 / 56.90). No SWE, OSWorld or ScreenSpot numbers. https://huggingface.co/inclusionAI/Ling-3.0-flash-VL-fp8 . No abliterated variant found.
- **Keye-VL-2.0-30B-A3B** (Kwai, Apache, May 2026): video-focused, no coding or GUI numbers on the card; Q4_K_M 18.6 GB. https://huggingface.co/Kwai-Keye/Keye-VL-2.0-30B-A3B
- **Step-3.7-Flash** (Apache, 201B): NVFP4 129.2 GB, IQ4_XS 105 GB + 4 GB mmproj. Does not fit with KV headroom.
- **MiniMax-M3.1-Flash** (220B, minimax-community licence): 252 GB BF16. Does not fit.
- **Kimi K2.5 / K2.6 (1T), K3 (2.8T)**: reference only. The Qwen3.5-397B card lists K2.5: SWE-V 76.8, OSWorld-Verified 63.3, TB2 50.8.
- **InternVL3.5, Kimi-VL-A3B, Llama 4, Phi-4-multimodal, MiMo-VL, ERNIE-4.5-VL**: no 2026 refresh that beats the Qwen line on both coding and GUI. Not scored in detail.
- **DeepSeek-V4-Flash-Vision-Exp** (MIT, 305B, Aug 2026): too big.

---

## LoRA training + multi-LoRA serving notes

- **vLLM**: the supported-models table marks LoRA ✅ for `Qwen3_5ForConditionalGeneration` (this covers Qwen3.5, Qwen3.6 and Qwen3.8-27B), `Qwen3_5MoeForConditionalGeneration`, `Qwen3NextForCausalLM`, `Qwen3VL(Moe)ForConditionalGeneration`, `NemotronH_Omni_Reasoning_V3` (Nemotron 3.5 Super VL; PP blank), `Gemma4ForConditionalGeneration`, `Glm4vMoeForConditionalGeneration` and InternVL. On multimodal models LoRA applies to the language backbone, and vLLM "experimentally supports adding LoRA to the tower and connector modules for some" models. https://docs.vllm.ai/en/latest/models/supported_models/
- **Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`)** is not yet listed in that table. In vLLM main, `Qwen4ExpForCausalLM` declares `SupportsLoRA`, and the VL class subclasses `Qwen3_5ForConditionalGeneration` (https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/qwen4_exp/nvidia/model.py). The NVIDIA FP4-Local checkpoint, however, needs a community container. **Multi-LoRA on top of the NVFP4/PLE-offload path is [UNVERIFIED]: test before you commit.**
- **llama.cpp server**: `--lora` loads several adapters, `--lora-init-without-apply` defers them, and a per-request `"lora": [{"id":0,"scale":...}]` selects one; "requests with different LoRA configurations will not be batched together". https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md . Fine for one operator and one coder alternating, worse for concurrent use.
- **SGLang**: `--enable-lora`, `--lora-paths`, `--max-loras-per-batch` (default 8), per-request `lora_path`, triton/csgmv backends. The docs list no restrictions for MoE, hybrid-GDN or VL models, but they do not confirm support either [UNVERIFIED for Qwen3.5-class]. https://docs.sglang.io/advanced_features/lora.html
- **Training**: Unsloth supports Qwen3.5 (0.8B to 122B-A10B) vision and text LoRA, and says Qwen3.6 follows the 3.5 guide. It does **not recommend QLoRA 4-bit for the Qwen3.5 MoE or dense models** (quantization error) and recommends bf16 LoRA (35B-A3B bf16 LoRA about 74 GB VRAM, which fits in the Spark's 128 GB). https://unsloth.ai/docs/models/qwen3.6.md and https://unsloth.ai/docs/de/modelle/qwen3.5/fine-tune (via search summary). NVIDIA NeMo AutoModel lists Qwen3.6-27B under VLM coverage: https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/qwen/Qwen3.6-27B . For Nemotron 3.5 the card names SFT/RL via NeMo RL/Gym but not LoRA by name.
- Implication: bf16 LoRA training of a 27B dense model (about 55 GB of weights) or a 35B-A3B (about 72 GB) fits on one Spark. Qwen3.8-Flash-Next (180B BF16) and Qwen3.5-122B (about 250 GB BF16) **cannot be LoRA-trained in bf16 on one Spark**. They would need QLoRA, which Unsloth discourages for this family, or training elsewhere.

## Caveats

1. Almost all GUI numbers are vendor-reported with in-house scaffolds. The only cross-lab OSWorld-Verified reproductions I found are NVIDIA's: Qwen3.5-122B 51.5 vs Qwen's 58.0, and Qwen3.8-Flash-Next 55.8. Qwen3.8-27B's 84.3 has not been reproduced.
2. Mac-specific GUI ability (the target is macOS) is basically unmeasured. OSWorld is Ubuntu-centric. Qwen's RecreationBench includes macOS (card description) but reports one aggregate score.
3. Qwen re-scores its own older models between cards (e.g. Qwen3.5-27B SWE-V 72.4 vs 75.0). Compare models only within one table.
4. Dense 27B decode on GB10 is roughly 10 to 18 tok/s ceiling before MTP, versus about 80 to 150 for the 3B to 6B active MoEs. For long agent loops (many screenshots, long coding traces) the MoE options are much faster.
5. The Qwen3.8-Flash-Next licence's "AI Work Assistant" clause can block commercial coding or office-agent products without a separate deal from Qwen. Qwen3.8-27B and Qwen3.6 are plain Apache-2.0.
6. Abliteration on these VLMs edits the text decoder; repos are tagged image-text-to-text and keep the vision tower, but whether GUI and coding scores survive abliteration is unmeasured [UNVERIFIED]. Applying a LoRA trained on the original base to an abliterated base is also untested.
