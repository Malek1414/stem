# DGX Spark (GB10) measured throughput + abliteration landscape

Research date: 2026-11-15. Web research only. Every number below comes from a page I fetched in this session; the URL is on the row. Anything I could not check on a fetched page is marked [UNVERIFIED].
Abbreviations: pp = prefill / prompt processing tok/s, tg = decode / token generation tok/s, d = context depth already in the KV cache (llama-bench/llama-benchy), c = concurrency, BS = batch size.

## Key takeaways (read first)
- **Decode is bound by memory bandwidth (273 GB/s).** At batch 1, tg is set by the bytes of *active* parameters read per token. Low-active MoEs (A3B / gpt-oss) are the only ones that feel interactive: about 40-85 tg on a single stream. A 27B dense model sits at about 8-15 tg without speculative decoding, and Llama-3.1-70B FP8 at 2.7 tg.
- **The software stack matters more than the quant.** Stock vLLM images were missing native SM121 cubins: Qwen3.5-35B-A3B ran at 13.3 tg stock vs 48.6 with an SM121 build. NVFP4 often falls back to Marlin W4A16 or emulation and is *slower* than INT4 AutoRound or FP8 on GB10: Qwen3-Coder-Next NVFP4 gave 14.1 tg vs 69 tg for INT4 AutoRound. One harness saw about 1.1 tg on vanilla vLLM emulation vs 77.1 with FlashInfer CUTLASS.
- **Batching is where vLLM pays off.** One harness got Qwen3-Coder-Next 74.8 tg single → 289 tok/s aggregate at c=8. Ollama does not batch: its c=8 total matches its single-stream figure.
- **Hybrid Gated DeltaNet models (Qwen3-Next, Qwen3-Coder-Next, Qwen3.5) run on GB10 in vLLM and llama.cpp,** but they have had bugs: Triton allocator crashes under Ray, GDN-path crashes in cutlass_scaled_mm with FP8+MTP, and prefix caching disabled for hybrid/mamba models in older vLLM. Enable `--enable-prefix-caching` explicitly for coding agents; without it pp2048@d32768 collapsed from 2162 to 193 tok/s.
- **Two models on one Spark is workable, but `--gpu-memory-utilization` is a fraction of the TOTAL pool (~119.7 GiB), not of free memory.** The fractions of co-resident processes must sum below about 0.95. Load the big model first, read actual residency, then size the second one.

---

## PART 1: Measured throughput on DGX Spark / GB10 (single Spark unless noted)

### 1a. Coder-class candidates

| Model | Engine | Quant | Context / depth | pp tok/s | tg tok/s | Conc. | Date | URL |
|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-Next (80B-A3B) | vLLM (eugr spark-vllm-docker), flashinfer + prefix caching | FP8 (official) | d0 / d32768 | 3006.5 / 2162.0 (pp2048) | 42.68 / 37.20 (tg128) | 1 | 2026-02-03 | https://forums.developer.nvidia.com/t/how-to-run-qwen3-coder-next-on-spark/359571 |
| Qwen3-Coder-Next | same, WITHOUT prefix caching | FP8 | d0 / d32768 | 3743.5 / 193.1 | 44.63 / 38.47 | 1 | 2026-02-03 | same as above |
| Qwen3-Coder-Next | vLLM, **2x Spark** | FP8 | d0 / d100000 | 3396.6 / 1189.2 | 43.98 / 29.41 | 1 | 2026-02-05 | https://forums.developer.nvidia.com/t/how-to-run-qwen3-coder-next-on-spark/359571?page=2 |
| Qwen3-Coder-Next | vLLM | unsloth FP8-Dynamic | d0 / d16384 | 2441.7 / 1926.1 | 32.07 / 29.91 | 1 | 2026-02-04 | same as above (page 2) |
| Qwen3-Coder-Next | vLLM 0.17.0 | RedHatAI NVFP4 | n/a | n/a | 14.1 | 1 | 2026-03-11 | https://forums.developer.nvidia.com/t/slow-performance-for-redhatai-qwen3-coder-next-nvfp4/363129 |
| Qwen3-Coder-Next | vLLM | Intel int4-AutoRound | n/a | n/a | 69 (quoted, no detail) | ? | 2026-03-11 | same as above |
| Qwen3-Coder-Next | Ollama | Q4 | 400 max_tokens | TTFT 215 ms | 59.2 single / 60.4 total @c8 | 1, 8 | 2026-08-10 | https://forums.developer.nvidia.com/t/measured-inference-benchmarks-on-a-single-dgx-spark-same-harness-across-ollama-llama-cpp-and-vllm-notes-data-published/379766 |
| Qwen3-Coder-Next | vLLM 0.26.1rc1 | 4-bit | 400 max_tokens | TTFT 104 ms | 74.8 single / **289.0 total @c8** | 1, 8 | 2026-08-10 | same as above |
| Qwen3-Coder-30B-A3B | llama.cpp b7941 | Q8_0 | d0 / d32768 | 2986.97 / 1348.17 (pp2048) | 61.06 / 30.21 (tg32) | 1 | ~2026-02 (build 7941) | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/benches/dgx-spark/dgx-spark.md |
| Qwen3-Coder-30B-A3B (as "qwen3moe 30B.A3B") | llama.cpp (first run, Gerganov) | Q8_0 | d0 | 2916.25 | 47.08 | 1 | 2025-10-15 | https://www.hardware-corner.net/first-dgx-spark-llm-benchmarks/ |
| Qwen3-30B-A3B | Ollama / vLLM 4-bit | Q4 / 4-bit | 400 max_tokens | TTFT 161 ms (Ollama) | 85.2 / 77.1 single; 87.5 / **313.4 total @c8** | 1, 8 | 2026-08-10 | https://forums.developer.nvidia.com/t/measured-inference-benchmarks-on-a-single-dgx-spark-same-harness-across-ollama-llama-cpp-and-vllm-notes-data-published/379766 |
| Qwen3-Next-80B-A3B-Instruct | vLLM | FP8 | n/a | n/a | 44 (peak output) | 1 | 2025-12-30 | https://forums.developer.nvidia.com/t/from-20-to-35-tps-on-qwen3-next-nvfp4-w-flashinfer-12-1f/356153 |
| Qwen3-Next-80B-A3B-Instruct | vLLM + flashinfer 0.6.0rc2 | NVFP4 | n/a | n/a | 35-36 single; ~680 system tok/s @100 conc | 1, 100 | 2025-12-31 / 2026-01-06 | same as above |
| Qwen3.5-35B-A3B | vLLM v0.17.0rc1 stock image vs SM121 build | FP8 (on the fly) | n/a | n/a | **13.3 stock → 48.6** | 1 | 2026-03-30 | https://forums.developer.nvidia.com/t/dgx-spark-13-49-tok-s-with-qwen3-5-35b-native-sm121-kernel-build-guide/365083 |
| Qwen3.5-35B-A3B | vLLM 0.18.3 nightly (llama-benchy) | FP8 | d0 | 4240.97 (pp2048, high variance) | 52.85 (tg32) | 1 | 2026-04-01 | same as above |
| Qwen3.5-35B-A3B | vLLM, hybrid INT4+FP8 checkpoint, MTP=2 | INT4+FP8 | 256-2048 out tok | n/a | 104-125 single (wall-clock, MTP); 157.9 total @c4 | 1, 4 | 2026-04-12 | https://forums.developer.nvidia.com/t/qwen3-5-35b-a3b-optimizations-on-single-spark/366326 |
| Qwen3.5-35B-A3B | vLLM, DFlash drafter | int4-AutoRound | 256-2048 out tok | ~5800 (pp1024) | 117-169 single (wall-clock) | 1 | 2026-04-13 | same as above |
| Qwen3.5-35B-A3B | vLLM, native FP8 + MTP=2 | FP8 | 256-2048 out tok | n/a | 68-82 single; 138 total @c4 (another user: 185 @c4) | 1, 4 | 2026-04-13 | same as above |
| Qwen3.5-35B-A3B | vLLM 0.17 + MXFP4 patches, **2x Spark TP=2** | MXFP4 (online) | 1024 in / 128 out | n/a | 42.85 vanilla → 70.68 patched | 1 | 2026-03-09 | https://forums.developer.nvidia.com/t/vllm-0-17-0-mxfp4-patches-for-dgx-spark-qwen3-5-35b-a3b-70-tok-s-gpt-oss-120b-80-tok-s-tp-2/362824 |
| Qwen3.5-122B-A10B | vLLM 0.19, Intel int4-AutoRound | INT4 | 256-2048 out tok | n/a | 28.3 baseline → 30.8 hybrid INT4+FP8 → **38.4 hybrid+MTP-1** (thread title later claims "up to 51") | 1 | 2026-04-05 | https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-on-single-spark-up-to-51-tok-s-v2-1-patches-quick-start-benchmark/365639 |
| Qwen3.5-122B-A10B | llama-benchy raw decode steps (same thread) | INT4 | d2048 / d12000 | 2089.65 (pp512) / 2153.08 (pp8192) | 21.05 / 19.39 | 1 | 2026-04 | same as above |
| Qwen3.5-122B-A10B | vLLM | RedHatAI NVFP4 | n/a | n/a | 16.6 (rejected, slower than INT4) | 1 | 2026-04-05 | same as above |
| Qwen3.5-122B-A10B | vLLM, **2x Spark** | FP8 | d0 | 2384.91 | 21.60 | 1 | 2026-02-25 | https://forums.developer.nvidia.com/t/qwen3-5-397b-a17b-dgx-spark-duo/360780 |
| Qwen3.5-397B-A17B | llama.cpp RPC, **2x Spark** | UD-Q4_K_XL | n/a | n/a | ~11 (8 with MXFP4_MOE) | 1 | 2026-02/03 | same as above. Does NOT fit one Spark: NVFP4 weights ~240 GB |
| Qwen3.5-27B (dense) | vLLM, llama-benchy | int4-AutoRound | d0 / d32768 | 1004.03 (pp2048) | 8.71 / 7.45 (tg32) | 1 | 2026-04-08 | https://forums.developer.nvidia.com/t/qwen3-5-27b-optimisation-thread-starting-at-30-t-s-tp-1/366009 |
| Qwen3.5-27B | vLLM + DFlash (15-16 spec tokens) | int4-AutoRound | 256-2048 out tok | n/a | 36-71 wall-clock (code best); real-world "15-20" on complex work | 1 | 2026-04-08..12 | same as above |
| gpt-oss-120b | llama.cpp b7941 | MXFP4 | d0 / d32768 | 2443.91 / 1567.08 | 58.72 / 42.76 | 1 | ~2026-02 | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/benches/dgx-spark/dgx-spark.md |
| gpt-oss-120b | llama.cpp (Gerganov first run) | MXFP4 | d0 | 1723.07 | 38.55 | 1 | 2025-10-15 | https://www.hardware-corner.net/first-dgx-spark-llm-benchmarks/ |
| gpt-oss-120b | llama.cpp (NVIDIA blog), ISL/OSL 2048/128 | MXFP4 | 2048 | 1725.47 | 55.37 | BS1 | 2025-10-24 | https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks |
| gpt-oss-120b | Ollama v0.12.6 | MXFP4 | 500 out tok | 1169 | 41.14 | 1 | 2025-10-23 | https://registry.ollama.ai/blog/nvidia-spark-performance |
| gpt-oss-120b | llama.cpp, eugr: no UMA vs `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` | MXFP4 | d0 | 1737 vs 1672 | 45.87 vs 40.61 | 1 | 2025-10 | https://github.com/ggml-org/llama.cpp/discussions/16578 |
| gpt-oss-120b | Ollama / vLLM | MXFP4 | 400 out tok | TTFT 449 / 131 ms | 42.1 / 60.7 single; vLLM 116-153 total @c8 | 1, 8 | 2026-08-10 | https://forums.developer.nvidia.com/t/measured-inference-benchmarks-on-a-single-dgx-spark-same-harness-across-ollama-llama-cpp-and-vllm-notes-data-published/379766 |
| gpt-oss-120b | vLLM (NVIDIA container) | MXFP4 | ~1000 in / ~300 out | TTFT 578 ms | 116 total @c10 | 10 | 2026-02-04 | https://github.com/mrdbourke/gpu-benchmarking/blob/main/benchmark_analysis/README.md |
| gpt-oss-120b | vLLM, batch 64 | MXFP4 | n/a | n/a | ~1300 system tok/s (max 1385) | 64 | 2026-01-07 | https://forums.developer.nvidia.com/t/from-20-to-35-tps-on-qwen3-next-nvfp4-w-flashinfer-12-1f/356153 |
| gpt-oss-120b | vLLM + MXFP4 patches, **2x Spark TP=2** | MXFP4 | d0 / d100000 | 6069 / 2019 | 81.48 / 59.56 | 1 | 2026-03-11 | https://forums.developer.nvidia.com/t/vllm-0-17-0-mxfp4-patches-for-dgx-spark-qwen3-5-35b-a3b-70-tok-s-gpt-oss-120b-80-tok-s-tp-2/362824 |
| gpt-oss-20b | llama.cpp b7941 | MXFP4 | d0 / d32768 | 4505.82 / 2689.42 | 83.43 / 61.65 | 1 | ~2026-02 | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/benches/dgx-spark/dgx-spark.md |
| gpt-oss-20b | Ollama (LMSYS) | MXFP4 | n/a | 2053 | 49.7 | 1 | 2025-10-13 | https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark |
| gpt-oss-20b | Ollama v0.12.6 | MXFP4 | 500 out | 3224 | 58.27 | 1 | 2025-10-23 | https://registry.ollama.ai/blog/nvidia-spark-performance |
| gpt-oss-20b | vLLM (NVIDIA container) | MXFP4 | ~1000/300 | TTFT 283 ms | 230 total @c10 | 10 | 2026-02-04 | https://github.com/mrdbourke/gpu-benchmarking/blob/main/benchmark_analysis/README.md |
| GLM-4.7-Flash (30B-A3B) | llama.cpp b7941 | Q8_0 | d0 / d32768 | 2364.18 / 567.20 | 48.68 / 32.65 | 1 | ~2026-02 | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/benches/dgx-spark/dgx-spark.md |
| GLM-4.5-Air (106B-A12B, "GLM4MoE") | llama.cpp (first run, Oct 2025 build) | Q4_K | d0 | 817.00 | 18.45 | 1 | 2025-10-15 | https://www.hardware-corner.net/first-dgx-spark-llm-benchmarks/ |
| MiniMax-M2.5-REAP-172B-A10B | vLLM (eugr docker), Marlin NVFP4 | NVFP4 | d0 / d128000 | 2349.17 / 514.16 | 27.60 / 9.71 | 1 | 2026-03-23 | https://forums.developer.nvidia.com/t/minimax-2-5-reap-nvfp4-on-single-dgx-spark/361248 |
| MiniMax-M2.5-REAP-139B-A10B | vLLM 0.14 (scitrera) | NVFP4 | d0 / d65535 | 3342.54 / 571.21 | 16.71 / 10.38 | 1 | 2026-02-21 | same as above |
| MiniMax-M2.5-REAP-172B-A10B | llama.cpp b8123 | i1-Q4_K_S | d0 | 642.37 | 26.15 | 1 | 2026-02-27 | same as above |
| Nemotron-3-Super-120B-A12B | vLLM nightly, Marlin + MTP | NVFP4 | d0..d100000, tg128 | n/a | 23.45 (flat to 100k) | 1 | 2026-05-13 | https://forums.developer.nvidia.com/t/nemotron-3-super-120b-a12b-nvfp4-on-single-dgx-spark-23-45-tok-s-spark-arena-com-benhmarks/370070 |
| Nemotron-3-Super-120B-A12B | vLLM cu130-nightly (official vLLM blog) | NVFP4 | 1.8k-7.2k prompt | 1636-1884 | 22.7-23.7 | ≤4 (max-num-seqs) | 2026-06-01 | https://vllm.ai/blog/2026-06-01-vllm-dgx-spark |
| Nemotron-3-Nano-30B-A3B | vLLM (NVIDIA container) | FP8 | ~1000/300 | TTFT 611 ms | 230 total @c10 | 10 | 2026-02-04 | https://github.com/mrdbourke/gpu-benchmarking/blob/main/benchmark_analysis/README.md |
| Nemotron-3-Nano-30B-A3B | llama.cpp | UD-Q8_K_XL | n/a | 1037.78 | 47.56 | 1 | 2026-02-04 | same as above |
| Devstral 2 (123B) / Devstral Small 2 (24B) | none found | n/a | n/a | n/a | n/a | n/a | n/a | No measured Spark numbers found. llmconfigurator gives *estimates* only (~15 tg Small 2 Q4_K_M): https://llmconfigurator.com/en/can-i-run/devstral-small-2-24b-on-nvidia-dgx-spark [estimate, not a measurement] |

### 1b. Computer-use / VLM candidates

| Model | Engine | Quant | Context | pp tok/s | tg tok/s | Conc. | Date | URL |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-30B-A3B-Instruct | vLLM (NVIDIA container) | FP8 | ~1000 in / ~300 out, 0 images | TTFT 338 ms | 150 total @c10 (TPOT 82 ms) | 10 | 2026-02-04 | https://github.com/mrdbourke/gpu-benchmarking/blob/main/benchmark_analysis/README.md |
| Qwen3-VL-30B-A3B-Instruct | same | FP8 | 1 image | TTFT 766 ms | 130 total @c10 (TPOT 95 ms) | 10 | 2026-02-04 | same as above |
| Qwen3-VL-8B-Instruct | same | FP8 | 0 / 1 image | TTFT 273 / 714 ms | 220 / 191 total @c10 (TPOT 50 / 64 ms) | 10 | 2026-02-04 | same as above |
| Qwen3-VL-8B-Instruct | llama.cpp | UD-Q4_K_XL | text / multimodal | 2478 / 799 | 35.5 / 33.1 | 1 | 2026-02-04 | same as above |
| Qwen3-VL-32B-Instruct (dense) | vLLM | FP8 | 0 / 1 image | TTFT 1037 / 1878 ms | 65 / 62 total @c10 (TPOT 179 / 210 ms) | 10 | 2026-02-04 | same as above |
| Qwen2.5-VL-7B-Instruct (UI-TARS-1.5-7B / OpenCUA-7B class) | TRT-LLM (NVIDIA blog), ISL/OSL 2048/128 | NVFP4 | 2048 | 65831.77 (as published) | 41.71 | BS1 | 2025-10-24 | https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks |
| Qwen2.5-Coder-7B (dense 7B text, reference) | llama.cpp b7941 | Q8_0 | d0 / d32768 | 2250 / 1471 | 29.43 / 22.51 | 1 | ~2026-02 | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/benches/dgx-spark/dgx-spark.md |
| GLM-4.6V (106B-A12B VLM) | vLLM, **2x Spark TP=2** | FP8 | n/a | n/a | 22.98 single; 40.96 @10 | 1, 10 | 2025-12-08 | https://forums.developer.nvidia.com/t/glm-4-6v-works-on-spark/353982 |
| GLM-4.6V | llama.cpp, single Spark | GGUF 8-bit (114 GB) | -c 15000 | n/a | "slow", 126 GB RAM (98%) | 1 | 2026-01-03 | same as above; single Spark should use AWQ-4bit (~64 GB) |
| UI-TARS-1.5-7B, OpenCUA, GUI-Owl, Holo2, Fara-7B | none found | n/a | n/a | n/a | n/a | n/a | n/a | No Spark-specific numbers. Use the Qwen2.5-VL-7B (~42 tg NVFP4 TRT-LLM) and Qwen3-VL-8B rows as proxies. The bases are, to my knowledge, Qwen2.5-VL-7B (UI-TARS-1.5-7B, OpenCUA-7B, Fara-7B) and Qwen3-VL (Holo2) [UNVERIFIED in this session] |

### 1c. Reference / dense models

| Model | Engine | Quant | pp | tg | Conc. | Date | URL |
|---|---|---|---|---|---|---|---|
| Llama 3.1 8B | SGLang (LMSYS) | FP8 | 7991 | 20.5 (BS1); 368 (BS32) | 1 / 32 | 2025-10-13 | https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark |
| Llama 3.1 8B | TRT-LLM (NVIDIA) | NVFP4 | 10256.9 | 38.65 | BS1 | 2025-10-24 | https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks |
| Llama 3.1 8B | Ollama v0.12.6 | q4_K_M / q8_0 | 7614 / 6110 | 38.02 / 25.23 | 1 | 2025-10-23 | https://registry.ollama.ai/blog/nvidia-spark-performance |
| Llama 3.1 70B | SGLang (LMSYS) | FP8 | 803 | 2.7 | n/a | 2025-10-13 | https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark |
| Llama 3.1 70B | Ollama v0.12.6 | q4_K_M | 1911 | 4.42 | 1 | 2025-10-23 | https://registry.ollama.ai/blog/nvidia-spark-performance |
| Qwen3 32B (dense) | Ollama | q4_K_M | 705 | 9.41 | 1 | 2025-10-23 | same as above |
| Qwen3 14B | TRT-LLM | NVFP4 | 5928.95 | 22.71 | BS1 | 2025-10-24 | https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks |

Other claims seen but not fully checked: RedHatAI/Qwen3.6-35B-A3B-NVFP4 at 218.85 tok/s (tg128, c1) via the "Atlas" engine with MTP K=2, which dropped to about half at c=2 (https://forums.developer.nvidia.com/t/nemotron-3-super-120b-a12b-nvfp4-on-single-dgx-spark-23-45-tok-s-spark-arena-com-benhmarks/370070). This is a reported leaderboard number, newer model family. Spark Arena (https://spark-arena.com/leaderboard) is the community leaderboard, but its table did not render for me, so I took no numbers from it.

### 1d. Spark practical caveats

1. **vLLM on sm_121.** The stock `vllm/vllm-openai` images did not compile SM121 cubins. That gave 13 tok/s vs 49 with a native build on Qwen3.5-35B-A3B in March 2026 (thread 365083). The community standard is eugr's `spark-vllm-docker` (used in most forum benchmarks). By June 2026 the official vLLM blog recommends `vllm/vllm-openai:cu130-nightly` builds "validated specifically for sm_121" (https://vllm.ai/blog/2026-06-01-vllm-dgx-spark). PyTorch still warns that it supports CC 8.0-12.0 while GB10 reports 12.1 (thread 366326). vLLM logs "SM12x detected - using native FlashInfer CUTLASS attention instead of TRT-LLM attention (cubins not available for SM12x)" (thread 379766).
2. **NVFP4 on GB10.** The hardware supports it, but the software lags. vLLM CUTLASS NVFP4 MoE kernels were not built for sm12x (vLLM PR #29242). In practice NVFP4 runs through Marlin W4A16 (bandwidth-bound) or emulation. NVIDIA (johnny_nv, 2025-12-15) said: "sm120 and 121 is equally the same, it is more lack support for sm12x in general". Fixes were still landing in March 2026 (https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069). Measured results: NVFP4 vs INT4 AutoRound on Qwen3.5-122B was 15-16.6 vs 28.4 tg, and on Qwen3-Coder-Next 14.1 vs 69 tg. The August 2026 harness saw ~1.1 tok/s vanilla emulation vs 77.1 with FlashInfer CUTLASS, attributed to missing `cvt.e2m1x2` (thread 379766). The NIM container clamps gpu_memory_utilization to 0.50 on UMA (thread 370070). **Recommendation: prefer FP8, INT4 AutoRound/AWQ, or MXFP4 GGUF on Spark. Use NVFP4 only through a tuned recipe** (Marlin backend `VLLM_NVFP4_GEMM_BACKEND=marlin`, or FlashInfer CUTLASS).
3. **MXFP4 (gpt-oss) in vLLM.** On SM121 there was a Marlin MoE shared-memory race at TP=1 that "silently corrupts the w2 GEMM output", plus other bugs. These are patched out-of-tree (thread 362824). llama.cpp MXFP4 is the low-risk path for gpt-oss (58.7 tg / 2444 pp at build 7941).
4. **Gated DeltaNet (Qwen3-Next / Qwen3-Coder-Next / Qwen3.5) on GB10.** It works in vLLM through the FLA Triton kernels and in llama.cpp. Known issues:
   - Triton "no allocator was set" crash under Ray/cluster mode (vLLM issue #33857; fixed by monkey-patching NullAllocator).
   - GDN Triton tuning patch for Qwen3.5 dimensions (thread 362824).
   - `gdn_linear_attn` crash in `cutlass_scaled_mm` with FP8+MTP on a generic image (thread 366326).
   - "Using default MoE config. Performance might be sub-optimal!" (no tuned GB10 fp8 MoE config).
   - Hybrid models in older vLLM logged "Hybrid or mamba-based model detected without support for prefix caching" (Dec 2025, thread 356153). By Feb 2026 Qwen3-Coder-Next ran with `--enable-prefix-caching` and it is essential: the model-card defaults disable it.
   - llama.cpp ngram-mod speculative decoding crashed on Qwen3-Coder-Next (thread 359571).
5. **KV cache / attention backend.** For Qwen3-Coder-Next FP8 at 0.8 utilization (~92 GB), FLASH_ATTN fits only about 60K tokens of KV, while `--attention-backend flashinfer` fits about 170K without fp8 KV (thread 359571). DFlash needs flash_attn, which shrinks the KV budget (thread 366326).
6. **Unified memory.** Usable pool is about 119-121 GiB.
   - `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` costs about 10% tg in llama.cpp (45.9 → 40.6 on gpt-oss-120b; discussion #16578).
   - fastsafetensors loading is "not usable beyond 0.85 of total RAM" and model loads can take >15 min without it (thread 361248).
   - vLLM "often does OOM just because buffers/cache memory filled" (forum comment; see the search summary). Drop page cache before launch [practice, UNVERIFIED as an official recommendation].
7. **Two models on one Spark.**
   - `gpu_memory_utilization` is a fraction of TOTAL memory, snapshotted at start. A real error: "Free memory on device (12.58/119.67 GiB) is less than desired GPU memory utilization (0.12, 14.36 GiB)".
   - Worked example: Qwen3-Next-80B FP8 @0.80 (resident 87.8 GiB) + Qwen3-4B @0.10 (13.8 GiB) = 101.6 GiB, ~18 GiB headroom. The 80B failed at 0.75 ("No available memory for the cache blocks") and at 0.85 (https://dev.to/ric03uec/two-qwen3-models-on-one-dgx-spark-the-residency-math-for-local-llm-coding-5bpj).
   - Procedure: start the big model first → `nvidia-smi --query-gpu=memory.used` → size the second against real free memory minus ~5 GiB → restart both twice to confirm.
   - Ollama KEEP_ALIVE keeps models resident in the same pool.
   - The official vLLM blog covers only single-model serving and recommends 0.85 and `--max-num-seqs 4` for one model.
   - Example budget for our pair: Qwen3-Coder-Next FP8 is ~80 GB of weights alone, so with a VLM it is tight. Qwen3-Coder-Next INT4 (~45 GB) or Qwen3-Coder-30B-A3B plus Qwen3-VL-8B/30B-A3B FP8 fits comfortably [sizing estimate, UNVERIFIED].
8. **Bandwidth contention.** Two co-resident models share the same 273 GB/s. When both decode at once, expect each model's tg to drop well below its solo number [physics-based expectation; no Spark measurement of concurrent dual-model decode found, UNVERIFIED].
9. **Speculative decoding** (MTP / EAGLE3 / DFlash) is the biggest single-stream lever on Spark: 1.3-2x on Qwen3.5. Acceptance rates vary (DFlash on 27B: 17-30%). LMSYS reports EAGLE3 "up to 2x".

---

## PART 2: Abliteration / uncensored variants on Hugging Face (checked via HF API, 2026-11-15)

Note: several original **huihui-ai** repos now return HTTP 401 on the public API (deleted, private or gated). These include GLM-4.5-Air, gpt-oss-120b, GLM-4.6V(-Flash), Devstral-Small-2, Qwen3-Next-80B and Qwen3.5-122B safetensors. Mirrors and quants by others remain. "Heretic" = p-e-w/heretic automatic abliteration. "Derestricted" = ArliAI using grimjim's norm-preserving biprojected abliteration.

| Base model | Abliterated / uncensored repos (exact links) | Notes |
|---|---|---|
| Qwen3-Coder-Next (80B-A3B, hybrid GDN) | https://huggingface.co/huihui-ai/Huihui-Qwen3-Coder-Next-abliterated ; https://huggingface.co/llmfan46/Qwen3-Coder-Next-Uncensored-Heretic (refusals 8/100 vs 100/100, KL 0.1488) ; GGUF: https://huggingface.co/mradermacher/Huihui-Qwen3-Coder-Next-abliterated-GGUF , https://huggingface.co/bartowski/huihui-ai_Qwen3-Coder-Next-abliterated-GGUF , https://huggingface.co/llmfan46/Qwen3-Coder-Next-Uncensored-Heretic-GGUF ; NVFP4: https://huggingface.co/trohrbaugh/Qwen3-Coder-Next-heretic-nvfp4 | huihui used remove-refusals-with-transformers and ships an alternative chat template "more compatible with using Tool Calling in llama-server" (opencode). No benchmark/KL published by huihui |
| Qwen3-Coder-30B-A3B-Instruct | https://huggingface.co/huihui-ai/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated ; https://huggingface.co/dpankros/Qwen3-Coder-30B-A3B-Instruct-Heretic (+ -AWQ) ; FP8: https://huggingface.co/jeffrog/Qwen3-Coder-30B-A3B-abliterated-FP8 ; GGUF: https://huggingface.co/mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-i1-GGUF | |
| Qwen3-Next-80B-A3B (bonus) | GGUF only now: https://huggingface.co/mradermacher/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-GGUF ; https://huggingface.co/noctrex/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-MXFP4_MOE-GGUF | huihui safetensors repo 401 |
| Qwen3-VL-30B-A3B-Instruct | https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated (also -Thinking-abliterated) ; https://huggingface.co/catplusplus/Qwen3-VL-30B-A3B-Instruct-Heretic ; GGUF: https://huggingface.co/noctrex/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-GGUF , https://huggingface.co/mradermacher/Qwen3-VL-30B-A3B-Instruct-Heretic-GGUF | |
| Qwen3-VL-32B-Instruct | https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-32B-Instruct-abliterated ; GGUF: https://huggingface.co/mradermacher/Huihui-Qwen3-VL-32B-Instruct-abliterated-GGUF , https://huggingface.co/mradermacher/Qwen3-VL-32B-Instruct-abliterated-v1-GGUF | |
| Qwen3-VL-8B-Instruct | https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated ; https://huggingface.co/saidutta69/Qwen3-VL-8B-Instruct-heretic ; GGUF: https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-abliterated-GGUF (500k downloads), https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF | |
| Qwen3.5-35B-A3B | https://huggingface.co/huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated ; https://huggingface.co/ArliAI/Qwen3.5-35B-A3B-Derestricted ; https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic ; https://huggingface.co/trohrbaugh/Qwen3.5-35B-A3B-heretic ; FP8/NVFP4: https://huggingface.co/nivvis/Qwen3.5-35B-A3B-heretic-v2-FP8 , https://huggingface.co/nivvis/Qwen3.5-35B-A3B-heretic-v2-NVFP4 ; MTP-preserved GGUF: https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF | Many quants. MTP-preserved variants matter for Spark speed (MTP gives 1.3-2x) |
| Qwen3.5-122B-A10B | https://huggingface.co/trohrbaugh/Qwen3.5-122B-A10B-heretic (+ -nvfp4) ; https://huggingface.co/happypatrick/Qwen3.5-122B-A10B-heretic-int4-AutoRound ; https://huggingface.co/GaelicThunder/Qwen3.5-122B-A10B-heretic-hybrid-int4fp8 ; https://huggingface.co/OptimizeLLM/Qwen3.5-122B-A10B-heretic-MTP-NVFP4 ; https://huggingface.co/bjk110/Qwen3.5-122B-A10B-abliterated-NVFP4 ; GGUF: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-122B-A10B-abliterated-GGUF , https://huggingface.co/llmfan46/Qwen3.5-122B-A10B-Uncensored-Heretic-Native-MTP-Preserved-GGUF (refusals 8/100 vs 92/100, KL 0.0856) | int4-AutoRound and hybrid-int4fp8 heretic variants line up with the fastest Spark recipes (Intel AutoRound INT4 / hybrid INT4+FP8 + MTP) |
| Qwen3.5-27B | https://huggingface.co/huihui-ai/Huihui-Qwen3.5-27B-abliterated ; https://huggingface.co/ArliAI/Qwen3.5-27B-Derestricted ; GGUF: https://huggingface.co/mradermacher/Qwen3.5-27B-heretic-GGUF | |
| Qwen3.5-397B-A17B (bonus) | GGUF: https://huggingface.co/Rick02280/Huihui-Qwen3.5-397B-A17B-abliterated-GGUF ; MLX: https://huggingface.co/cs2764/Qwen3.5-397B-A17B-6bit-abliterated-mlx | Too big for one Spark anyway |
| gpt-oss-120b | https://huggingface.co/kldzj/gpt-oss-120b-heretic (Heretic v1.0.1: refusals 19/100 vs 97/100, KL 0.92; v2: 22/100, KL 0.53) ; https://huggingface.co/ArliAI/gpt-oss-120b-Derestricted ; https://huggingface.co/wangzhang/gpt-oss-120b-abliterated ; MXFP4: https://huggingface.co/batsclamp/Huihui-gpt-oss-120b-mxfp4-abliterated , GGUF https://huggingface.co/noctrex/Huihui-gpt-oss-120b-abliterated-MXFP4_MOE-GGUF , https://huggingface.co/bartowski/kldzj_gpt-oss-120b-heretic-v2-GGUF | gpt-oss resists abliteration: high KL for the refusal reduction achieved |
| gpt-oss-20b | https://huggingface.co/p-e-w/gpt-oss-20b-heretic (Heretic v1.0.0: refusals 58/100 vs 100/100, KL 0.96) ; https://huggingface.co/huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2 ; https://huggingface.co/huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated-v2 ; https://huggingface.co/ArliAI/gpt-oss-20b-Derestricted | |
| GLM-4.5-Air | https://huggingface.co/ArliAI/GLM-4.5-Air-Derestricted (+ -FP8, -W8A8-INT8, -GPTQ-W4A16) ; https://huggingface.co/concedo/Huihui-GLM-4.5-Air-abliterated-lossytensors ; https://huggingface.co/gregfrank/GLM-4.5-Air-ULRE-abliterated ; GGUF: https://huggingface.co/bartowski/ArliAI_GLM-4.5-Air-Derestricted-GGUF ; REAP heretic: https://huggingface.co/jtl11/GLM-4.5-Air-REAP-82B-A12B-heretic | huihui-ai original now 401 |
| GLM-4.6V (106B) | GGUF only: https://huggingface.co/mradermacher/GLM-4.6V-heretic-GGUF ; MLX NVFP4: https://huggingface.co/mlx-community/GLM-4.6V-heretic-nvfp4 | No full-precision safetensors abliterated GLM-4.6V found |
| GLM-4.6V-Flash (9B) | https://huggingface.co/AliBilge/Huihui-GLM-4.6V-Flash-abliterated ; https://huggingface.co/AiAsistent/GLM-4.6V-Flash-heretic ; https://huggingface.co/3MPER0RR/GLM-4.6V-Flash-3MPER0RR-abliterated | |
| GLM-4.7-Flash (30B-A3B) | https://huggingface.co/huihui-ai/Huihui-GLM-4.7-Flash-abliterated ; https://huggingface.co/ThalisAI/GLM-4.7-Flash-heretic ; GGUF: https://huggingface.co/mradermacher/Huihui-GLM-4.7-Flash-abliterated-GGUF , https://huggingface.co/Olafangensan/GLM-4.7-Flash-heretic-GGUF | |
| Devstral Small 2 (24B, 2512) | https://huggingface.co/AliBilge/Huihui-Devstral-Small-2-24B-Instruct-2512-abliterated (mirror of huihui) ; https://huggingface.co/wangzhang/Devstral-Small-2-24B-Instruct-abliterated ; GGUF: https://huggingface.co/mradermacher/Devstral-Small-2-24B-Instruct-abliterated-GGUF | No heretic variant found. No Devstral 2 123B abliteration found |
| MiniMax-M2 (original) | **None found** for M2 / M2.1. Successors exist: M2.5 https://huggingface.co/wangzhang/MiniMax-M2.5-abliterated , https://huggingface.co/RadicalNotionAI/MiniMax-M2.5-bf16-heretic ; M2.7 https://huggingface.co/llmfan46/MiniMax-M2.7-BF16-ultra-uncensored-heretic , GGUF https://huggingface.co/Youssofal/MiniMax-M2.7-Abliterated-Heretic-GGUF | |
| UI-TARS-1.5-7B | **None found** (HF search: 168 UI-TARS repos, 0 abliterated/uncensored/heretic) | Nearest: abliterated base https://huggingface.co/huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated , but that is NOT UI-TARS |
| OpenCUA-7B / 32B | **None found** (17 repos, 0 hits) | |
| GUI-Owl-7B / 32B | **None found** (49 repos, 0 hits) | |
| Holo2 (and Holo1.5) | **None found** (22 + 13 repos, 0 hits) | |
| Fara-7B | https://huggingface.co/huihui-ai/Huihui-Fara-7B-abliterated ; https://huggingface.co/josephmayo/Fara-7B-Abliterated-v2 ; GGUF: https://huggingface.co/mradermacher/Huihui-Fara-7B-abliterated-GGUF | The only computer-use-specific model with abliterated weights |

### 2b. Does abliteration hurt agentic / tool-calling / coding performance?

The evidence is mixed. It depends on the model and the method. Refusal rate and KL do not capture damage to tool use.
- **Young, "Comparative Analysis of LLM Abliteration Methods" (arXiv 2512.13655, Dec 2025 / Jan 2026).** Tested 4 tools (Heretic, DECCP, ErisForge, FailSpy) on 16 models of 7-14B. GSM8K changed by +1.51 to **-18.81 pp (-26.5% relative)** depending on tool and model. Math is "the highest sensitivity". Single-pass ErisForge -0.28 pp and DECCP -0.13 pp averaged. Heretic KL ranged 0.043-1.646, model-dependent. https://arxiv.org/abs/2512.13655
- **David & Gervais (UCL), "Measuring Safety Alignment Effects in Autonomous Security Agents" (arXiv 2605.19722, May 2026).** Abliterated Llama-3.1-8B *broke the tool protocol*: 124/150 tool-failure labels and grounding 0.21, vs 35/150 and 2.80 for stock. Success was 0.0% vs 3.3%. Abliterated Qwen2.5-Coder-7B was *worse*: 2.0% vs 5.3% success. Less-restricted Gemma 4 31B / 26B-A4B did *better* (14.0% vs 0.7%; 10.7% vs 0.0%). https://arxiv.org/html/2605.19722
- **Level1Techs (thr3e, 2026-08-20) on agentic coding traces with Qwen3.8-27B variants.**
   - Heretic-ARA and huihui stayed close to stock: 0.7-1.4% top-1 token flips, p95 KLD 0.037-0.045, 0 invalid tool branches.
   - Cruder uncensors (AEON, Blackfrost) flipped 3-6% of tokens, had p95 KLD up to 0.685, and **corrupted tool calls**: port "5432" became "543ql", the tool envelope was not closed, and `page_size` became `size`.
   - Conclusion: "refusal benchmarks and model-card KLD numbers do not characterize collateral changes to tool use, exact literals, code, or long-context agentic work". https://forum.level1techs.com/t/why-your-local-llm-feels-dumber-than-it-is/253917/11
- **Heretic README benchmark (gemma-3-12b-it).** Refusals 97/100 → 3/100 in all three versions. KL: Heretic 0.16 vs huihui 0.45 vs mlabonne v2 1.04. I saw these numbers in a search summary of the README table; the current README page I fetched no longer shows the table, so treat them as [partially verified]. https://github.com/p-e-w/heretic
- **mlabonne (2024-06-13).** Abliteration "degraded the model's quality". A DPO "healing" pass recovered most of it, except GSM8K. Numbers appear only in images. https://huggingface.co/blog/mlabonne/abliteration
- **grimjim norm-preserving biprojected abliteration (2025-11-06), on Gemma-3-12B.** NatInt 21.33 vs 18.72 baseline vs 18.64 for standard abliteration. UGI 32.61 vs 19.58. This method is used by ArliAI "Derestricted" models. https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
- **Practical implication for our build.** Prefer low-KL variants: Heretic with KL < ~0.15, or Derestricted. Then **run our own tool-call regression suite** (BFCL-style JSON schema adherence, exact literals, multi-turn agent traces) against the stock model before switching. Keep the stock model's tool parser and chat template (huihui's Qwen3-Coder-Next ships a template tweak for llama-server tool calling). gpt-oss abliterations carry high KL (0.53-0.96), so expect more collateral damage there.

### 2c. Tool support for MoE and hybrid (Gated DeltaNet) abliteration

- **Heretic (p-e-w/heretic).** The README says it "supports most dense models, including many multimodal models, several different MoE architectures, and many modern hybrid models". It is now v2.x and also does slop and sycophancy steering. It optimizes refusal rate plus KL with Optuna, and supports 4-bit bnb loading to cut VRAM. Example cost: Qwen3-4B takes 30-40 min on an RTX 3090.
   - **Hybrid GDN is proven in practice:** heretic variants exist for Qwen3-Coder-Next (KL 0.149), Qwen3.5-35B-A3B, 122B-A10B (KL 0.086) and 27B.
   - **MoE is proven:** gpt-oss-20b and 120b, GLM-4.5-Air-REAP, GLM-4.7-Flash, MiniMax-M2.5/2.7.
   - Multimodal: Qwen3-VL-8B/30B-A3B and GLM-4.6V(-Flash) heretic variants exist.
   - https://github.com/p-e-w/heretic
- **remove-refusals-with-transformers (Sumandora).** Used by huihui-ai, including for Qwen3-Coder-Next and Qwen3-VL-30B-A3B. It is a "crude, proof-of-concept" script. It works where layers sit at `model.model.layers`; the README says "Some Qwen implementations for example don't work". It says nothing about MoE or hybrid support, but huihui's outputs show it can be adapted. https://github.com/Sumandora/remove-refusals-with-transformers
- **FailSpy abliterator.** Built on TransformerLens, so it is limited to TransformerLens-supported architectures (Llama-3 era). The README calls itself "a glorified IPython notebook". It is not suitable for Qwen3-Next/Qwen3.5 hybrid or recent MoE without porting [inferred from TransformerLens dependency]. https://github.com/FailSpy/abliterator
- **Hardware note.** Abliterating Qwen3-Coder-Next (80B BF16 ≈ 160 GB) does not fit in 128 GB at BF16. You would need heretic's 4-bit bnb mode (with an accuracy cost) or a bigger box. 30B-A3B / 35B-A3B / 8B VLMs are feasible on the Spark itself [sizing estimate].
- **No abliterated weights exist for UI-TARS-1.5-7B, OpenCUA, GUI-Owl or Holo2.** Running heretic ourselves on a 7-8B VLM is cheap. Validate grounding and click-coordinate accuracy afterwards: abliteration effects on GUI grounding are unmeasured in anything I found.
