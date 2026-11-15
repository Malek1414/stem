# Evidence base: training two specialised agents from own trajectories, and benchmarking them

Date: 2026-11-15. Scope: sections 1-4 of the brief (distillation legality excluded).

Verification: every arXiv ID below was resolved through the arXiv API on 2026-11-15 (title and first author match). Key numbers marked [abs] come from the arXiv abstract, [full] from the paper text read via fetch, [blog] from the blog page read, [snippet] from a search-result summary only (treat as unverified detail), [own] are my own calculations or inferences, not from a paper. Items I did not open at all are marked UNVERIFIED-NUMBER (existence confirmed, number not).

Bottom line: the literature supports the core recipe (verified self-generated trajectories + SFT, then RL) and shows large gains from only hundreds to a few thousand trajectories, but almost all of it is on many-repo or many-task data. Nothing found studies fine-tuning on one person's agent sessions. ~490 sessions is plausible for style/workflow/tool-habit adaptation, not for general capability. The weakest part of the design is evaluation power (12 B tasks, 20 C tasks) and a single LLM judge with no human calibration.

---

## 1. Learning agents from trajectories

### Data size vs gain

| Work | Data | Result | Cite |
|---|---|---|---|
| SWE-Gym (Pan et al., 2024) arXiv 2412.21139 | 491 rejection-sampled successful trajectories (gpt-4o, claude-3.5-sonnet) from 2,438 tasks; Qwen2.5-Coder 7B/14B/32B | SWE-Bench Verified 1.8->10.6 (7B), 4.0->16.4 (14B), 7.0->20.6 (32B); still rising at 491, no saturation; with verifier 32.0 [abs][full] | Closest analogue to our size (~500) and recipe |
| SWE-smith (Yang et al., 2025) arXiv 2504.21798 | 5,016 trajectories (Claude 3.7 expert), 50k task instances from 128 repos | SWE-agent-LM-32B 40.2% Verified [abs][full]; roughly logarithmic gain with number of repos (4/25/50/100 repos at 700 traj each) [full]; single-repo specialist tuning 21.2 vs 13.6 and 42.4 vs 33.3 on the target repo with minor generalisation loss [full] | Repo-specific tuning works; repo diversity matters for transfer |
| R2E-Gym (Jain et al., 2025) arXiv 2504.07164 | 8.7K procedurally built tasks | 34.4% Verified at 32B pass@1; 51% with hybrid verifier test-time scaling, competitive with o1 / sonnet-3.5-v2 [abs] | Verifier / best-of-n adds a lot |
| Skywork-SWE (Zeng et al., 2025) arXiv 2506.19290 | 8,000+ validated trajectories, 10,169 tasks | 38.0% pass@1, 47.0% with test-time scaling; data scaling shows no saturation [abs] | Log-linear scaling continues |
| SWE-RL (Wei et al., 2025) arXiv 2502.18449 | RL on PR history with similarity reward | Llama3-SWE-RL-70B 41.0% Verified; RL gave OOD gains while an SFT baseline degraded on average [abs] | Directly relevant: our benchmark B uses a similarity score (>=0.75); SWE-RL uses similarity as a training reward, not as a trusted test |
| Kimi-Dev (Yang et al., 2025) arXiv 2509.23045 | Agentless prior + 5k public trajectories SFT | 60.4% (workflow), 48.6% as agent, on par with Claude 3.5 Sonnet [abs] | A strong skill prior plus few thousand trajectories matches an older frontier model |
| SWE-Fixer (Xie et al., 2025) arXiv 2501.05040 | not opened | UNVERIFIED-NUMBER | |
| SWE-agent / OpenHands (Yang 2024 arXiv 2405.15793; Wang 2024 arXiv 2407.16741) | Scaffolds, not training data | Interface design (ACI) matters as much as the model [abs, SWE-agent title/claim]; OpenHands is the harness behind SWE-Gym/Skywork trajectories | Fix the scaffold and keep it identical for training and eval |
| AgentTuning (Zeng et al., 2023) arXiv 2310.12823 | AgentInstruct, mixed with general instructions | AgentLM-70B comparable to GPT-3.5-turbo on unseen agent tasks; mixing general data preserves general ability [abs]. Trajectory count (I recall 1,866) UNVERIFIED-NUMBER | Mix in general data |
| FireAct (Chen et al., 2023) arXiv 2310.05915 | 500 GPT-4 trajectories | Llama2-7B +77% on HotpotQA; more diverse (multi-task / multi-prompt) data improves further [abs] | ~500 trajectories can move a narrow task a lot |
| AgentBank (Song et al., 2024) arXiv 2410.07706 | 50k+ trajectories, 16 tasks | Scaling trajectory data gives generalist agents; skill generalisation limits [abs] | |
| Agent-FLAN (Chen et al., 2024) arXiv 2403.12881 | Redesigned corpus | Separate format-following from reasoning; negative samples reduce hallucination; Llama2-7B +3.5% over prior best [abs] | Add negative / "should refuse or ask" examples |
| Agent Data Protocol (Song et al., 2025) arXiv 2510.24702 | 13 datasets unified | ~20% average SFT gain over base models [abs] | Normalise formats; our hashed snapshots fit this |
| AgentGym (Xi 2024) arXiv 2406.04151; ETO "Trial and Error" (Song 2024) arXiv 2403.02502; Learn-by-interact (Su 2025) arXiv 2501.10893 | not opened in depth | Exploration-based, failure-aware and synthesized-instruction methods; ETO learns from failed trajectories via preference pairs [title-level] | Failures can be used as negatives (DPO-style) |

### Rejection sampling, STaR, ReST, expert iteration
- STaR (Zelikman et al., 2022) arXiv 2203.14465; ReST (Gulcehre et al., 2023) arXiv 2308.08998; ReST-EM "Beyond Human Data" (Singh et al., 2023) arXiv 2312.06585: sample, filter by binary feedback, fine-tune, repeat; scales with model size and beats training on human data alone (MATH, APPS) [abs]. Expert Iteration (Anthony et al., 2017) arXiv 1705.08439 is the origin.
- RFT (Yuan et al., 2023) arXiv 2308.01825: log-linear data/performance relation; more distinct correct reasoning paths give more gain; weaker models gain more; LLaMA-7B GSM8K 35.9 (SFT) -> 49.3 with RFT from multiple models [abs]. Implication: keep several distinct passing attempts per task, and deduplicate near-identical ones.
- Caveat [own]: all of these have strong verifiers. Our outcome checks for operator tasks (browser / app control) are weaker; verifier noise gets baked into the training set. R2E-Gym finds execution-based verifiers have low distinguishability and execution-free ones are stylistically biased (abs 2504.07164), so combine both.

### Computer-use agents
- UI-TARS (Qin et al., 2025) arXiv 2501.12326: OSWorld 24.6 (50 steps) vs Claude 22.0; AndroidWorld 46.6 vs GPT-4o 34.5 [abs]. UI-TARS-2 (Wang et al., 2025) arXiv 2509.02544: OSWorld 47.5, AndroidWorld 73.3, multi-turn RL with data flywheel [abs].
- OpenCUA (Wang et al., 2025) arXiv 2508.09123: AgentNet 22,625 human tasks, 18.6 steps average; OpenCUA-32B 34.8% (100 steps) and 72B 45.0% OSWorld-Verified, above Claude 4 Sonnet 41.5 and OpenAI CUA 31.4 but below Claude Sonnet 4.5 61.4; scaling Ubuntu 3K->10K trajectories +72%, Win/Mac 3K->14K +125%; mixing reasoning formats lifted a ablation from 13.1 to 18.5; general text data slightly helped [full].
- OS-Atlas (Wu 2024) arXiv 2410.23218 (13M GUI elements, grounding); Aguvis (Xu 2024) arXiv 2412.04454 (grounding then planning two-stage training); AgentTrek (Xu 2024) arXiv 2412.09605 (tutorial-guided synthesis, VLM-verified); Explorer (Pahuja 2025) arXiv 2502.11357; ScaleCUA (Liu 2025) arXiv 2509.15221 (+26.6 WebArena-Lite-v2) [abs].
- AndroidControl (Li et al., 2024) arXiv 2406.03679: LoRA PaLM-2S, step accuracy in-domain 1k episodes 52.5/71.4 (high/low-level), 10k 62.0/85.7, all (13.6k) 65.6/81.8 (rank 4); log-linear, extrapolated 500K-1M episodes for 95% in-domain; out-of-domain scales far slower (10M-60M) [full]. Takeaway: in-domain gains come cheap, out-of-domain do not. Our held-out project is OOD for project facts but in-domain for user style.

### Is ~490 sessions enough?
- For rejection-sampled coding: SWE-Gym shows 491 trajectories give +13.6 points at 32B but from a very low zero-shot start (7.0%); gains are larger when the base is weak and the verified-trajectory teacher is strong. FireAct shows 500 suffices for a narrow QA tool-use task.
- Your count after filtering matters, not 490 sessions. 36k tool calls, split across coder and operator and only the passing self-generated attempts, may leave a few hundred trajectories per agent. Expect the operator side (computer use) to be the weak one: published CUA gains rely on 10K+ trajectories plus large grounding corpora (OpenCUA, OS-Atlas, ScaleCUA).
- LIMA (Zhou 2023) arXiv 2305.11206: 1,000 curated examples shape style/format, 43% equal or preferred vs GPT-4 [abs]. LIMO (Ye 2025) arXiv 2502.03387 and s1 (Muennighoff 2025) arXiv 2501.19393 make the same small-data point for reasoning [title-level]. These support "style and format from hundreds", not new capability.

### Augmentations that have evidence
1. Mix in general / open agent data (AgentTuning, ADP, OpenCUA general text). Prevents forgetting and gives breadth.
2. Add public verified trajectories from other repos (SWE-Gym, SWE-smith, Kimi-Dev 5k) as base layer, then specialise on own data. SWE-smith repo-specialist result supports this ordering.
3. Multiple distinct passing samples per task (RFT).
4. Use failed trajectories as negatives (ETO) and negative/refusal examples (Agent-FLAN).
5. Backward-constructed instructions from your own interaction logs (Learn-by-interact, AgentTrek) for extra synthetic tasks.
6. Verifier / best-of-n at inference (SWE-Gym +11.4 points, R2E-Gym 34.4->51).
7. SFT first, then RL: SWE-RL and UI-TARS-2 show RL generalises better; RL's Razor (Shenfeld 2025) arXiv 2509.04259 and SFT Memorizes, RL Generalizes (Chu 2025) arXiv 2501.17161 explain why [abs].

---

## 2. Personalisation: one user's data

- Direct evidence is thin. I found no paper fine-tuning an agent on one developer's sessions. Nearest: LaMP (Salemi 2023) arXiv 2304.11406 and OPPU "one PEFT per user" (Tan et al., 2024) arXiv 2402.04401, which show per-user LoRA beats retrieval-only personalisation and combines well with retrieval [abs, qualitative]; PersonalWAB / personalized web agents (Cai 2024) arXiv 2410.17236 [title-level]; Agent Workflow Memory (Wang 2024) arXiv 2409.07429 (memory instead of weights) [title-level]. The domain is text personalisation, not tool-using coding agents.
- Fine-tune vs retrieve: Ovadia et al. 2023 arXiv 2312.05934 find RAG beats fine-tuning for injecting knowledge; Gekhman et al. 2024 arXiv 2405.05904 find new facts are learned slowly and encourage hallucination once learned [abs]. For your design: project facts (file layout, conventions) belong in context/retrieval (CLAUDE.md-style memory), behaviour and tool-use habits in weights.
- LoRA vs full fine-tune: LoRA Learns Less and Forgets Less (Biderman 2024) arXiv 2405.09673: LoRA underperforms full FT on code/math at standard ranks (full FT learns 10-100x higher-rank perturbations) but forgets less [abs]. LoRA vs Full Fine-tuning: Illusion of Equivalence (Shuttleworth 2024) arXiv 2410.21228: LoRA creates "intruder dimensions"; forgetting localised there [abs]. Scaling Laws for Forgetting (Kalajdzievski 2024) arXiv 2401.05605: LoRA still forgets; forgetting grows with parameters tuned and steps, inverse-linear with fine-tuning gain; early stopping does not avoid it [abs]. Luo 2023 arXiv 2308.08747: forgetting seen across 1B-7B, worse at larger scale in that range [abs].
- Thinking Machines "LoRA Without Regret" (Schulman et al., 2025-09-29) https://thinkingmachines.ai/blog/lora/ [blog]: LoRA on all layers equals full FT for small-to-medium datasets and for RL (works even at rank 1, since policy gradient gives O(1) bits per episode); underperforms when the dataset exceeds adapter capacity; optimal LoRA LR about 10x full FT (fit 9.8x). For ~500 sessions and RL, LoRA (all layers, not attention-only) is well supported. Blog, not peer reviewed.
- Overfitting: with a few hundred trajectories, expect memorising the user's repos. RL on self-generated data stays close to base policy (RL's Razor; SWE-RL kept OOD skills while SFT degraded), another reason to prefer rejection sampling from the open model itself over imitation of foreign trajectories.

---

## 3. Small specialised vs frontier

- Supports: LoRA Land (Zhao et al., 2024) arXiv 2405.00732: 4-bit LoRA fine-tuned models beat base by 34 points and GPT-4 by 10 on average across 31 narrow tasks, 10 bases, 310 models [abs]. xLAM (Zhang 2024) arXiv 2409.03215: tops Berkeley Function-Calling Leaderboard above GPT-4 / Claude-3 [abs]. Distilling Step-by-Step (Hsieh 2023) arXiv 2305.02301: small models beat larger LLMs with less data [abs]. Small Language Models are the Future of Agentic AI (Belcak 2025) arXiv 2506.02153 (position paper; argues repetitive narrow agent calls suit SLMs and heterogeneous systems) [abs]. ToolLLM (Qin 2023) arXiv 2307.16789 [title-level].
- SWE agents: open 32B models reached 34-48% (R2E-Gym 51% with verifiers, Kimi-Dev 48.6% agentic) versus frontier models far higher on Verified; OpenCUA-72B 45.0 vs Claude Sonnet 4.5 61.4 on OSWorld-Verified [full]. So open models can match older frontier snapshots on narrow suites, not the current frontier.
- Limits: narrow-task wins are on classification/extraction/function calling with clean metrics; open-ended multi-hour agent work shows a persistent gap. Out-of-domain scaling is far slower (AndroidControl). "Match frontier" for your goal should be defined as non-inferior on your B/C tasks within a margin, not parity on public benchmarks, and it must be tested against the same-date frontier model, not an older one.
- Hardware [own]: DGX Spark with 128 GB unified memory fits a 30-70B model in 4-8 bit for inference; the quantization penalty is small (Kurtic et al., 2024 arXiv 2411.02355: FP8 lossless, INT8 1-3% drop, INT4 weight-only competitive [abs]) but see section 4 on evaluating the exact quantized artifact you deploy.

---

## 4. Evaluation science

### 4.1 Next-action prediction as a proxy (benchmark A)
- Evidence is mixed and mostly self-reported. AndroidControl (2406.03679) states offline evaluation does not reward alternative routes or corrective actions [full]. OpenCUA reports a power-law relation between offline step success and online task success under a 15-step budget, with AgentNetBench ranking agreeing reasonably with online [full + snippet]. UItron (Zeng 2025) arXiv 2508.21767 reportedly finds step SR and task SR positively correlated but online scores consistently higher than offline, because errors can be recovered online [snippet; verify in paper]. SeeAct (Zheng 2024) arXiv 2401.01614: whole-task success is much higher online than offline (51.1 on live sites with manual grounding) [abs; offline figure not checked]. WebCanvas (Pan 2024) arXiv 2406.12373 argues similarly [snippet]. IntentScore (Chen 2026) arXiv 2604.05157 reports a large offline-to-online gap (97.5% offline pairwise accuracy, modest online gain) [snippet]. Survey Sager 2025 arXiv 2501.16150: offline task success is a lower bound [snippet; a quoted 12%->36% example UNVERIFIED].
- Conclusion: A is useful for ranking checkpoints cheaply and monitoring regression, but one-user, one-trajectory ground truth penalises valid alternatives. Report A as a development signal, never as a headline claim, and measure its correlation with B/C across your own checkpoints (you will have the data).
- Metric validity for B: Evtikhiev et al. 2022 arXiv 2208.03133 find BLEU/ROUGE/CodeBLEU-style similarity metrics agree weakly with human judgement and small differences should not be used to claim superiority [abs]. CodeBLEU (Ren 2020) arXiv 2009.10297 [existence only]. So the 0.75 line-similarity threshold needs an empirical check (see changes). Test-based checks are better where available (SWE-bench style); ABC checklist (Zhu 2025) arXiv 2507.02825: flawed task setup/reward design can shift measured performance by up to 100% relative [abs].

### 4.2 Contamination, held-out design, statistical power
- Contamination: SWE-Bench Illusion (Liang 2025) arXiv 2506.12286: Verified gains partly memorisation [abs]; SWE-Bench+ (Aleithan 2024) arXiv 2410.06992: 32.67% of successful patches involved solution leakage in the issue/comments [abs]; SWE-bench-Live (Zhang 2025) arXiv 2505.23419 and LiveCodeBench (Jain 2024) arXiv 2403.07974 use time-split fresh data; survey Xu 2024 arXiv 2406.04244. For you: the real risk is not public contamination but (a) frontier models having seen the same open-source repos if any are public, and (b) your training sessions sharing code, dependencies or later commits with the held-out project. Dedup at repo and time level.
- One project held out: a single split answers "does it generalise to this unseen project", with no variance estimate across projects. AI Agents That Matter (Kapoor 2024) arXiv 2407.01502: inadequate holdouts make fragile agents; report cost with accuracy [abs]. HAL (Kapoor 2025) arXiv 2510.11977: 21,730 rollouts across 9 models, 9 benchmarks; agent evaluation needs standardised harness and log inspection [abs].
- Power [own calculation, Wilson 95% interval]: 6/12 passes gives [0.25, 0.75]; 10/20 gives [0.30, 0.70]. Two systems differing by 15-20 points on 12 or 20 tasks are statistically indistinguishable unless paired by task and very consistent. Repeating runs reduces sampling noise of the agent but not task-sampling noise (Miller 2024 arXiv 2411.00640: treat questions as draws from a super-population, use paired differences and clustered SEs [abs]). Bowyer et al. 2025 arXiv 2503.01747: CLT error bars are too narrow below a few hundred items; use Bayesian / exact intervals [abs]. Madaan 2024 arXiv 2406.10229: quantifies seed variance in benchmarks [abs]. tau-bench (Yao 2024) arXiv 2406.12045 pass^k for reliability [abs].
- SWE-Gym reports +-2.1 standard error on ~500 Verified tasks [full]; with 12-20 tasks the standard error is about 12-14 points [own].

### 4.3 LLM-as-judge (benchmark C)
- Agreement: Zheng et al. 2023 arXiv 2306.05685: GPT-4 judges reach >80% agreement with humans, the same as human-human; documented position, verbosity and self-enhancement biases [abs].
- Position bias: Wang 2023 arXiv 2305.17926 (Vicuna-13B beat ChatGPT on 66/80 queries by order alone; remedy: swap order and average) [abs]; Shi 2024 arXiv 2406.07791 (15 judges, 150k instances; bias varies by judge and task, grows with small quality gap) [abs]. Your 1-5 absolute scoring is not pairwise, but score anchoring and scale compression are the analogue: Stureborg 2024 arXiv 2405.01724 (familiarity bias, skewed rating distributions, anchoring) [abs].
- Self-preference: Panickssery 2024 arXiv 2404.13076 and Wataoka 2024 arXiv 2410.21819: judges favour outputs with low perplexity / their own style [abs]. gpt-oss-120b (OpenAI 2025, arXiv 2508.10925, model card) is a different family from the Claude references and from Qwen-type open bases, which helps, but if gpt-oss is ever a candidate agent or a base, exclude it from judging itself.
- Verbosity/length: Dubois 2024 arXiv 2404.04475 (regression-based length control) [abs]. Null models: Zheng 2024 arXiv 2410.07137: a constant irrelevant response gets 86.5% LC win rate on AlpacaEval 2.0 [abs]; so include null/garbage controls in your C set. Other: CALM, Ye 2024 arXiv 2410.02736 (12 biases) [abs]; JudgeBench, Tan 2024 arXiv 2410.12784 (judges weak on hard reasoning/coding pairs) [abs].
- Calibration methods: Shankar 2024 arXiv 2404.12272 (validate validators with human grades; criteria drift) [abs]; Jung 2024 arXiv 2407.18370 (judge confidence + selective trust with agreement guarantees) [abs]; Prediction-Powered Inference, Angelopoulos 2023 arXiv 2301.09633 (small human-labelled set corrects bias of an automatic judge and yields valid CIs) [abs]; Prometheus 2, Kim 2024 arXiv 2405.01535 (open judge trained for rubric scoring) [title-level]; Agent-as-a-Judge, Zhuge 2024 arXiv 2410.10934 (judge sees intermediate steps; beats plain LLM judge, DevAI 55 tasks) [abs]; CodeJudge Tong 2024 arXiv 2410.02184 and Ahmed 2024 arXiv 2408.05534 on LLM judges for code [existence only]; survey Gu 2024 arXiv 2411.15594.
- Practical calibration recipe (inference from the above): human-grade 30-50% of C items (you are the only ground truth), report weighted kappa / Spearman between judge and you, run swap/paraphrase-rubric robustness, add null and reference-solution controls, and report PPI-corrected intervals.

### 4.4 Reproducibility
- "Temperature 0" is not deterministic. Atil 2024 arXiv 2408.04667: up to 15% accuracy variation across runs, best-to-worst gap up to 70%, no model repeatable across all tasks [abs]. Thinking Machines (He, 2025-09-10) https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ [blog]: root cause is lack of batch invariance in kernels; Qwen3-235B at temp 0, 1000 completions gave 80 unique outputs, first divergence at token 103; batch-invariant kernels gave 1000 identical; vLLM overhead 26 s -> 42-55 s for 1000 sequences on Qwen3-8B. Yuan 2025 arXiv 2506.09501: batch size, GPU count and GPU type change outputs; DeepSeek-R1-Distill-Qwen-7B up to 9% accuracy variation under bf16 greedy [abs]. gpt-oss-120b is a reasoning model, the case where early-token rounding cascades into divergent chains of thought.
- Quantization: Kurtic 2024 arXiv 2411.02355 and Li 2024 arXiv 2402.18158 [abs] show accuracy depends on format; Biderman 2024 arXiv 2405.14782 (lm-eval lessons): evaluation setup sensitivity, publish exact prompts and configs [abs].
- Implication: "pinned model + temperature 0 + fixed rubric" pins the inputs, not the outputs. Pin also the serving stack (engine version, hardware, quantization, batch size / batch-invariant mode), cache and store every judge output with a hash, and treat the cache, not the re-run, as the reproducible artifact.

---

## Keep / change / add for our design

Keep
- Keep outcome-verified, self-generated trajectories as training targets (rejection sampling plus SFT then RL): SWE-Gym 2412.21139 (491 trajectories, +13.6 pts), ReST-EM 2312.06585, RFT 2308.01825.
- Keep SFT then RL ordering: SWE-RL 2502.18449, RL's Razor 2509.04259, SFT Memorizes RL Generalizes 2501.17161.
- Keep LoRA (all layers) as the default for the ~500-session regime and for RL: Thinking Machines LoRA blog (thinkingmachines.ai/blog/lora), Biderman 2405.09673 (less forgetting).
- Keep hashed, versioned, byte-reproducible snapshots and a unified trajectory format: Agent Data Protocol 2510.24702, Biderman lm-eval 2405.14782.
- Keep a pinned local judge from a different model family than the agents: Panickssery 2404.13076, Wataoka 2410.21819.
- Keep separate coder/operator specialisation and repo-specialist logic: SWE-smith 2504.21798 (specialist 21.2 vs 13.6 on target repo, minor loss elsewhere).

Change
- Change the claim "match frontier": define non-inferiority margins on your own tasks against same-date frontier models, and add pass^k reliability: OpenCUA 2508.09123 (72B 45.0 vs Sonnet 4.5 61.4), tau-bench 2406.12045.
- Change B/C reporting from point scores to paired, task-level comparisons with exact or Bayesian intervals; 6/12 gives [25%,75%] [own], so decisions need larger n or a rotating split: Miller 2411.00640, Bowyer 2503.01747.
- Change the held-out design from one project to leave-one-project-out rotation (or at least add a temporal split inside projects) and dedup by repo, dependency and time: AI Agents That Matter 2407.01502, SWE-bench Live 2505.23419, SWE-Bench+ 2410.06992.
- Change the 0.75 line-similarity pass rule: calibrate it against your own judgement on a sample, and prefer test/build/behaviour checks; treat similarity as secondary: Evtikhiev 2208.03133, ABC 2507.02825.
- Change "temperature 0 means reproducible": pin engine, hardware, quantization and batch size, cache every judge output, or use batch-invariant kernels: He/Thinking Machines blog, Yuan 2506.09501, Atil 2408.04667.
- Change project-fact learning from weights to retrieval/context memory: Ovadia 2312.05934, Gekhman 2405.05904.
- Change the operator expectation: published CUA gains need 10K+ trajectories and large grounding data, so start the operator from an existing open CUA base (UI-TARS / OpenCUA / ScaleCUA class) rather than a text-only base: OpenCUA 2508.09123, UI-TARS 2501.12326, AndroidControl 2406.03679.

Add
- Add a human-graded calibration set for judge C (you grade 30-50%), report judge-human agreement and PPI-corrected CIs: Shankar 2404.12272, Angelopoulos 2301.09633, Jung 2407.18370.
- Add judge controls: null/constant answers, order/paraphrase swap, reference solutions, length-vs-score regression: Zheng 2410.07137, Wang 2305.17926, Dubois 2404.04475.
- Add a general/public-agent data mix (and public verified SWE trajectories) as a base layer, with a forgetting check on a general benchmark before/after: AgentTuning 2310.12823, Kalajdzievski 2401.05605, SWE-smith 2504.21798.
- Add multiple distinct passing samples per task, failed attempts as negatives, and a best-of-n verifier at inference: RFT 2308.01825, ETO 2403.02502, SWE-Gym verifier +11.4, R2E-Gym 2504.07164.
- Add a measured A-vs-B/C correlation across your checkpoints and label A a development signal only: AndroidControl 2406.03679, OpenCUA 2508.09123 (power-law claim), UItron 2508.21767 [snippet].
- Add evaluation of the exact deployed quantized artifact on the Spark, not only the bf16 checkpoint: Kurtic 2411.02355, Li 2402.18158.
- Add cost/latency per task alongside accuracy: Kapoor 2407.01502, HAL 2510.11977.

## Gaps and unverified
- No paper found on fine-tuning agents on a single user's sessions; the personalisation evidence is by analogy (OPPU, LaMP, SWE-smith repo specialists).
- Not opened: SWE-Fixer, AgentGym, Learn-by-interact, LIMO, s1, ToolLLM, Prometheus 2, CodeJudge, Ahmed 2024, Explorer (existence and abstract only). AgentTuning's trajectory count is from memory. UItron, IntentScore, WebCanvas, the 12%->36% survey example and SeeAct's offline figure come from a search-result summary, not the paper text.
- Not found: any study quantifying correlation between next-action accuracy and task success for coding agents specifically.
