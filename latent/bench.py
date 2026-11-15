"""Latent vs text handoff benchmark.

Setting: an operator (sender) has read a code file plus an error log; the coder (receiver) must answer a
question that depends on it. Both run the same model. Conditions:

  none        receiver gets only the question (floor)
  text_full   receiver re-reads the operator's whole context as text (ceiling, costs a full re-prefill)
  text_card   operator decodes a short handoff note; receiver reads the note (what the link does today)
  kv_full     operator's KV cache for its context is the message, fp16, all layers
  kv_q8       same, int8 per-vector quantised
  kvcomm      only the top `ratio` of layers, chosen once on calibration tasks by attention mass x Gaussian
              prior over depth (KVComm, Shi et al. 2025, arXiv 2510.03346); other layers mask the sender out
  kv_goal     like latentmas with 0 latent steps: the operator's KV after context + the coder's goal. Ablation that
              separates "the sender knew the goal" from "the latent steps helped" (added after run 1 showed the confound)
  latentmas   operator adds `latent_steps` latent thoughts after its context (LatentMAS, arXiv 2511.20639)

Every condition records accuracy, sender seconds, receiver time-to-first-token, decode seconds and wire
bytes. End-to-end time is reported for three links: co-located (no transfer), 1 Gbit/s LAN, and the
measured Mac <-> Spark link (--mbps, --rtt-ms). Greedy decoding, fixed task seed."""
from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from forge.stats import wilson

from . import attn, cache as kvc, latentmas

CONDITIONS = ("none", "text_full", "text_card", "kv_full", "kv_q8", "kvcomm", "kv_goal", "latentmas")
VERBS = "load parse apply compute merge validate render sync flush resolve encode fetch".split()
NOUNS = "order invoice cart token session ledger quota batch refund schema cursor payload".split()
CONSTS = "RATE_LIMIT MAX_RETRIES TIMEOUT_MS PAGE_SIZE CACHE_TTL BATCH_CAP SHARD_COUNT GRACE_DAYS".split()


# --- tasks ----------------------------------------------------------------------------------------
FILLER_VERBS = "scan build emit trace probe stage drain index patch split".split()
FILLER_NOUNS = "widget bucket mirror ticket stream report plan shard vault queue".split()


def _filler(r: random.Random, chars: int) -> list[str]:
    """Distractor code and logs: different names, no real constants, so the answer stays unique."""
    out, size, k = [], 0, 0
    while size < chars:
        k += 1
        f = f"{r.choice(FILLER_VERBS)}_{r.choice(FILLER_NOUNS)}_{k}"
        block = [f"def {f}(item, ctx):", f"    limit = {r.randint(1, 5000)}",
                 f"    if item.size > limit:", f"        ctx.log('{f} skipped', item.id)", "        return None",
                 f"    return ctx.emit('{f}', item)", "",
                 f"# 2026-11-15 08:{r.randint(10, 59)}:{r.randint(10, 59)} DEBUG {f} handled {r.randint(1, 999)} items", ""]
        out += block
        size += sum(len(x) + 1 for x in block)
    return out


def make_tasks(n: int, seed: int, ctx_chars: int = 0) -> list[dict]:
    """ctx_chars > 0 pads the operator's context with distractors up to that size (long-context sweep)."""
    out = []
    for i in range(n):
        r = random.Random(f"{seed}:{i}")
        fns = r.sample([f"{v}_{o}" for v in VERBS for o in NOUNS], 8)
        params = [r.choice(["amount", "limit", "user_id", "region", "code", "offset"]) for _ in fns]
        consts = {c: r.randint(10, 999) for c in r.sample(CONSTS, 6)}
        bad = r.randrange(len(fns))
        val = r.choice([str(r.randint(1000, 9999)), r.choice(["eu-west-9", "NaN", "-1", "ZX81", "null"])])
        mod = r.choice(["billing", "orders", "auth", "search"])
        lines = [f"# file: {mod}/core.py"] + [f"{c} = {v}" for c, v in consts.items()] + [""]
        for f, p in zip(fns, params):
            c = r.choice(list(consts))
            lines += [f"def {f}({p}, ctx):", f"    if {p} is None or {p} > {c}:",
                      f"        raise ValueError(f'invalid {p}={{{p}}}')", f"    return ctx.step('{f}', {p})", ""]
        log = [f"2026-11-15 09:{r.randint(10, 59)}:{r.randint(10, 59)} INFO {r.choice(fns)} ok in {r.randint(2, 90)} ms"
               for _ in range(6)]
        tb = ["Traceback (most recent call last):", f'  File "{mod}/runner.py", line {r.randint(20, 200)}, in run',
              f'  File "{mod}/core.py", line {r.randint(10, 60)}, in {fns[bad]}', f"ValueError: invalid {params[bad]}={val}"]
        if ctx_chars:
            fill = _filler(random.Random(f"{seed}:{i}:fill"), ctx_chars - sum(len(x) + 1 for x in lines + log + tb))
            half = len(fill) // 2
            lines = lines[:1] + fill[:half] + lines[1:] + fill[half:]  # facts end up in the middle of the file
        ctx = "\n".join(lines + log + tb)
        kind = i % 3
        if kind == 0:
            q, a = "Which function raised the ValueError?", fns[bad]
        elif kind == 1:
            c = r.choice(list(consts))
            q, a = f"What is the value of {c} in {mod}/core.py?", str(consts[c])
        else:
            q, a = "What exact value was rejected in the ValueError?", val
        out.append({"id": f"{seed}-{i}", "context": ctx, "question": q, "answer": a, "kind": kind})
    return out


def correct(answer: str, target: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(target.lower())}(?![\w-])", answer.lower()) is not None


# --- model plumbing -------------------------------------------------------------------------------
class Runner:
    def __init__(self, model_id: str, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(self.device, torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype, attn_implementation=attn.NAME).to(self.device).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self._align = None

    def ids(self, text: str) -> torch.Tensor:
        return self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

    def sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, cache=None):
        t = time.perf_counter()
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
        self.sync()
        return out.past_key_values, out.logits[:, -1], out.hidden_states[-1][:, -1:], time.perf_counter() - t

    @torch.no_grad()
    def decode(self, cache, logits, max_new: int) -> tuple[str, float]:
        t, toks = time.perf_counter(), []
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        for _ in range(max_new):
            nxt = int(logits.argmax(-1))
            if nxt in eos:
                break
            toks.append(nxt)
            out = self.model(input_ids=torch.tensor([[nxt]], device=self.device), past_key_values=cache, use_cache=True)
            cache, logits = out.past_key_values, out.logits[:, -1]
        self.sync()
        return self.tok.decode(toks, skip_special_tokens=True).strip(), time.perf_counter() - t

    def alignment(self):
        if self._align is None:
            self._align = latentmas.alignment(self.model)
        return self._align


# Qwen chat format, split so the operator's part is an exact token prefix of the coder's prompt.
CTX = "<|im_start|>user\n[Operator context: what I read and saw]\n{context}\n"
ASK = "\nQuestion: {question}\nReply with only the answer.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CARD_REQ = ("\nThe coder must answer: {question}\nWrite a handoff note of at most 30 words with the facts the coder "
            "needs.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
CARD = "<|im_start|>user\n[Handoff note from the operator]\n{card}\n"
THINK = "\nThe coder will be asked: {question}\nThink about what the coder needs.\n"
NONE = "<|im_start|>user\n"


def run_condition(R: Runner, task: dict, cond: str, cfg: dict) -> dict:
    ctx_ids = R.ids(CTX.format(context=task["context"]))
    ask_ids = R.ids(ASK.format(question=task["question"]))
    rec = {"sender_s": 0.0, "pack_s": 0.0, "bytes": 0, "sender_tokens": 0}
    hide, selected, cache = 0, None, None
    if cond == "none":
        prompt = torch.cat([R.ids(NONE), ask_ids], 1)
    elif cond == "text_full":
        prompt = torch.cat([ctx_ids, ask_ids], 1)
        rec["bytes"] = len(task["context"].encode())
    elif cond == "text_card":
        c, lg, _, s1 = R.prefill(torch.cat([ctx_ids, R.ids(CARD_REQ.format(question=task["question"]))], 1))
        card, s2 = R.decode(c, lg, cfg["card_tokens"])
        rec.update(sender_s=s1 + s2, card=card, bytes=len(card.encode()))
        prompt = torch.cat([R.ids(CARD.format(card=card)), ask_ids], 1)
    else:
        sender_ids = ctx_ids if cond not in ("latentmas", "kv_goal") else torch.cat([ctx_ids, R.ids(THINK.format(question=task["question"]))], 1)
        c, _, h, s1 = R.prefill(sender_ids)
        if cond == "latentmas":
            t = time.perf_counter()
            w_a, norm = R.alignment()
            c, _ = latentmas.think(R.model, c, h, cfg["latent_steps"], w_a, norm)
            R.sync()
            s1 += time.perf_counter() - t
        layers = sorted(cfg["kvcomm_layers"]) if cond == "kvcomm" else None
        t = time.perf_counter()
        blob = kvc.pack(c, R.model_id, layers=layers, quant="int8" if cond == "kv_q8" else "fp16")
        cache, hdr = kvc.unpack(blob, R.model_id, R.device, R.dtype, config=R.model.config)
        rec.update(sender_s=s1, pack_s=time.perf_counter() - t, bytes=len(blob), sender_tokens=hdr["seq_len"])
        hide = kvc.seq_len(cache)
        selected = set(layers) if layers is not None else None
        prompt = ask_ids
    with attn.hide_sender(hide if selected is not None else 0, selected):
        cache, lg, _, ttft = R.prefill(prompt, cache)
        ans, dec = R.decode(cache, lg, cfg["answer_tokens"])
    rec.update(answer=ans, correct=correct(ans, task["answer"]), receiver_ttft_s=ttft, decode_s=dec,
               receiver_prompt_tokens=prompt.shape[1])
    return rec


@torch.no_grad()
def calibrate_kvcomm(R: Runner, tasks: list[dict], ratio: float) -> dict:
    """Layer importance = attention mass the coder's question puts on the operator's positions, averaged over
    calibration tasks, times a Gaussian prior over depth (centre L/2, sigma L/4; our choice of prior parameters)."""
    L, scores = R.n_layers, {}
    attn_layers = [i for i, k in enumerate(getattr(R.model.config, "layer_types", None) or ["full_attention"] * L)
                   if k != "linear_attention"]
    for task in tasks:
        c, _, _, _ = R.prefill(R.ids(CTX.format(context=task["context"])))
        rec: dict = {}
        with attn.hide_sender(kvc.seq_len(c), None, record=rec):
            R.prefill(R.ids(ASK.format(question=task["question"])), c)
        for li, v in rec.items():
            scores[li] = scores.get(li, 0.0) + v / len(tasks)
    prior = {i: math.exp(-((i - L / 2) ** 2) / (2 * (L / 4) ** 2)) for i in range(L)}
    ranked = sorted(attn_layers, key=lambda i: -scores.get(i, 0.0) * prior[i])
    k = max(1, math.ceil(ratio * len(attn_layers)))  # hybrid models: recurrent state always travels
    return {"layers": sorted(ranked[:k]), "scores": {i: round(scores.get(i, 0.0), 4) for i in attn_layers}}


def e2e(r: dict, mbps: float, rtt_ms: float) -> float:
    """Co-located agents hand over the cache object itself, so no serialisation and no wire time."""
    if mbps == math.inf:
        return r["sender_s"] + r["receiver_ttft_s"] + r["decode_s"]
    wire = r["bytes"] * 8 / (mbps * 1e6) + rtt_ms / 1000
    return r["sender_s"] + r["pack_s"] + wire + r["receiver_ttft_s"] + r["decode_s"]


def run(model_id: str, n: int = 60, seed: int = 20261115, conditions=CONDITIONS, out_dir: Path = Path("results/latent"),
        mbps: float = 100.0, rtt_ms: float = 17.0, ratio: float = 0.3, latent_steps: int = 20, device: str | None = None,
        ctx_chars: int = 0, runner: "Runner | None" = None) -> dict:
    R = runner or Runner(model_id, device)
    tasks = make_tasks(n, seed, ctx_chars)
    cfg = {"card_tokens": 60, "answer_tokens": 24, "latent_steps": latent_steps, "ratio": ratio}
    if "kvcomm" in conditions:
        cal = calibrate_kvcomm(R, make_tasks(8, seed + 1), ratio)
        cfg["kvcomm_layers"], cfg["kvcomm_scores"] = cal["layers"], cal["scores"]
    R.prefill(R.ids("warm up"))  # first call pays kernel compilation; keep it out of the timings
    rows = []
    for task in tasks:
        for cond in conditions:
            rows.append({"task": task["id"], "kind": task["kind"], "cond": cond, **run_condition(R, task, cond, cfg)})
    links = {"colocated": (math.inf, 0.0), "lan_1gbps": (1000.0, 1.0), "mac_spark": (mbps, rtt_ms)}
    summary = {}
    for cond in conditions:
        rs = [r for r in rows if r["cond"] == cond]
        k = sum(r["correct"] for r in rs)
        summary[cond] = {"accuracy": round(k / len(rs), 4), "ci95": [round(x, 4) for x in wilson(k, len(rs))],
                         "mean_bytes": round(sum(r["bytes"] for r in rs) / len(rs)),
                         "mean_sender_s": round(sum(r["sender_s"] for r in rs) / len(rs), 4),
                         "mean_receiver_ttft_s": round(sum(r["receiver_ttft_s"] for r in rs) / len(rs), 4),
                         "e2e_s": {name: round(sum(e2e(r, *l) for r in rs) / len(rs), 4) for name, l in links.items()}}
    if {"kv_full", "text_full"} <= set(conditions):  # sanity: same tokens, so answers should agree
        a = {r["task"]: r["answer"] for r in rows if r["cond"] == "text_full"}
        b = {r["task"]: r["answer"] for r in rows if r["cond"] == "kv_full"}
        summary["kv_full_vs_text_full_agreement"] = round(sum(a[t] == b[t] for t in a) / len(a), 4)
    res = {"model": model_id, "device": R.device, "ctx_chars": ctx_chars,
           "mean_sender_tokens": round(sum(r["sender_tokens"] for r in rows) / len(rows)), "dtype": str(R.dtype), "torch": torch.__version__,
           "transformers": __import__("transformers").__version__, "n": n, "seed": seed, "cfg": cfg,
           "links": {k: [v[0] if v[0] != math.inf else "inf", v[1]] for k, v in links.items()},
           "summary": summary, "rows": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{model_id.replace('/', '_')}_n{n}_s{seed}" + (f"_ctx{ctx_chars}" if ctx_chars else "") + ".json"
    p.write_text(json.dumps(res, indent=1))
    res["path"] = str(p)
    return res
