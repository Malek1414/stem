"""LoRA fine-tuning on forge.sft examples (chat messages + tool calls), loss on assistant turns only.

  python -m forge.train_lora --model hf/qwen36-heretic --data data/sft/v0002/coder.jsonl --out adapters/coder-v1 \
      --max-len 16384 --max-tokens 4000000

Plain PyTorch loop (no Trainer): bf16 base, LoRA adapters on every linear layer of the language model except the
MoE router and the vision tower, gradient checkpointing, AdamW with warmup + cosine decay, a held-out split for
eval loss, checkpoints every --save-every steps (resumable), metrics in <out>/log.jsonl. Serve the result with
llama.cpp: convert_lora_to_gguf.py --base <model> <out> then llama-server --lora.

Note: Claude's thinking is redacted in the logs, so assistant turns carry empty think blocks; the tuned model
learns to answer without long thinking (lower latency, which the operator wants)."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch

ASSISTANT = re.compile(r"<\|im_start\|>assistant\n(.*?)(<\|im_end\|>)", re.S)
EMPTY_THINK = re.compile(r"<think>\s*</think>\s*")  # what the template writes for a turn without reasoning


def dict_args(msgs: list[dict]) -> list[dict]:
    """Tool-call arguments as dicts: the Qwen3.8 template iterates them (arguments|items) and fails on the JSON
    strings forge.sft writes; the Qwen3.6 template took either."""
    out = []
    for m in msgs:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": _as_dict(c["function"].get("arguments"))}}
                                     for c in m["tool_calls"]]}
        out.append(m)
    return out


def _as_dict(a) -> dict:
    if isinstance(a, dict):
        return a
    try:
        v = json.loads(a or "{}")
    except json.JSONDecodeError:
        return {"input": a}
    return v if isinstance(v, dict) else {"input": v}


def fit(tok, ex: dict, max_len: int) -> dict:
    """An example with loss_from (forge.distill) that is longer than max_len loses its oldest history messages
    (after the system prompt) instead of its end: cutting the end would cut the new turns, which are all it
    teaches (16k: 20 of 35 examples lost every label, 17.11.2026). The benchmark trims history the same way."""
    lf = ex.get("loss_from")
    if not lf:
        return ex
    msgs = list(ex["messages"])
    size = lambda m: len(tok(tok.apply_chat_template(dict_args(m), tools=ex.get("tools"), tokenize=False),
                             add_special_tokens=False)["input_ids"])
    n = size(msgs)
    while n > max_len and lf > 2:
        drop = 1
        while drop < lf - 1 and msgs[1 + drop]["role"] == "tool":  # never leave a tool result without its call
            drop += 1
        if lf - drop < 2:
            break
        msgs = msgs[:1] + msgs[1 + drop:]
        lf -= drop
        n = size(msgs)
    return {**ex, "messages": msgs, "loss_from": lf}


def encode(tok, ex: dict, max_len: int) -> dict | None:
    """Token ids and labels; labels are -100 outside assistant replies (and their closing <|im_end|>). With
    ex["loss_from"] (forge.distill), assistant replies before that message index are context only."""
    ex = fit(tok, ex, max_len)
    text = tok.apply_chat_template(dict_args(ex["messages"]), tools=ex.get("tools"), tokenize=False)
    spans = [(m.start(1), m.end(2)) for m in ASSISTANT.finditer(text)]
    skip = sum(m["role"] == "assistant" for m in ex["messages"][: ex.get("loss_from", 0)])
    spans = spans[skip:]
    if not spans:
        return None
    # an empty think block is never a target: training on it taught both LoRA v1s to skip thinking and end turns
    # early (27B v1: 2.30 vs 2.85, 17.11.2026); turns with Claude's summarised thinking keep theirs
    empty = [(m.start(), m.end()) for m in EMPTY_THINK.finditer(text)]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_len)
    ids, labels, j, k = enc["input_ids"], [], 0, 0
    for tid, (a, b) in zip(ids, enc["offset_mapping"]):
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        while k < len(empty) and empty[k][1] <= a:
            k += 1
        inside = j < len(spans) and spans[j][0] <= a < spans[j][1]
        in_empty = k < len(empty) and empty[k][0] <= a < empty[k][1]
        labels.append(tid if inside and not in_empty else -100)
    if all(x == -100 for x in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def sparse_loss(model, ids: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
    """Next-token loss with logits only where a label follows (logits_to_keep = those positions): full logits are
    seq x 248k vocab in float32, 14 GiB at 16k tokens, which ran the 27B out of memory (17.11.2026). Same value as
    the model's own shifted cross-entropy."""
    keep = (lab[0, 1:] != -100).nonzero().squeeze(-1)
    logits = model(input_ids=ids, logits_to_keep=keep).logits[0]
    return torch.nn.functional.cross_entropy(logits.float(), lab[0, keep + 1])


def lora_targets(model) -> list[str]:
    skip = ("visual", "vision", "lm_head", "embed")
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and not any(s in name for s in skip):
            if name.endswith(".gate") or name.endswith("shared_expert_gate"):  # MoE router: leave it alone
                continue
            out.append(name)
    return out


def load_model(path, dtype, device: str):
    """The base model with one copy of its weights: built directly on the device (random init), then every tensor is
    copied in from the safetensors files one at a time. transformers' from_pretrained held a second copy while
    loading, which is 2 x 65 GB for the 35B coder on the Spark's 121 GB unified memory (T1, 16.11.2026).
    Checkpoint names model.language_model.* (Qwen3.5/3.6 multimodal) map to the causal-LM's model.*; weights the
    causal LM has no place for (vision tower, MTP head) are skipped."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    path = Path(path)
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path), dtype=dtype)
    own = dict(model.state_dict(keep_vars=True))
    idx = path / "model.safetensors.index.json"
    files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
    done = set()
    with torch.no_grad():
        for f in files:
            with safe_open(str(path / f), framework="pt", device=str(device)) as sf:
                for k in sf.keys():
                    name = k if k in own else k.replace("model.language_model.", "model.", 1)
                    if name in own:
                        own[name].copy_(sf.get_tensor(k))
                        done.add(name)
    tied = {"lm_head.weight"} if getattr(model.config.get_text_config(), "tie_word_embeddings", False) else set()
    missing = set(own) - done - tied - {n for n, _ in model.named_buffers()}
    if missing:
        raise ValueError(f"{len(missing)} weights not in the checkpoint, e.g. {sorted(missing)[:3]}")
    return model


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="forge.train_lora")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=0, help="token budget (0 = all examples), one epoch")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--eval-frac", type=float, default=0.03)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-mem-gb", type=float, default=100,
                    help="cap on what PyTorch may allocate on the GPU; on the Spark's unified memory going over it "
                         "raises a clean out-of-memory error instead of swapping the whole machine")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (smoke tests)")
    a = ap.parse_args(argv)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    exs = load(Path(a.data))
    random.Random(a.seed).shuffle(exs)
    n_eval = max(1, int(len(exs) * a.eval_frac))
    eval_raw, train_raw = exs[:n_eval], exs[n_eval:]
    train, used = [], 0
    for ex in train_raw:
        e = encode(tok, ex, a.max_len)
        if e is None:
            continue
        train.append(e)
        used += len(e["input_ids"])
        if a.max_tokens and used >= a.max_tokens:
            break
    evals = [e for e in (encode(tok, ex, a.max_len) for ex in eval_raw) if e]
    total_steps = math.ceil(len(train) / a.accum)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)

    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    if a.device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_mem_gb * 2**30 / total))
    model = load_model(a.model, dtype, a.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    ckpt = out / "checkpoint"
    state = json.loads((out / "state.json").read_text()) if (out / "state.json").exists() else {"step": 0, "seen": 0}
    if (ckpt / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, ckpt, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0,
                                                 target_modules=lora_targets(model), task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    if (out / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(out / "optimizer.pt", map_location=a.device, weights_only=True))

    run = {"model": a.model, "data": a.data, "examples": len(train), "eval_examples": len(evals), "train_tokens": used,
           "total_steps": total_steps, "trainable_params": sum(p.numel() for p in params), **vars(a)}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    log = open(out / "log.jsonl", "a")

    def step_loss(e) -> tuple[torch.Tensor, int]:
        ids = torch.tensor([e["input_ids"]], device=a.device)
        lab = torch.tensor([e["labels"]], device=a.device)
        return sparse_loss(model, ids, lab), int((lab != -100).sum())

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        tot = n = 0
        for e in evals:
            loss, k = step_loss(e)
            tot += float(loss) * k
            n += k
        model.train()
        return tot / max(1, n)

    def save():
        model.save_pretrained(ckpt)
        torch.save(opt.state_dict(), out / "optimizer.pt")
        (out / "state.json").write_text(json.dumps(state) + "\n")

    model.train()
    first_eval = evaluate() if state["step"] == 0 else None
    if first_eval is not None:
        log.write(json.dumps({"step": 0, "eval_loss": round(first_eval, 4)}) + "\n")
        log.flush()
    t0, i = time.time(), state["seen"]
    while state["step"] < total_steps and i < len(train):
        for g in opt.param_groups:
            g["lr"] = lr_at(state["step"], total_steps, a.lr, a.warmup)
        losses, toks = [], 0
        for _ in range(a.accum):
            if i >= len(train):
                break
            loss, _k = step_loss(train[i])
            (loss / a.accum).backward()
            losses.append(float(loss.detach()))
            toks += len(train[i]["input_ids"])
            i += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        state["seen"] = i
        rec = {"step": state["step"], "loss": round(sum(losses) / len(losses), 4), "lr": opt.param_groups[0]["lr"],
               "tokens": toks, "tok_per_s": round(toks / max(1e-6, time.time() - t0), 1)}
        t0 = time.time()
        if state["step"] % a.save_every == 0 or state["step"] == total_steps:
            rec["eval_loss"] = round(evaluate(), 4)
            save()
        log.write(json.dumps(rec) + "\n")
        log.flush()
    save()
    model.save_pretrained(out)  # final adapter at the top level
    res = {"steps": state["step"], "final_eval_loss": evaluate(), "first_eval_loss": first_eval}
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
