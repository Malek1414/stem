"""Training examples from a snapshot's train split: chat messages with OpenAI-style tool calls, ready for a
chat-template SFT trainer (loss on assistant turns only).

- One example per window of a segment. Windows hold whole steps and never split a tool call from its result;
  a window that starts mid-turn opens with the turn's user prompt so the assistant turns have their ask.
- Tool outputs are clipped (head + tail); Claude's thinking is empty in the logs and is dropped.
- Secrets that show up in tool outputs or prompts (API keys, tokens, private keys) are redacted, so the model
  cannot memorise them.
- Tools offered per example: the harness's file/shell tools plus every tool the window uses, with the schemas
  the snapshot recorded.
The held-out project never reaches this split, so benchmark tasks do not leak into training."""
from __future__ import annotations

import json
import re
from pathlib import Path

CORE = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
SYSTEM = {
    "coder": "You are a coding agent on the user's Mac. Use the tools to read, run and change code until the task "
             "is done, then tell the user what you did in a few plain sentences.",
    "operator": "You are an operator agent on the user's Mac: you drive apps, the browser and the shell with tools, "
                "precisely and quickly, and hand coding work to the coder with a clear task card.",
}
SECRETS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED PRIVATE KEY]"),
    (re.compile(r"\b(sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,})"), "[REDACTED KEY]"),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{30,})"), "[REDACTED TOKEN]"),
    (re.compile(r"\b(hf_[A-Za-z0-9]{30,})"), "[REDACTED TOKEN]"),
    (re.compile(r"\b(xox[abpsr]-[A-Za-z0-9-]{10,})"), "[REDACTED TOKEN]"),
    (re.compile(r"\b(AKIA[0-9A-Z]{16})\b"), "[REDACTED KEY]"),
    (re.compile(r"\b(AIza[0-9A-Za-z_-]{35})\b"), "[REDACTED KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}"), "[REDACTED JWT]"),
    (re.compile(r"(?i)((?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*)(['\"]?)[^\s'\"]{6,}\2"), r"\1[REDACTED]"),
]


# Behaviour Malek has since forbidden must not be learned: plain ssh/scp/rsync to the DGX Spark crashed it
# (CLAUDE.md, 31.10.2026). Windows with such a call are dropped.
FORBIDDEN = re.compile(r"\b(ssh|scp|rsync)\b[^|;&\n]*\b(spark(-6da7)?|192\.168\.18\.199)\b")


def forbidden(window: list[dict]) -> bool:
    return any(FORBIDDEN.search(json.dumps(c["input"], ensure_ascii=False)) for b in window if "calls" in b for c in b["calls"])


def redact(text: str) -> str:
    for pat, rep in SECRETS:
        text = pat.sub(rep, text)
    return text


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit * 2 // 3] + f"\n[... {len(text) - limit} chars cut ...]\n" + text[-limit // 3:]


def _blocks(steps: list[dict], out_chars: int) -> list[dict]:
    """Steps -> blocks that must stay together: a user prompt, or one assistant turn (its text and calls) with the
    results of those calls."""
    blocks, cur, results = [], None, {}
    for s in steps:
        if s["kind"] == "tool_result":
            results[s.get("id")] = s
    for s in steps:
        k = s["kind"]
        if k == "user_prompt":
            if s.get("meta") or not s.get("text", "").strip():
                continue
            cur = None
            blocks.append({"user": redact(s["text"])})
        elif k == "assistant_text" and s.get("text", "").strip():
            if cur is None or cur["calls"]:
                cur = {"text": [], "calls": []}
                blocks.append(cur)
            cur["text"].append(redact(s["text"]))
        elif k == "tool_call":
            if cur is None:
                cur = {"text": [], "calls": []}
                blocks.append(cur)
            r = results.get(s.get("id"), {})
            out = r.get("output", "")
            out = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            cur["calls"].append({"name": s["name"], "input": s.get("input") or {},
                                 "output": redact(clip(out or "(no output)", out_chars)),
                                 "error": bool(r.get("is_error"))})
    return [b for b in blocks if "user" in b or b["text"] or b["calls"]]


def _messages(blocks: list[dict]) -> list[dict]:
    msgs, n = [], 0
    for b in blocks:
        if "user" in b:
            msgs.append({"role": "user", "content": b["user"]})
            continue
        m = {"role": "assistant", "content": "\n\n".join(b["text"])}
        if b["calls"]:
            m["tool_calls"] = []
            for c in b["calls"]:
                n += 1
                m["tool_calls"].append({"id": f"call_{n}", "type": "function",
                                        "function": {"name": c["name"], "arguments": json.dumps(c["input"], ensure_ascii=False)}})
        msgs.append(m)
        for i, c in enumerate(b["calls"]):
            msgs.append({"role": "tool", "tool_call_id": m["tool_calls"][i]["id"], "content": c["output"]})
    return msgs


def _size(b: dict) -> int:
    return len(json.dumps(b, ensure_ascii=False))


def windows(blocks: list[dict], max_chars: int) -> list[list[dict]]:
    """Cut into windows of at most max_chars (a single oversized block stands alone). A window that does not start
    with a user prompt gets the last prompt in front, so the model always sees what it is working on."""
    out, cur, size, last_user = [], [], 0, None
    for b in blocks:
        if cur and size + _size(b) > max_chars:
            out.append(cur)
            cur, size = ([last_user] if last_user and "user" not in b else []), (_size(last_user) if last_user and "user" not in b else 0)
        cur.append(b)
        size += _size(b)
        if "user" in b:
            last_user = b
    if cur:
        out.append(cur)
    # a window needs a prompt first and at least one assistant turn to learn from
    return [w for w in out if "user" in w[0] and any("user" not in b for b in w)]



def examples(segment: dict, schemas: dict, max_chars: int = 40_000, out_chars: int = 3_000) -> list[dict]:
    role = segment.get("agent", "coder")
    res = []
    for w in windows(_blocks(segment["steps"], out_chars), max_chars):
        if forbidden(w):
            res.append(None)
            continue
        used = sorted({c["name"] for b in w if "calls" in b for c in b["calls"]})
        names = CORE + [u for u in used if u not in CORE]
        tools = [{"type": "function", "function": {"name": n, "description": n,
                                                   "parameters": schemas.get(n) or {"type": "object", "properties": {}}}}
                 for n in names]
        res.append({"messages": [{"role": "system", "content": SYSTEM.get(role, SYSTEM["coder"])}, *_messages(w)],
                    "tools": tools, "source": {"file": segment.get("file"), "segment": segment.get("segment"), "agent": role}})
    return res


def build(snapshot: Path, out: Path, max_chars: int = 40_000, out_chars: int = 3_000) -> dict:
    snapshot, out = Path(snapshot), Path(out)
    schemas = json.loads((snapshot / "tools.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    stats = {"snapshot": snapshot.name, "max_chars": max_chars, "out_chars": out_chars}
    for role in ("coder", "operator"):
        n_seg = n_ex = n_calls = n_red = chars = n_drop = 0
        with open(out / f"{role}.jsonl", "w") as f:
            for line in open(snapshot / "train" / f"{role}.jsonl"):
                seg = json.loads(line)
                n_seg += 1
                for ex in examples(seg, schemas, max_chars, out_chars):
                    if ex is None:
                        n_drop += 1
                        continue
                    s = json.dumps(ex, ensure_ascii=False)
                    n_ex += 1
                    chars += len(s)
                    n_calls += sum(len(m.get("tool_calls", [])) for m in ex["messages"])
                    n_red += s.count("[REDACTED")
                    f.write(s + "\n")
        stats[role] = {"segments": n_seg, "examples": n_ex, "tool_calls": n_calls, "redactions": n_red, "dropped_forbidden": n_drop,
                       "approx_tokens": chars // 3}
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    return stats
