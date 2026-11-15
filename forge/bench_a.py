"""Benchmark A: next-action prediction on the held-out project.

Each item is one real tool call from a held-out session. The model sees the session up to that point and
must produce the next tool call. Scored against what was actually done. Deterministic: seeded sampling,
temperature 0, fixed seed, fixed context budget, and the model's weights digest is recorded."""
from __future__ import annotations

import json
import random
import re
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

from . import segment as sg
from .stats import wilson

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CORE_TOOLS = ["Bash", "Read", "Write", "Edit", "Grep", "Glob", "WebSearch", "WebFetch", "Agent", "ToolSearch"]
IGNORED_ARGS = {"description"}
SYSTEM = (
    "You are an agent working on the user's Mac through tools. Given the conversation so far, "
    "respond with exactly one tool call: the next action to take."
)


def load_heldout(snapshot: Path) -> list[dict]:
    with open(Path(snapshot) / "heldout" / "trajectories.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_items(snapshot: Path, n: int = 300, seed: int = 0) -> list[dict]:
    items = []
    for t in load_heldout(snapshot):
        labels = sg.label_steps(t["steps"])
        for i, s in enumerate(t["steps"]):
            if s["kind"] == "tool_call":
                items.append({
                    "id": f"{t['meta']['file']}#{i}", "file": t["meta"]["file"], "step": i,
                    "agent": labels[i], "gold": {"name": s["name"], "input": s["input"]},
                })
    order = lambda x: (x["file"], x["step"])
    items.sort(key=order)
    if len(items) > n:
        items = sorted(random.Random(seed).sample(items, n), key=order)
    return items


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n[... {len(text) - limit} chars cut ...]\n" + text[-half:]


def render(steps: list[dict], budget_chars: int = 32_000, per_result: int = 2_000) -> list[dict]:
    """Steps before the target -> Ollama chat messages. Oldest messages after the first prompt are
    dropped until the context fits the budget."""
    msgs: list[dict] = []
    for s in steps:
        k = s["kind"]
        if k == "user_prompt":
            msgs.append({"role": "user", "content": s["text"]})
        elif k == "assistant_text":
            if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls"):
                msgs[-1]["content"] += "\n" + s["text"]
            else:
                msgs.append({"role": "assistant", "content": s["text"]})
        elif k == "tool_call":
            call = {"function": {"name": s["name"], "arguments": s["input"]}}
            if msgs and msgs[-1]["role"] == "assistant":
                msgs[-1].setdefault("tool_calls", []).append(call)
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif k == "tool_result":
            msgs.append({"role": "tool", "content": _clip(s["output"], per_result)})
    size = lambda m: len(m.get("content", "")) + len(json.dumps(m.get("tool_calls", [])))
    first_user = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
    total = sum(size(m) for m in msgs)
    keep_head = msgs[: first_user + 1] if first_user is not None else []
    tail = msgs[first_user + 1:] if first_user is not None else msgs
    while tail and total > budget_chars:
        total -= size(tail.pop(0))
    while tail and tail[0]["role"] == "tool":  # never start on an orphaned tool result
        tail.pop(0)
    return keep_head + tail


def tools_for(steps: list[dict], registry: dict) -> list[dict]:
    names = list(CORE_TOOLS) + sorted({s["name"] for s in steps if s["kind"] == "tool_call"} - set(CORE_TOOLS))
    return [
        {"type": "function", "function": {
            "name": n, "description": "",
            "parameters": registry.get(n, {"type": "object", "properties": {}}),
        }}
        for n in names
    ]


def _norm(v):
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v).strip()
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in sorted(v.items()) if k not in IGNORED_ARGS}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def score(gold: dict, pred: dict | None) -> dict:
    if not pred or pred.get("name") != gold["name"]:
        return {"name": 0.0, "exact": 0.0, "args": 0.0}
    g, p = _norm(gold["input"] or {}), _norm(pred.get("input") or {})
    keys = sorted(set(g) | set(p))
    if not keys:
        return {"name": 1.0, "exact": 1.0, "args": 1.0}
    sims = []
    for k in keys:
        a, b = g.get(k), p.get(k)
        if a == b:
            sims.append(1.0)
        elif isinstance(a, str) and isinstance(b, str):
            sims.append(SequenceMatcher(None, a, b).ratio())
        else:
            sims.append(0.0)
    return {"name": 1.0, "exact": float(g == p), "args": sum(sims) / len(sims)}


def _post(url: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_digest(base_url: str, model: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=300) as r:
        tags = json.loads(r.read())
    with urllib.request.urlopen(f"{base_url}/api/version", timeout=300) as r:
        version = json.loads(r.read()).get("version")
    names = {model, model if ":" in model else f"{model}:latest"}
    m = next((x for x in tags.get("models", []) if x.get("name") in names or x.get("model") in names), {})
    return {"digest": m.get("digest"), "quantization": (m.get("details") or {}).get("quantization_level"),
            "runtime": "ollama", "runtime_version": version}


def run(snapshot: Path, model: str, base_url: str = "http://localhost:11434", n: int = 300,
        limit: int | None = None, num_ctx: int | None = None, seed: int = 0) -> Path:
    snapshot = Path(snapshot)
    registry = json.loads((snapshot / "tools.json").read_text())
    trajs = {t["meta"]["file"]: t for t in load_heldout(snapshot)}
    items = build_items(snapshot, n=n, seed=seed)
    if limit:
        items = items[:limit]
    out = RESULTS / re.sub(r"[^A-Za-z0-9._-]", "_", model) / snapshot.name / "bench_a.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = {"temperature": 0, "seed": seed, "num_ctx": num_ctx or "model default", "budget_chars": 32_000, "per_result": 2_000}
    done = {}
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get("settings") == settings:  # resume only a run with identical settings
            done = {r["id"]: r for r in prev.get("items", [])}
    digest = model_digest(base_url, model)
    rows = []
    for it in items:
        if it["id"] in done:
            rows.append(done[it["id"]])
            continue
        steps = trajs[it["file"]]["steps"][: it["step"]]
        body = {
            "model": model, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM}] + render(steps),
            "tools": tools_for(trajs[it["file"]]["steps"], registry),
            "options": {"temperature": 0, "seed": seed} | ({"num_ctx": num_ctx} if num_ctx else {}),
        }
        t0 = time.time()
        try:
            resp = _post(f"{base_url}/api/chat", body)
            calls = (resp.get("message") or {}).get("tool_calls") or []
            fn = calls[0]["function"] if calls else None
            args = fn.get("arguments") if fn else None
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            pred = {"name": fn["name"], "input": args or {}} if fn else None
            err = None
        except Exception as e:  # recorded, scored as a miss
            pred, err = None, repr(e)
        rows.append({**it, "pred": pred, "error": err, "seconds": round(time.time() - t0, 2),
                     "score": score(it["gold"], pred)})
        _save(out, model, snapshot, settings, rows, digest)
    _save(out, model, snapshot, settings, rows, digest)
    return out


def _save(out: Path, model: str, snapshot: Path, settings: dict, rows: list[dict], digest: dict) -> None:
    def agg(rs):
        if not rs:
            return None
        return {k: round(sum(r["score"][k] for r in rs) / len(rs), 4) for k in ("name", "exact", "args")} | {
            "n": len(rs),
            "ci95_name": wilson(sum(r["score"]["name"] for r in rs), len(rs)),
            "ci95_exact": wilson(sum(r["score"]["exact"] for r in rs), len(rs)),
            "mean_seconds": round(sum(r["seconds"] for r in rs) / len(rs), 1),
        }
    manifest = json.loads((snapshot / "manifest.json").read_text())
    report = {
        "benchmark": "A", "model": model, "model_record": digest, "snapshot": snapshot.name,
        "snapshot_heldout": manifest["heldout_project"], "settings": settings,
        "overall": agg(rows),
        "by_agent": {a: agg([r for r in rows if r["agent"] == a]) for a in (sg.CODER, sg.OPERATOR, sg.NEUTRAL)},
        "items": rows,
    }
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
