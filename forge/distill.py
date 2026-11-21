"""A3: Claude's attempts on train-split turns through this harness, checked, as LoRA data for a local coder.

  python -m forge.distill data/snapshots/v0002 --n 100            # run (resumable), then
  python -m forge.distill data/snapshots/v0002 --build             # write data/sft/v0002-distill/coder.jsonl

- Tasks: benchmark C's task builder on the train split (the held-out project never reaches it), so the turns, the
  history window, the work folder and the replayed app/web tools are the same kind as in the benchmark.
- Agent: claude-p:<model> (Claude through the CLI on Malek's plan) with the agent instructions under test (v2).
- Checker: the same model grades each attempt against what really happened (the train turn's recorded outcome),
  with the benchmark's rubric. It is not the benchmark's judge, and no benchmark task is involved.
- Kept: attempts the checker scores >= --min-overall. The loss covers only the new turns (from loss_from on), not
  the replayed history."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from . import backends, bench_c, harness

OUT = Path(__file__).resolve().parent.parent / "data" / "distill"
GRADE = {"type": "object", "required": [*bench_c.CRITERIA, "overall", "reason"],
         "properties": {**{k: {"type": "integer"} for k in (*bench_c.CRITERIA, "overall")}, "reason": {"type": "string"}}}


def train_trajectories(snapshot: Path, role: str = "coder") -> list[dict]:
    """Train segments in the shape bench_c.build_tasks reads (one segment = one trajectory, no repo)."""
    out = []
    for line in open(Path(snapshot) / "train" / f"{role}.jsonl", encoding="utf-8"):
        seg = json.loads(line)
        sub = "/subagents/" in seg["file"] or seg["file"].rsplit("/", 1)[-1].startswith("agent-")
        out.append({"meta": {"file": f"{seg['file']}@{seg.get('segment', 0)}", "subagent": sub,
                             "touched_root": None, "git_root": None}, "steps": seg["steps"]})
    return out


def claude_checker(model: str):
    empty = Path(tempfile.mkdtemp(prefix="claude-p-check-"))

    def ask(prompt: str) -> dict:
        r = subprocess.run(["claude", "-p", "--model", model, "--tools", "", "--no-session-persistence",
                            "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
                            "--output-format", "json", "--system-prompt", bench_c.RUBRIC, "--json-schema", json.dumps(GRADE)],
                           input=prompt, capture_output=True, text=True, cwd=empty, timeout=600)
        d = json.loads(r.stdout)
        if d.get("is_error"):
            raise RuntimeError(str(d.get("result"))[:300])
        return d.get("structured_output") or json.loads(d["result"])
    return ask


def api_checker(model: str):
    """The checker through the API (Malek, 17.11.: use the organisation's API credits instead of claude -p)."""
    import anthropic
    from . import anthropic_backend
    client = anthropic.Anthropic(max_retries=5, timeout=600)

    def ask(prompt: str) -> dict:
        r = client.messages.create(model=model, max_tokens=4000, system=bench_c.RUBRIC,
                                   messages=[{"role": "user", "content": prompt}],
                                   output_config={"effort": "high", "format": {"type": "json_schema",
                                                                               "schema": GRADE | {"additionalProperties": False}}})
        anthropic_backend.USAGE["calls"] += 1
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            anthropic_backend.USAGE[k] += getattr(r.usage, k, 0) or 0
        return json.loads(next(b.text for b in r.content if b.type == "text"))
    return ask


# Opus 5.5 API prices, $ per million tokens (checked against the 17.11 run: $37 for 78 attempts)
PRICE = {"input_tokens": 4.0, "output_tokens": 20.0, "cache_read_input_tokens": 0.20, "cache_creation_input_tokens": 5.0}
TASK_RESERVE_USD = 2.5  # the most one attempt plus its check cost on 17.11 was well under this


def usd(usage: dict) -> float:
    return sum(usage.get(k, 0) * p for k, p in PRICE.items()) / 1e6


def spent(runs: Path) -> float:
    """Dollars already spent by this worker (rows written with "usd"), so a restarted run keeps counting."""
    return sum(json.loads(l).get("usd", 0) for l in open(runs)) if runs.exists() else 0.0


VIA = {"claude-p": ("claude-p:{m}", claude_checker), "api": ("anthropic:{m}", api_checker)}


def example(ep: dict, role: str = "coder") -> dict:
    """Episode -> chat example (forge.sft format) with loss_from = index of the first new message."""
    msgs = [{"role": "system", "content": ep["system"]}, *ep["history"], {"role": "user", "content": ep["prompt"]}]
    loss_from, n = len(msgs), 0
    for t in ep["transcript"]:
        if "assistant" not in t:
            continue
        m = {"role": "assistant", "content": t.get("assistant") or ""}
        if t.get("reasoning"):  # Claude's summarised thinking (API runs): the chat template renders it as <think>
            m["reasoning_content"] = t["reasoning"]
        calls = [c for c in t.get("calls", [])]
        if calls:
            m["tool_calls"] = []
            for c in calls:
                n += 1
                m["tool_calls"].append({"id": f"call_{n}", "type": "function", "function": {
                    "name": c["name"], "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False)}})
        msgs.append(m)
        results = t.get("results", [])
        for i, tc in enumerate(m.get("tool_calls", [])):
            if tc["function"]["name"] == "done":
                continue
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": results[i] if i < len(results) else "(not run)"})
    return {"messages": msgs, "tools": ep["tools"], "loss_from": loss_from, "source": {"role": role}}


def run(snapshot: Path, model: str, n: int, seed: int, agent: str, tag: str = "", via: str = "claude-p",
        budget_usd: float = 0.0) -> Path:
    snapshot = Path(snapshot)
    spec, checker = VIA[via]
    spec = spec.format(m=model)
    out = OUT / snapshot.name / backends.label(spec)
    out.mkdir(parents=True, exist_ok=True)
    tasks_file = out / f"tasks_n{n}_seed{seed}.json"
    if tasks_file.exists():
        tasks = json.loads(tasks_file.read_text())
    else:
        tasks = bench_c.build_tasks(train_trajectories(snapshot), n=n, seed=seed)
        tasks_file.write_text(json.dumps(tasks, ensure_ascii=False) + "\n")
    trajs = {t["meta"]["file"]: t for t in train_trajectories(snapshot)}
    runs = out / (f"runs-{tag}.jsonl" if tag else "runs.jsonl")  # one file per worker: parallel workers never interleave
    done = {json.loads(l)["id"] for d in run_dirs(snapshot, model) for f in d.glob("runs*.jsonl") for l in open(f)}
    schemas = json.loads((snapshot / "tools.json").read_text())
    chat, ask = backends.chat(spec), checker(model)
    from . import anthropic_backend
    if via == "api" and not budget_usd:
        raise SystemExit("--via api needs --budget-usd (Malek's credits; the worker stops before going over)")
    before = spent(runs)
    for task in tasks:
        if task["id"] in done:
            continue
        if via == "api" and before + usd(anthropic_backend.USAGE) + TASK_RESERVE_USD > budget_usd:
            print(json.dumps({"stopped": "budget", "spent_usd": round(before + usd(anthropic_backend.USAGE), 2),
                              "budget_usd": budget_usd}))
            break
        start = usd(anthropic_backend.USAGE)
        row = bench_c.run_task(task, trajs[task["file"]]["steps"], chat, ask, schemas=schemas, agent=agent,
                               keep=True, null_control=False)
        if row["status"] != "ok":  # usage limit or CLI failure: stop, the next run resumes here
            print(json.dumps({"stopped": task["id"], "error": row.get("error")}))
            break
        row |= {"agent": agent, "model": model, "via": via}
        if via == "api":
            row["usage_so_far"] = dict(anthropic_backend.USAGE)
            row["usd"] = round(usd(anthropic_backend.USAGE) - start, 4)
        with open(runs, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"id": task["id"][-40:], "overall": row["grade"].get("overall"), "steps": row["steps"],
                          "s": row["seconds"]}), flush=True)
    return runs


def reads_logs(row: dict) -> bool:
    return ".claude/projects" in json.dumps(row["episode"]["transcript"], ensure_ascii=False)


def run_dirs(snapshot: Path, model: str) -> list[Path]:
    """Attempts of one model from every route (claude -p and the API), so no turn is run twice and build merges all."""
    return [d for v in VIA.values() if (d := OUT / Path(snapshot).name / backends.label(v[0].format(m=model))).exists()]


def build(snapshot: Path, model: str, min_overall: int) -> dict:

    dst = Path(__file__).resolve().parent.parent / "data" / "sft" / f"{Path(snapshot).name}-distill"
    dst.mkdir(parents=True, exist_ok=True)
    rows = list({r["id"]: r for d in run_dirs(snapshot, model) for f in sorted(d.glob("runs*.jsonl"))
                 for r in map(json.loads, open(f))}.values())
    # an attempt that read the session logs may have read the turn's own outcome: it teaches a lookup shortcut that
    # finds nothing on new work, so it is left out until reads are cut at the task's time (Malek decides, 16.11.)
    peeked = [r for r in rows if reads_logs(r)]
    kept = [r for r in rows if (r["grade"].get("overall") or 0) >= min_overall and not reads_logs(r)]
    with open(dst / "coder.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(example(r["episode"]), ensure_ascii=False) + "\n")
    stats = {"runs": len(rows), "kept": len(kept), "min_overall": min_overall, "dropped_read_logs": len(peeked),
             "mean_overall": round(sum(r["grade"].get("overall", 0) for r in rows) / max(1, len(rows)), 2)}
    (dst / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="forge.distill")
    ap.add_argument("snapshot")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--agent", default="v2")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--min-overall", type=int, default=4)
    ap.add_argument("--tag", default="", help="worker name: its own runs-<tag>.jsonl")
    ap.add_argument("--via", default="claude-p", choices=sorted(VIA), help="claude -p (plan) or the API (credits)")
    ap.add_argument("--budget-usd", type=float, default=0.0, help="API only: hard cap for this worker (--tag), across restarts")
    a = ap.parse_args()
    if a.build:
        print(json.dumps(build(Path(a.snapshot), a.model, a.min_overall), indent=1))
    else:
        run(Path(a.snapshot), a.model, a.n, a.seed, a.agent, a.tag, a.via, a.budget_usd)
