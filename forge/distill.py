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


def example(ep: dict, role: str = "coder") -> dict:
    """Episode -> chat example (forge.sft format) with loss_from = index of the first new message."""
    msgs = [{"role": "system", "content": ep["system"]}, *ep["history"], {"role": "user", "content": ep["prompt"]}]
    loss_from, n = len(msgs), 0
    for t in ep["transcript"]:
        if "assistant" not in t:
            continue
        m = {"role": "assistant", "content": t.get("assistant") or ""}
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


def run(snapshot: Path, model: str, n: int, seed: int, agent: str, tag: str = "") -> Path:
    snapshot = Path(snapshot)
    out = OUT / snapshot.name / backends.label(f"claude-p:{model}")
    out.mkdir(parents=True, exist_ok=True)
    tasks_file = out / f"tasks_n{n}_seed{seed}.json"
    if tasks_file.exists():
        tasks = json.loads(tasks_file.read_text())
    else:
        tasks = bench_c.build_tasks(train_trajectories(snapshot), n=n, seed=seed)
        tasks_file.write_text(json.dumps(tasks, ensure_ascii=False) + "\n")
    trajs = {t["meta"]["file"]: t for t in train_trajectories(snapshot)}
    runs = out / (f"runs-{tag}.jsonl" if tag else "runs.jsonl")  # one file per worker: parallel workers never interleave
    done = {json.loads(l)["id"] for f in out.glob("runs*.jsonl") for l in open(f)}  # skip turns any worker did
    schemas = json.loads((snapshot / "tools.json").read_text())
    chat, ask = backends.chat(f"claude-p:{model}"), claude_checker(model)
    for task in tasks:
        if task["id"] in done:
            continue
        row = bench_c.run_task(task, trajs[task["file"]]["steps"], chat, ask, schemas=schemas, agent=agent,
                               keep=True, null_control=False)
        if row["status"] != "ok":  # usage limit or CLI failure: stop, the next run resumes here
            print(json.dumps({"stopped": task["id"], "error": row.get("error")}))
            break
        row |= {"agent": agent, "model": model}
        with open(runs, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"id": task["id"][-40:], "overall": row["grade"].get("overall"), "steps": row["steps"],
                          "s": row["seconds"]}), flush=True)
    return runs


def reads_logs(row: dict) -> bool:
    return ".claude/projects" in json.dumps(row["episode"]["transcript"], ensure_ascii=False)


def build(snapshot: Path, model: str, min_overall: int) -> dict:
    src_dir = OUT / Path(snapshot).name / backends.label(f"claude-p:{model}")
    dst = Path(__file__).resolve().parent.parent / "data" / "sft" / f"{Path(snapshot).name}-distill"
    dst.mkdir(parents=True, exist_ok=True)
    rows = list({r["id"]: r for f in sorted(src_dir.glob("runs*.jsonl")) for r in map(json.loads, open(f))}.values())
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
    a = ap.parse_args()
    if a.build:
        print(json.dumps(build(Path(a.snapshot), a.model, a.min_overall), indent=1))
    else:
        run(Path(a.snapshot), a.model, a.n, a.seed, a.agent, a.tag)
