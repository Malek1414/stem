"""Benchmark C: judged outcomes on the held-out project.

Covers the requests benchmark B can't check mechanically (research, outreach, planning, anything that
didn't edit the repo). The model does the request in the sandboxed harness (no network), and a pinned
local judge model grades its result against what actually happened in the session, with a fixed rubric.
Reproducible because the judge's weights digest, the rubric hash, temperature 0 and the seed are recorded.
Limitation: without network or apps, operator requests are judged on approach and plan, not on effect."""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import urllib.request
from pathlib import Path

from . import backends, bench_a, bench_b, harness, net
from .stats import mean_ci

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DEFAULT_JUDGE = "gpt-oss:120b"
CRITERIA = ("understood", "approach", "correctness", "completeness")
RUBRIC = """You grade an AI agent's work on one request from a real working session.
You get: the request, what actually happened when it was done for real (the reference), and the
agent's attempt (its actions and final message). The agent had no network and no apps, so judge
actions it could not run by whether they were the right actions with the right inputs.
Score each from 1 (bad) to 5 (as good as the reference or better):
- understood: did it grasp what the user actually wanted?
- approach: were the steps and tools the right ones, in a sensible order?
- correctness: is what it produced or concluded right?
- completeness: did it cover everything the request asked for?
Reply with JSON only: {"understood": n, "approach": n, "correctness": n, "completeness": n,
"overall": n, "reason": "one sentence"}"""
RUBRIC_SHA = hashlib.sha256(RUBRIC.encode()).hexdigest()


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " [...]"


def build_tasks(trajectories: list[dict], n: int = 20, seed: int = 0) -> list[dict]:
    pool = []
    for t in trajectories:
        m = t["meta"]
        if m.get("subagent"):
            continue
        repo = m.get("touched_root") or m.get("git_root")
        steps = t["steps"]
        starts = bench_b.prompts(steps)
        for a, b in zip(starts, starts[1:] + [len(steps)]):
            turn = steps[a:b]
            calls = [s for s in turn if s["kind"] == "tool_call"]
            edits_repo = repo and any(s["name"] in bench_b.EDIT_TOOLS and
                                      str(s["input"].get("file_path", "")).startswith(repo + "/") for s in calls)
            replies = [s["text"] for s in turn if s["kind"] == "assistant_text" and s["text"].strip()]
            if edits_repo or not replies or len(steps[a]["text"].strip()) < 8:
                continue
            pool.append({
                "id": f"{m['file']}#{a}", "file": m["file"], "step": a, "ts": steps[a]["ts"], "repo": repo,
                "prompt": steps[a]["text"],
                "reference": {
                    "final_reply": _clip(replies[-1], 3_000),
                    "actions": [f"{s['name']} {_clip(json.dumps(s['input'], ensure_ascii=False), 160)}" for s in calls[:30]],
                },
            })
    pool.sort(key=lambda x: (x["file"], x["step"]))
    if len(pool) > n:
        pool = sorted(random.Random(seed).sample(pool, n), key=lambda x: (x["file"], x["step"]))
    return pool


def tasks_for(snapshot: Path, n: int = 20) -> list[dict]:
    snapshot = Path(snapshot)
    frozen = RESULTS / "bench_c_tasks" / f"{snapshot.name}.json"
    if frozen.exists():
        return json.loads(frozen.read_text())
    tasks = build_tasks(bench_a.load_heldout(snapshot), n=n)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(tasks, ensure_ascii=False, sort_keys=True) + "\n")
    return tasks


def ollama_json(model: str, base_url: str = net.SYNC_OLLAMA, seed: int = 0, timeout: int = 900):
    def ask(prompt: str) -> dict:
        body = {"model": model, "stream": False, "format": "json",
                "messages": [{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt}],
                "options": {"temperature": 0, "seed": seed}}
        req = urllib.request.Request(f"{base_url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with net.urlopen(req, timeout=timeout) as r:
            return json.loads(json.loads(r.read())["message"]["content"])
    return ask


def judge(ask, task: dict, attempt: dict) -> dict:
    prompt = json.dumps({
        "request": _clip(task["prompt"], 4_000),
        "reference": task["reference"],
        "agent_attempt": {"actions": attempt["actions"][:30], "final_message": _clip(attempt["final"], 3_000)},
    }, ensure_ascii=False, indent=1)
    for _ in range(2):
        try:
            g = ask(prompt)
            scores = {k: max(1, min(5, int(g[k]))) for k in (*CRITERIA, "overall")}
            return scores | {"reason": str(g.get("reason", ""))[:300]}
        except Exception as e:  # malformed judge output: retry once, then record a failure
            err = repr(e)
    return {"error": err}


def missing_tools_note(steps: list[dict]) -> str:
    have = {t["function"]["name"] for t in harness.TOOLS}
    used = sorted({s["name"] for s in steps if s.get("kind") == "tool_call" and s.get("name") not in have})
    if not used:
        return ""
    return (" These tools from the original session do not exist in this run: " + ", ".join(used[:40])
            + ". Do not call them; say exactly what you would do with them instead.")


def run_task(task: dict, steps: list[dict], chat: harness.ChatFn, ask, root: Path = bench_b.WORK,
             max_steps: int = 15, wall_seconds: int = 600) -> dict:
    root = Path(root)
    if task.get("repo") and Path(task["repo"], ".git").exists():
        commit = bench_b._git("rev-list", "-1", f"--before={task['ts']}", "HEAD", cwd=task["repo"]).strip()
        wd = bench_b.prepare({"id": task["id"], "repo": task["repo"], "commit": commit}, root)
        home = task["repo"]
    else:
        wd = root / hashlib.sha1(task["id"].encode()).hexdigest()[:12]
        shutil.rmtree(wd, ignore_errors=True)
        wd.mkdir(parents=True)
        home = None
    try:
        history = bench_a.render(steps[: task["step"]], budget_chars=harness.HISTORY_CHARS)
        prompt = task["prompt"]
        if home:
            history, prompt = bench_b._rewrite(history, home, str(wd)), bench_b._rewrite(prompt, home, str(wd))
        system = (f"You are an agent working on the user's Mac. Your work folder is {wd}. There is no network "
                  "and no apps in this run: when a step would need them, say exactly what you would do and with "
                  "what inputs. Finish with a final message to the user, then call done."
                  + missing_tools_note(steps[: task["step"]]))
        run = harness.run_agent(chat, system, history, prompt, wd, max_steps=max_steps, wall_seconds=wall_seconds,
                                force_final=True)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    if run.get("infra_error"):  # the model never answered: do not judge, rerun later
        return {"id": task["id"], "prompt": task["prompt"][:300], "status": "infra_error", "error": run["infra_error"],
                "steps": run["steps"], "seconds": run["seconds"]}
    texts = [t.get("assistant", "") for t in run["transcript"] if t.get("assistant")]
    attempt = {"actions": [f"{c['name']} {_clip(json.dumps(c.get('arguments'), ensure_ascii=False), 160)}"
                           for t in run["transcript"] for c in t.get("calls", [])],
               "final": texts[-1] if texts else ""}
    grade = judge(ask, task, attempt)
    # null control: an empty attempt must score low, or the judge is not measuring the work
    # (Zheng et al. 2024, "Cheating automatic LLM benchmarks", arXiv 2410.07137)
    null = judge(ask, task, {"actions": [], "final": ""})
    return {"id": task["id"], "prompt": task["prompt"][:300], "status": "ok", "grade": grade, "null_grade": null, "attempt": attempt,
            "steps": run["steps"], "seconds": run["seconds"]}


def run(snapshot: Path, model: str, judge_model: str = DEFAULT_JUDGE, base_url: str = net.SYNC_OLLAMA,
        limit: int | None = None, seed: int = 0) -> Path:
    snapshot = Path(snapshot)
    tasks = tasks_for(snapshot)
    tasks = tasks[:limit] if limit else tasks
    trajs = {t["meta"]["file"]: t for t in bench_a.load_heldout(snapshot)}
    settings = {"temperature": 0, "seed": seed, "max_steps": 15, "wall_seconds": 600, "harness": harness.HARNESS_VERSION,
                "num_ctx": harness.NUM_CTX, "history_chars": harness.HISTORY_CHARS, "forced_final": True,
                "judge": judge_model, "rubric_sha256": RUBRIC_SHA, "null_control": True}
    out = RESULTS / re.sub(r"[^A-Za-z0-9._-]", "_", backends.label(model)) / snapshot.name / "bench_c.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    done = ({r["id"]: r for r in prev.get("tasks", []) if r.get("status", "ok") != "infra_error"}
            if prev.get("settings") == settings else {})
    record = {"model": backends.record(model), "judge": bench_a.model_digest(base_url, judge_model)}
    chat, ask = backends.chat(model, seed=seed), ollama_json(judge_model, base_url, seed=seed)
    rows = []
    for task in tasks:
        rows.append(done.get(task["id"]) or run_task(task, trajs[task["file"]]["steps"], chat, ask))
        _save(out, model, snapshot, settings, record, rows)
    _save(out, model, snapshot, settings, record, rows)
    return out


def _save(out: Path, model: str, snapshot: Path, settings: dict, record: dict, rows: list[dict]) -> None:
    infra = [r for r in rows if r.get("status") == "infra_error"]
    rows_ok = [r for r in rows if r.get("status") != "infra_error"]
    graded = [r["grade"] for r in rows_ok if "overall" in r["grade"]]
    mean = lambda k: round(sum(g[k] for g in graded) / len(graded), 3) if graded else None
    nulls = [r["null_grade"]["overall"] for r in rows_ok if "overall" in r.get("null_grade", {})]
    report = {"benchmark": "C", "model": model, "model_record": record, "snapshot": snapshot.name,
              "settings": settings, "rubric": RUBRIC,
              "overall": {"tasks": len(rows_ok), "infra_errors": len(infra), "graded": len(graded),
                          **{k: mean(k) for k in (*CRITERIA, "overall")},
                          "ci95_overall": mean_ci([g["overall"] for g in graded]),
                          "null_control_overall": round(sum(nulls) / len(nulls), 3) if nulls else None,
                          "mean_seconds": round(sum(r["seconds"] for r in rows_ok) / len(rows_ok), 1) if rows_ok else None},
              "tasks": rows}
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
