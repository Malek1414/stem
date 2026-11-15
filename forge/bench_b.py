"""Benchmark B: task replay on the held-out project.

A task is one user request (prompt to next prompt) in a held-out session that edited the project's
repo. The repo is restored at the last commit before that request, the model gets the conversation so
far plus the request, works in a sandboxed clone, and is scored by automatic checks:
  changed:<file>  it changed every file Claude changed
  content:<file>  the file ends within 0.75 line-similarity of what Claude's edits produced
  valid:<file>    changed .py files still compile, changed .json files still parse
Checks come from the session itself, so some are wrong (an accepted cost of fully automatic checks)."""
from __future__ import annotations

import hashlib
import json
import py_compile
import re
import shutil
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

from . import net, backends, bench_a, duo, harness
from .stats import wilson

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
WORK = REPO / "work"
EDIT_TOOLS = {"Edit", "Write", "MultiEdit"}
SIMILAR = 0.75


def _git(*args: str, cwd: str | Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout


def _apply(text: str | None, call: dict) -> str | None:
    """Replay one of Claude's edits on known content; None when the edit can't be replayed exactly."""
    inp = call["input"]
    if call["name"] == "Write":
        return str(inp.get("content", ""))
    if text is None:
        return None
    edits = inp.get("edits") if call["name"] == "MultiEdit" else [inp]
    for e in edits or []:
        old, new = str(e.get("old_string", "")), str(e.get("new_string", ""))
        if not old or old not in text:
            return None
        text = text.replace(old, new) if e.get("replace_all") else text.replace(old, new, 1)
    return text


def prompts(steps: list[dict]) -> list[int]:
    return [i for i, s in enumerate(steps)
            if s["kind"] == "user_prompt" and not s.get("meta") and not s["text"].lstrip().startswith("<")]


def build_tasks(trajectories: list[dict]) -> list[dict]:
    tasks = []
    for t in trajectories:
        m = t["meta"]
        repo = m.get("touched_root") or m.get("git_root")
        if m.get("subagent") or not repo or not Path(repo, ".git").exists():
            continue
        steps = t["steps"]
        starts = prompts(steps)
        for a, b in zip(starts, starts[1:] + [len(steps)]):
            edits = [s for s in steps[a:b] if s["kind"] == "tool_call" and s["name"] in EDIT_TOOLS
                     and str(s["input"].get("file_path", "")).startswith(repo + "/")]
            if not edits:
                continue
            commit = _git("rev-list", "-1", f"--before={steps[a]['ts']}", "HEAD", cwd=repo).strip()
            if not commit:
                continue
            files: dict[str, dict] = {}
            for e in edits:
                rel = e["input"]["file_path"][len(repo) + 1:]
                if rel not in files:
                    show = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=repo, capture_output=True, text=True)
                    base = show.stdout if show.returncode == 0 else None
                    files[rel] = {"base": base, "expected": base if base is not None else "", "reliable": True,
                                  "existed": base is not None}
                f = files[rel]
                if not f["reliable"]:
                    continue
                # an edit to a file that wasn't committed yet can't be replayed; a Write always can
                nxt = _apply(f["expected"] if f["existed"] else None, e)
                if nxt is None:
                    f["reliable"] = False
                else:
                    f["expected"], f["existed"] = nxt, True
            tasks.append({
                "id": f"{m['file']}#{a}", "file": m["file"], "step": a, "ts": steps[a]["ts"], "repo": repo,
                "commit": commit, "prompt": steps[a]["text"],
                "files": {rel: {"base": f["base"], "expected": f["expected"] if f["reliable"] else None}
                          for rel, f in sorted(files.items())},
            })
    return sorted(tasks, key=lambda x: (x["file"], x["step"]))


def tasks_for(snapshot: Path) -> list[dict]:
    """Tasks are built once per snapshot and frozen, so later commits to the repo can't change them."""
    snapshot = Path(snapshot)
    frozen = RESULTS / "bench_b_tasks" / f"{snapshot.name}.json"
    if frozen.exists():
        return json.loads(frozen.read_text())
    tasks = build_tasks(bench_a.load_heldout(snapshot))
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(tasks, ensure_ascii=False, sort_keys=True) + "\n")
    return tasks


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False).ratio()


def check(task: dict, workdir: Path) -> list[dict]:
    out = []
    for rel, f in task["files"].items():
        p = Path(workdir) / rel
        now = p.read_text(errors="replace") if p.exists() else None
        changed = now is not None and now != f["base"]
        out.append({"check": f"changed:{rel}", "pass": changed})
        if f["expected"] is not None:
            sim = _similar(f["expected"], now or "")
            out.append({"check": f"content:{rel}", "pass": sim >= SIMILAR, "similarity": round(sim, 3)})
        if changed and rel.endswith(".py"):
            try:
                py_compile.compile(str(p), doraise=True, cfile=str(Path(workdir) / ".forge_pyc"))
                ok = True
            except py_compile.PyCompileError:
                ok = False
            out.append({"check": f"valid:{rel}", "pass": ok})
        elif changed and rel.endswith(".json"):
            try:
                json.loads(now)
                ok = True
            except json.JSONDecodeError:
                ok = False
            out.append({"check": f"valid:{rel}", "pass": ok})
    return out


def _rewrite(obj, old: str, new: str):
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [_rewrite(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite(v, old, new) for k, v in obj.items()}
    return obj


def prepare(task: dict, root: Path = WORK) -> Path:
    wd = Path(root) / hashlib.sha1(task["id"].encode()).hexdigest()[:12]
    shutil.rmtree(wd, ignore_errors=True)
    wd.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", task["repo"], str(wd)], check=True)
    subprocess.run(["git", "-C", str(wd), "checkout", "-q", "--detach", task["commit"]], check=True)
    return wd


def run_task(task: dict, steps: list[dict], chat: harness.ChatFn, root: Path = WORK,
             max_steps: int = 30, wall_seconds: int = 900, coder: harness.ChatFn | None = None) -> dict:
    """chat alone: single agent. chat + coder: chat is the operator, coder gets task cards (forge.duo)."""
    wd = prepare(task, root)
    try:
        history = _rewrite(bench_a.render(steps[: task["step"]], budget_chars=harness.HISTORY_CHARS), task["repo"], str(wd))
        system = (f"You are a coding agent. The repository {Path(task['repo']).name} is checked out at {wd}. "
                  "Work only inside that folder. There is no network. Use the tools to make the change the "
                  "user asks for, then call done.")
        prompt = _rewrite(task["prompt"], task["repo"], str(wd))
        if coder is None:
            run = harness.run_agent(chat, system, history, prompt, wd, max_steps=max_steps, wall_seconds=wall_seconds)
        else:
            run = duo.run(chat, coder, system, history, prompt, wd, max_steps=max_steps, wall_seconds=wall_seconds)
        checks = check(task, wd)
        diff = _git("diff", "--stat", cwd=wd)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    passed = sum(c["pass"] for c in checks)
    infra = run.get("infra_error")
    return {"id": task["id"], "prompt": task["prompt"][:300], "commit": task["commit"], "checks": checks,
            "status": "infra_error" if infra else "ok",
            "score": None if infra else (round(passed / len(checks), 4) if checks else 0.0), "diff_stat": diff[-2_000:],
            "finished": run["finished"], "steps": run["steps"], "seconds": run["seconds"],
            "transcript": run["transcript"],
            **({"handoffs": run["handoffs"], "coder_runs": run["coder_runs"]} if coder is not None else {})}


def run(snapshot: Path, model: str, base_url: str = net.SYNC_OLLAMA, limit: int | None = None,
        match: str | None = None, seed: int = 0, max_steps: int = 30, wall_seconds: int = 900,
        coder: str | None = None) -> Path:
    snapshot = Path(snapshot)
    tasks = [t for t in tasks_for(snapshot) if not match or match.lower() in t["prompt"].lower()]
    tasks = tasks[:limit] if limit else tasks
    trajs = {t["meta"]["file"]: t for t in bench_a.load_heldout(snapshot)}
    settings = {"sampling": backends.sampling(model), "max_tokens": backends.MAX_TOKENS, "seed": seed, "max_steps": max_steps, "wall_seconds": wall_seconds,
                "similar": SIMILAR, "history_budget_chars": harness.HISTORY_CHARS, "harness": harness.HARNESS_VERSION,
                "num_ctx": harness.NUM_CTX}
    name = backends.label(model) if coder is None else f"{backends.label(model)}+{backends.label(coder)}"  # operator+coder pairs get their own results folder
    settings |= {"coder": coder} if coder else {}
    out = RESULTS / re.sub(r"[^A-Za-z0-9._+-]", "_", name) / snapshot.name / "bench_b.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    done = {r["id"]: r for r in prev.get("tasks", []) if reusable(r)} if prev.get("settings") == settings else {}
    record = backends.record(model)
    if coder:
        record = {"operator": record, "coder": backends.record(coder)}
    chat = backends.chat(model, seed=seed)
    coder_chat = backends.chat(coder, seed=seed) if coder else None
    rows = []
    for task in tasks:
        rows.append(done.get(task["id"]) or run_task(task, trajs[task["file"]]["steps"], chat,
                                                      max_steps=max_steps, wall_seconds=wall_seconds, coder=coder_chat))
        _save(out, model, snapshot, settings, record, rows)
    _save(out, model, snapshot, settings, record, rows)
    return out


def _infra(r: dict) -> bool:
    """Rows from before the status field: a first transcript entry that is an error means the model never answered."""
    if "status" in r:
        return r["status"] == "infra_error"
    return r.get("steps") == 0 and bool(r.get("transcript")) and "error" in r["transcript"][0]


def reusable(r: dict) -> bool:
    return not _infra(r)


def summarise(rows: list[dict]) -> dict:
    """Infrastructure failures (model unreachable) are reported, never scored as zeros."""
    ok = [r for r in rows if not _infra(r)]
    checks = [c for r in ok for c in r["checks"]]
    return {"tasks": len(ok), "infra_errors": len(rows) - len(ok),
            "mean_score": round(sum(r["score"] for r in ok) / len(ok), 4) if ok else None,
            "ci95_mean_score": wilson(sum(r["score"] for r in ok), len(ok)) if ok else None,
            "checks_passed": sum(c["pass"] for c in checks), "checks": len(checks),
            "mean_seconds": round(sum(r["seconds"] for r in ok) / len(ok), 1) if ok else None}


def _save(out: Path, model: str, snapshot: Path, settings: dict, record: dict, rows: list[dict]) -> None:
    report = {
        "benchmark": "B", "model": model, "model_record": record, "snapshot": snapshot.name,
        "settings": settings, "overall": summarise(rows), "tasks": rows,
    }
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
