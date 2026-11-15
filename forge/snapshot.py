"""Build a versioned, hashed, byte-reproducible dataset snapshot from session files."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import extract as ex
from . import segment as sg
from . import split as sp

REPO = Path(__file__).resolve().parent.parent
SNAPSHOTS = REPO / "data" / "snapshots"
HELDOUT_FILE = REPO / "data" / "heldout.json"


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _json_type(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "null"


def _work(args) -> dict:
    """Worker: extract one file, write its pieces to tmp files, return its meta and a small summary."""
    idx, path, src_root, tmp, git_override = args
    traj = ex.extract(Path(path), Path(src_root))
    if git_override is not None:
        traj.meta["git_root"] = git_override.get(traj.meta["file"])
    segs = sg.segments(traj.steps)
    hos = sg.handoffs(traj.steps, segs)
    sid = traj.meta["session_id"]
    tools: dict[str, dict[str, str]] = {}
    for s in traj.steps:
        if s["kind"] == "tool_call" and isinstance(s.get("input"), dict):
            keys = tools.setdefault(s["name"] or "?", {})
            for k, v in s["input"].items():
                keys.setdefault(k, _json_type(v))
    base = Path(tmp) / f"{idx:06d}"
    with open(f"{base}.traj.jsonl", "w", encoding="utf-8") as fh:
        fh.write(dumps(traj.to_dict()) + "\n")
    with open(f"{base}.seg.jsonl", "w", encoding="utf-8") as fh:
        for n, seg in enumerate(segs):
            fh.write(dumps({
                "session_id": sid, "file": traj.meta["file"], "segment": n, "agent": seg["agent"],
                "claude_output": True, "steps": traj.steps[seg["start"]: seg["end"]],
            }) + "\n")
    with open(f"{base}.ho.jsonl", "w", encoding="utf-8") as fh:
        for h in hos:
            fh.write(dumps({"session_id": sid, "file": traj.meta["file"], **h}) + "\n")
    return {
        "idx": idx, "meta": traj.meta, "tools": tools,
        "agents": sorted({seg["agent"] for seg in segs if seg["agent"] != sg.NEUTRAL}),
        "n_steps": len(traj.steps),
        "n_calls": sum(1 for s in traj.steps if s["kind"] == "tool_call"),
    }


def _sha(path: Path) -> str:
    return ex.sha256_file(path)


def _lines(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def _code_commit() -> dict:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "forge"], capture_output=True, text=True).stdout.strip())
    except OSError:
        head, dirty = "", True
    return {"commit": head or None, "dirty": dirty}


def _write_raw(out: Path, src_root: Path, files: list[dict]) -> None:
    """Deterministic tar.gz of the exact source bytes, checked against the recorded hashes."""
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for f in files:
            data = (src_root / f["file"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != f["sha256"]:
                raise RuntimeError(f"{f['file']} changed while snapshotting; rerun with an earlier cutoff")
            info = tarfile.TarInfo(f["file"])
            info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))


def next_version() -> str:
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name[1:]) for p in SNAPSHOTS.glob("v[0-9][0-9][0-9][0-9]") if p.name[1:].isdigit()]
    return f"v{(max(nums) + 1 if nums else 1):04d}"


def build(
    src_root: Path = ex.PROJECTS_DIR,
    out_dir: Path | None = None,
    seed: int = sp.DEFAULT_SEED,
    cutoff: float | None = None,
    heldout: str | None = None,
    git_override: dict | None = None,
    raw: bool = True,
    workers: int | None = None,
    limit: int | None = None,
) -> Path:
    src_root = Path(src_root)
    cutoff = cutoff if cutoff is not None else time.time() - 15 * 60
    files = [p for p in ex.find_session_files(src_root) if p.stat().st_mtime <= cutoff]
    if limit:
        files = files[:limit]
    out = Path(out_dir) if out_dir else SNAPSHOTS / next_version()
    tmp = out / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    jobs = [(i, str(p), str(src_root), str(tmp), git_override) for i, p in enumerate(files)]
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 4) - 2)) as pool:
        results = sorted(pool.map(_work, jobs, chunksize=4), key=lambda r: r["idx"])

    # automated sessions, and subagents of automated sessions, are out of everything
    mains = {r["meta"]["session_id"]: r["meta"] for r in results if not r["meta"]["subagent"]}
    def automated(m: dict) -> bool:
        if m["subagent"] and m["parent_session"] in mains:
            return mains[m["parent_session"]]["automated"]
        return m["automated"]
    kept = [r for r in results if not automated(r["meta"])]
    for r in kept:
        r["project"] = sp.project_key(r["meta"], mains)

    infos = [
        {"project": r["project"], "has_git": bool(r["meta"]["git_root"]), "agents": r["agents"]}
        for r in kept if not r["meta"]["subagent"]
    ]
    eligible = sp.eligible_projects(infos)
    if heldout is None:
        if HELDOUT_FILE.exists() and out_dir is None:
            heldout = json.loads(HELDOUT_FILE.read_text())["project"]
        else:
            heldout = sp.draw(eligible, seed)
            if out_dir is None:
                HELDOUT_FILE.write_text(dumps({"project": heldout, "seed": seed, "drawn_from": eligible}) + "\n")

    (out / "train").mkdir(exist_ok=True)
    (out / "heldout").mkdir(exist_ok=True)
    outs = {
        "coder": open(out / "train" / "coder.jsonl", "w", encoding="utf-8"),
        "operator": open(out / "train" / "operator.jsonl", "w", encoding="utf-8"),
        "handoffs": open(out / "train" / "handoffs.jsonl", "w", encoding="utf-8"),
        "heldout": open(out / "heldout" / "trajectories.jsonl", "w", encoding="utf-8"),
    }
    try:
        for r in kept:
            base = tmp / f"{r['idx']:06d}"
            if r["project"] == heldout:
                outs["heldout"].write(Path(f"{base}.traj.jsonl").read_text(encoding="utf-8"))
                continue
            with open(f"{base}.seg.jsonl", encoding="utf-8") as fh:
                for line in fh:
                    agent = json.loads(line)["agent"]
                    outs["operator" if agent == sg.NEUTRAL else agent].write(line)
            outs["handoffs"].write(Path(f"{base}.ho.jsonl").read_text(encoding="utf-8"))
    finally:
        for fh in outs.values():
            fh.close()

    tools: dict[str, dict[str, str]] = {}
    for r in kept:
        for name, keys in r["tools"].items():
            t = tools.setdefault(name, {})
            for k, v in keys.items():
                t.setdefault(k, v)
    registry = {
        name: {"type": "object", "properties": {k: {"type": v} for k, v in sorted(keys.items())}}
        for name, keys in sorted(tools.items())
    }
    (out / "tools.json").write_text(dumps(registry) + "\n", encoding="utf-8")
    shutil.rmtree(tmp)

    sources = [
        {"file": r["meta"]["file"], "sha256": r["meta"]["sha256"], "bytes": r["meta"]["bytes"],
         "git_root": r["meta"]["git_root"], "project": r["project"]}
        for r in kept
    ]
    if raw:
        _write_raw(out / "raw.tar.gz", src_root, sources)

    output_files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in {"manifest.json", "raw.tar.gz"})
    manifest = {
        "version": out.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cutoff": cutoff,
        "seed": seed,
        "heldout_project": heldout,
        "eligible_projects": eligible,
        "code": _code_commit(),
        "training_targets": "fallback: claude_output steps are context only until Anthropic grants permission",
        "counts": {
            "files_seen": len(results),
            "automated_excluded": len(results) - len(kept),
            "kept": len(kept),
            "heldout_files": sum(1 for r in kept if r["project"] == heldout),
            "tool_calls": sum(r["n_calls"] for r in kept),
        },
        "outputs": [
            {"path": str(p.relative_to(out)), "sha256": _sha(p), "lines": _lines(p)} for p in output_files
        ],
        "raw": {"path": "raw.tar.gz", "sha256": _sha(out / "raw.tar.gz")} if raw else None,
        "sources": sources,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return out


def verify(snapshot: Path, rebuild: bool = False) -> list[str]:
    """Re-hash every output (and the raw archive). With rebuild, also rebuild from raw.tar.gz and compare."""
    snapshot = Path(snapshot)
    m = json.loads((snapshot / "manifest.json").read_text())
    problems = []
    for o in m["outputs"]:
        p = snapshot / o["path"]
        if not p.exists() or _sha(p) != o["sha256"]:
            problems.append(f"output differs: {o['path']}")
    if m.get("raw"):
        if _sha(snapshot / m["raw"]["path"]) != m["raw"]["sha256"]:
            problems.append("raw archive differs")
        elif rebuild:
            work = snapshot.parent / f".rebuild-{snapshot.name}"
            shutil.rmtree(work, ignore_errors=True)
            src = work / "src"
            src.mkdir(parents=True)
            with tarfile.open(snapshot / m["raw"]["path"], "r:gz") as tar:
                tar.extractall(src, filter="data")
            git = {s["file"]: s["git_root"] for s in m["sources"]}
            again = build(src, work / "out", seed=m["seed"], cutoff=time.time() + 10,
                          heldout=m["heldout_project"], git_override=git, raw=False)
            mine = {o["path"]: o["sha256"] for o in m["outputs"]}
            theirs = {str(p.relative_to(again)): _sha(p)
                      for p in again.rglob("*") if p.is_file() and p.name != "manifest.json"}
            for rel in sorted(set(mine) | set(theirs)):
                if mine.get(rel) != theirs.get(rel):
                    problems.append(f"rebuild differs: {rel}")
            shutil.rmtree(work, ignore_errors=True)
    return problems
