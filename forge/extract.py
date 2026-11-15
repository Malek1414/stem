"""One Claude Code session file in, one normalized trajectory out. Nothing is dropped or truncated."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

PROJECTS_DIR = Path.home() / ".claude" / "projects"
AUTOMATED_ENTRYPOINTS = {"sdk-py", "sdk-cli", "sdk-ts"}
AUTOMATED_CWD_MARKERS = (".claude-mem/observer-sessions", ".entropy/autopilot")


@dataclass
class Trajectory:
    meta: dict
    steps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"meta": self.meta, "steps": self.steps}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_session_files(root: Path = PROJECTS_DIR) -> list[Path]:
    """Main session files and subagent files, sorted for determinism."""
    main = root.glob("*/*.jsonl")
    sub = root.glob("*/*/subagents/*.jsonl")
    return sorted({*main, *sub})


def git_root(cwd: str | None) -> str | None:
    if not cwd:
        return None
    p = Path(cwd)
    for candidate in (p, *p.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return None


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                parts.append(str(b))
            elif b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif b.get("type") == "image":
                parts.append("[image]")
            else:
                parts.append(json.dumps(b, sort_keys=True, ensure_ascii=False))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _user_steps(rec: dict) -> list[dict]:
    content = (rec.get("message") or {}).get("content")
    ts = rec.get("timestamp")
    meta = bool(rec.get("isMeta"))
    if isinstance(content, str):
        return [{"kind": "user_prompt", "text": content, "ts": ts, "meta": meta}]
    steps = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "tool_result":
            steps.append({
                "kind": "tool_result",
                "id": b.get("tool_use_id"),
                "output": _result_text(b.get("content")),
                "is_error": bool(b.get("is_error")),
                "ts": ts,
            })
        elif t == "text":
            steps.append({"kind": "user_prompt", "text": b.get("text", ""), "ts": ts, "meta": meta})
        elif t == "image":
            steps.append({"kind": "user_prompt", "text": "[image]", "ts": ts, "meta": meta})
    return steps


def _assistant_steps(rec: dict) -> list[dict]:
    msg = rec.get("message") or {}
    ts = rec.get("timestamp")
    model = msg.get("model")
    steps = []
    for b in msg.get("content") or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            steps.append({"kind": "assistant_text", "text": b.get("text", ""), "ts": ts, "model": model})
        elif t == "thinking":
            steps.append({"kind": "thinking", "text": b.get("thinking", ""), "ts": ts, "model": model})
        elif t == "tool_use":
            steps.append({
                "kind": "tool_call",
                "id": b.get("id"),
                "name": b.get("name"),
                "input": b.get("input") or {},
                "ts": ts,
                "model": model,
            })
    return steps


def extract(path: Path, root: Path = PROJECTS_DIR) -> Trajectory:
    path = Path(path)
    steps: list[dict] = []
    cwd = branch = entrypoint = session_id = None
    first_ts = last_ts = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            cwd = cwd or rec.get("cwd")
            branch = branch or rec.get("gitBranch")
            entrypoint = entrypoint or rec.get("entrypoint")
            session_id = session_id or rec.get("sessionId")
            typ = rec.get("type")
            if typ == "user":
                new = _user_steps(rec)
            elif typ == "assistant":
                new = _assistant_steps(rec)
            else:
                continue
            if new:
                ts = rec.get("timestamp")
                first_ts = first_ts or ts
                last_ts = ts or last_ts
                steps.extend(new)

    is_sub = path.parent.name == "subagents"
    parent = path.parent.parent.name if is_sub else None
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    automated = (entrypoint in AUTOMATED_ENTRYPOINTS) or any(
        m in (cwd or "") for m in AUTOMATED_CWD_MARKERS
    )
    meta = {
        "session_id": path.stem if is_sub else (session_id or path.stem),
        "file": rel,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "cwd": cwd,
        "git_root": git_root(cwd),
        "branch": branch,
        "entrypoint": entrypoint,
        "automated": automated,
        "subagent": is_sub,
        "parent_session": parent,
        "start": first_ts,
        "end": last_ts,
        "claude_output": True,
    }
    return Trajectory(meta=meta, steps=steps)
