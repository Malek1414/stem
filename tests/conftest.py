import json
from pathlib import Path

import pytest


def rec(typ, content, cwd, ts, sid, **extra):
    return {"type": typ, "message": {"role": typ, "content": content}, "cwd": cwd, "timestamp": ts,
            "sessionId": sid, "gitBranch": "main", "entrypoint": extra.pop("entrypoint", "cli"), **extra}


def session(cwd, sid, entrypoint="cli"):
    """A session that reads, edits a file, runs tests, then takes a screenshot."""
    return [
        rec("user", "fix the bug and show me", cwd, "2026-11-01T10:00:00Z", sid, entrypoint=entrypoint),
        rec("assistant", [{"type": "thinking", "thinking": "plan"}], cwd, "2026-11-01T10:00:01Z", sid, entrypoint=entrypoint),
        rec("assistant", [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": f"{cwd}/a.py"}}], cwd, "2026-11-01T10:00:02Z", sid, entrypoint=entrypoint),
        rec("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "x = 1"}], cwd, "2026-11-01T10:00:03Z", sid, entrypoint=entrypoint),
        rec("assistant", [{"type": "tool_use", "id": "t2", "name": "Edit", "input": {"file_path": f"{cwd}/a.py", "old_string": "1", "new_string": "2"}}], cwd, "2026-11-01T10:00:04Z", sid, entrypoint=entrypoint),
        rec("user", [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}], cwd, "2026-11-01T10:00:05Z", sid, entrypoint=entrypoint),
        rec("assistant", [{"type": "tool_use", "id": "t3", "name": "Bash", "input": {"command": "python3 -m pytest -q", "description": "run tests"}}], cwd, "2026-11-01T10:00:06Z", sid, entrypoint=entrypoint),
        rec("user", [{"type": "tool_result", "tool_use_id": "t3", "content": [{"type": "text", "text": "1 passed"}]}], cwd, "2026-11-01T10:00:07Z", sid, entrypoint=entrypoint),
        rec("assistant", [{"type": "text", "text": "Tests pass, taking a screenshot."}], cwd, "2026-11-01T10:00:08Z", sid, entrypoint=entrypoint),
        rec("assistant", [{"type": "tool_use", "id": "t4", "name": "mcp__computer-use__screenshot", "input": {}}], cwd, "2026-11-01T10:00:09Z", sid, entrypoint=entrypoint),
        rec("user", [{"type": "tool_result", "tool_use_id": "t4", "content": [{"type": "image", "source": {}}]}], cwd, "2026-11-01T10:00:10Z", sid, entrypoint=entrypoint),
        {"type": "ai-title", "aiTitle": "ignored"},
    ]


@pytest.fixture
def fake_projects(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    for proj in ("alpha", "beta"):
        repo = tmp_path / "work" / proj
        (repo / ".git").mkdir(parents=True)
        d = root / f"-work-{proj}"
        d.mkdir(parents=True)
        for n in range(5):
            sid = f"{proj}-{n}"
            (d / f"{sid}.jsonl").write_text("\n".join(json.dumps(r) for r in session(str(repo), sid)) + "\n")
        sub = d / f"{proj}-0" / "subagents"
        sub.mkdir(parents=True)
        (sub / "agent-x.jsonl").write_text("\n".join(json.dumps(r) for r in session(str(repo), f"{proj}-0")) + "\n")
    auto = root / "-auto"
    auto.mkdir()
    (auto / "bot.jsonl").write_text("\n".join(json.dumps(r) for r in session(str(tmp_path), "bot", "sdk-py")) + "\n")
    return root
