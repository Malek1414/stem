import os
import subprocess
from pathlib import Path

import pytest

from forge import bench_b, harness


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "proj"
    r.mkdir()
    (r / "a.py").write_text("x = 1\nprint(x)\n")
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-10-01T10:00:00", "GIT_COMMITTER_DATE": "2026-10-01T10:00:00"}
    for cmd in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", *cmd], cwd=r, env=env, check=True)
    return r


def traj(repo):
    steps = [
        {"kind": "user_prompt", "text": "set x to 2", "ts": "2026-11-01T10:00:00Z"},
        {"kind": "tool_call", "id": "1", "name": "Edit",
         "input": {"file_path": f"{repo}/a.py", "old_string": "x = 1", "new_string": "x = 2"}},
        {"kind": "tool_result", "id": "1", "output": "ok"},
        {"kind": "user_prompt", "text": "thanks", "ts": "2026-11-01T10:05:00Z"},
    ]
    return {"meta": {"file": "s.jsonl", "subagent": False, "touched_root": str(repo), "git_root": None}, "steps": steps}


def scripted(*replies):
    it = iter(replies)
    return lambda msgs, tools: next(it)


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_tasks_and_expected_state(repo):
    tasks = bench_b.build_tasks([traj(repo)])
    assert len(tasks) == 1
    f = tasks[0]["files"]["a.py"]
    assert f["base"] == "x = 1\nprint(x)\n" and f["expected"] == "x = 2\nprint(x)\n"


def test_good_agent_passes_and_noop_fails(repo, tmp_path):
    t = traj(repo)
    task = bench_b.build_tasks([t])[0]
    good = scripted(call("Edit", file_path="a.py", old_string="x = 1", new_string="x = 2"), call("done"))
    r = bench_b.run_task(task, t["steps"], good, root=tmp_path / "w")
    assert r["score"] == 1.0 and r["finished"]
    noop = scripted({"role": "assistant", "content": "nothing to do"})
    assert bench_b.run_task(task, t["steps"], noop, root=tmp_path / "w")["score"] == 0.0
    assert (repo / "a.py").read_text() == "x = 1\nprint(x)\n"  # the real repo is never touched


def test_sandbox_blocks_escape(tmp_path):
    ws = harness.Workspace(tmp_path)
    target = Path.home() / "forge_escape_test.txt"
    out = ws.call("bash", {"command": f"echo x > {target}"})
    assert not target.exists() and "exit 0" not in out
    assert "outside" in ws.call("write_file", {"path": str(target), "content": "x"})
    assert "exit 0" in ws.call("bash", {"command": "echo hi > inside.txt"}) and (tmp_path / "inside.txt").exists()


def test_unreachable_model_is_infra_error_not_a_zero(repo, tmp_path):
    t = traj(repo)
    task = bench_b.build_tasks([t])[0]

    def down(msgs, tools):
        raise ConnectionRefusedError(61, "Connection refused")

    r = bench_b.run_task(task, t["steps"], down, root=tmp_path / "w")
    assert r["status"] == "infra_error" and r["score"] is None
    rows = [r, {**r, "status": "ok", "score": 0.5, "id": "other"}]
    s = bench_b.summarise(rows)
    assert s["tasks"] == 1 and s["infra_errors"] == 1 and s["mean_score"] == 0.5
    assert bench_b.reusable(r) is False and bench_b.reusable(rows[1]) is True
