import json
import time

from forge import bench_a, extract, segment, snapshot, split


def test_extract_keeps_every_block(fake_projects):
    f = sorted(fake_projects.glob("-work-alpha/*.jsonl"))[0]
    t = extract.extract(f, fake_projects)
    kinds = [s["kind"] for s in t.steps]
    assert kinds.count("tool_call") == 4 and kinds.count("tool_result") == 4
    assert kinds[0] == "user_prompt" and "thinking" in kinds and "assistant_text" in kinds
    assert t.meta["git_root"].endswith("work/alpha") and not t.meta["automated"]
    assert t.steps[7]["output"] == "1 passed"


def test_automated_and_subagent_flags(fake_projects):
    bot = extract.extract(fake_projects / "-auto" / "bot.jsonl", fake_projects)
    assert bot.meta["automated"]
    sub = extract.extract(next(fake_projects.glob("*/*/subagents/*.jsonl")), fake_projects)
    assert sub.meta["subagent"] and sub.meta["parent_session"] in {"alpha-0", "beta-0"}


def test_classify():
    assert segment.classify("Edit", {}) == "coder"
    assert segment.classify("Bash", {"command": "cd x && npm run build"}) == "coder"
    assert segment.classify("Bash", {"command": "open -a Safari https://x.y"}) == "operator"
    assert segment.classify("Bash", {"command": "osascript -e 'tell app'"}) == "operator"
    assert segment.classify("mcp__peekaboo__see", {}) == "operator"
    assert segment.classify("Bash", {"command": "ls -la"}) == "neutral"
    assert segment.classify("Read", {}) == "neutral"


def test_segments_and_handoff(fake_projects):
    t = extract.extract(sorted(fake_projects.glob("-work-alpha/*.jsonl"))[0], fake_projects)
    segs = segment.segments(t.steps)
    assert [s["agent"] for s in segs] == ["coder", "operator"]
    assert segs[0]["start"] == 0 and segs[-1]["end"] == len(t.steps)
    ho = segment.handoffs(t.steps, segs)
    assert ho[0]["from"] == "coder" and ho[0]["incoming_first_call"]["name"] == "mcp__computer-use__screenshot"
    assert ho[0]["prompt"] == "fix the bug and show me"


def test_split_is_seeded():
    projects = ["/a", "/b", "/c", "/d"]
    assert split.draw(projects, 7) == split.draw(list(reversed(projects)), 7)


def test_snapshot_is_reproducible_and_verifies(fake_projects, tmp_path):
    future = time.time() + 60
    a = snapshot.build(fake_projects, tmp_path / "s1", seed=1, cutoff=future, workers=2)
    b = snapshot.build(fake_projects, tmp_path / "s2", seed=1, cutoff=future, workers=2)
    ma, mb = (json.loads((x / "manifest.json").read_text()) for x in (a, b))
    assert ma["outputs"] == mb["outputs"] and ma["raw"] == mb["raw"]
    assert ma["counts"]["automated_excluded"] == 1
    assert ma["heldout_project"].split("/")[-1] in {"alpha", "beta"}
    assert ma["counts"]["heldout_files"] == 6  # 5 sessions + 1 subagent
    assert snapshot.verify(a, rebuild=True) == []
    (a / "train" / "coder.jsonl").write_text("tampered\n")
    assert snapshot.verify(a) == ["output differs: train/coder.jsonl"]


def test_bench_a_items_render_and_score(fake_projects, tmp_path):
    s = snapshot.build(fake_projects, tmp_path / "s", seed=1, cutoff=time.time() + 60, raw=False, workers=2)
    items = bench_a.build_items(s, n=100)
    assert len(items) == 24 and items == bench_a.build_items(s, n=100)
    traj = {t["meta"]["file"]: t for t in bench_a.load_heldout(s)}[items[0]["file"]]
    msgs = bench_a.render(traj["steps"][: items[0]["step"]])
    assert msgs[0]["role"] == "user"
    gold = {"name": "Bash", "input": {"command": "python3 -m pytest -q", "description": "a"}}
    assert bench_a.score(gold, {"name": "Bash", "input": {"command": "python3  -m pytest -q"}})["exact"] == 1.0
    assert bench_a.score(gold, {"name": "Read", "input": {}})["name"] == 0.0
    part = bench_a.score(gold, {"name": "Bash", "input": {"command": "pytest"}})
    assert part["exact"] == 0.0 and 0 < part["args"] < 1


def test_touched_root_beats_starting_folder(tmp_path):
    repo = tmp_path / "real"
    (repo / ".git").mkdir(parents=True)
    steps = [
        {"kind": "tool_call", "name": "Read", "input": {"file_path": "/nowhere/x.py"}},
        {"kind": "tool_call", "name": "Edit", "input": {"file_path": str(repo / "a.py")}},
    ]
    assert snapshot.touched_root(steps) == str(repo)
    meta = {"touched_root": str(repo), "git_root": "/started/here", "cwd": "/started/here"}
    assert split.project_key(meta, {}) == str(repo)


def test_eligibility_needs_editing_sessions():
    base = {"project": "/p", "has_git": True, "agents": ["coder", "operator"]}
    assert split.eligible_projects([{**base, "edits": True}] * 2 + [{**base, "edits": False}] * 3) == []
    assert split.eligible_projects([{**base, "edits": True}] * 3 + [{**base, "edits": False}] * 2) == ["/p"]


def test_stats():
    from forge import stats
    assert stats.wilson(6, 12) == (0.2538, 0.7462)
    assert stats.sign_test([1, 1, 1, 1, 1, 1], [0] * 6)["p_value"] == 0.0312
    assert stats.sign_test([1, 0], [1, 0])["p_value"] == 1.0
