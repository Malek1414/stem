from forge import distill


def test_example_marks_the_new_turns_and_pairs_results_with_calls():
    ep = {"system": "sys", "history": [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old reply"}],
          "prompt": "do it", "tools": [{"type": "function", "function": {"name": "Bash"}}],
          "transcript": [{"assistant": "", "calls": [{"name": "Bash", "arguments": {"command": "ls"}}], "results": ["a.py"]},
                         {"assistant": "Done: a.py.", "calls": [{"name": "done", "arguments": {}}], "done": ""}]}
    ex = distill.example(ep)
    m = ex["messages"]
    assert ex["loss_from"] == 4 and m[3] == {"role": "user", "content": "do it"}
    assert m[4]["tool_calls"][0]["function"]["name"] == "Bash" and m[5] == {"role": "tool", "tool_call_id": "call_1", "content": "a.py"}
    assert m[6]["content"] == "Done: a.py." and len(m) == 7  # no tool result for done


def test_subagent_segments_are_not_tasks(tmp_path):
    import json
    from forge import bench_c
    (tmp_path / "train").mkdir()
    steps = [{"kind": "user_prompt", "text": "please summarise the repo", "ts": "t"}, {"kind": "assistant_text", "text": "It is X."}]
    with open(tmp_path / "train" / "coder.jsonl", "w") as f:
        for name in ("s/subagents/agent-1.jsonl", "s.jsonl"):
            f.write(json.dumps({"file": name, "segment": 0, "steps": steps}) + "\n")
    tasks = bench_c.build_tasks(distill.train_trajectories(tmp_path))
    assert [t["file"] for t in tasks] == ["s.jsonl@0"]


def test_attempts_that_read_the_session_logs_are_not_kept():
    row = lambda cmd: {"episode": {"transcript": [{"assistant": "", "calls": [{"name": "Bash", "arguments": {"command": cmd}}]}]}}
    assert distill.reads_logs(row("grep -rl 2f80 ~/.claude/projects"))
    assert not distill.reads_logs(row("ls ~/Desktop"))


def test_build_merges_all_worker_files_and_drops_duplicates(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(distill, "OUT", tmp_path / "distill")
    d = tmp_path / "distill" / "v9" / "claude-p_m"
    d.mkdir(parents=True)
    ep = {"system": "s", "history": [], "prompt": "p", "tools": [], "transcript": [{"assistant": "ok", "calls": []}]}
    row = lambda i, o: json.dumps({"id": i, "grade": {"overall": o}, "episode": ep}) + "\n"
    (d / "runs.jsonl").write_text(row("a", 5) + row("b", 2))
    (d / "runs-w2.jsonl").write_text(row("a", 5) + row("c", 4))
    monkeypatch.chdir(tmp_path)
    import forge.distill as fd
    monkeypatch.setattr(fd.Path, "resolve", lambda self: tmp_path / "forge" / "distill.py" if self.name == "distill.py" else self)
    stats = distill.build(tmp_path / "v9", "m", 4)
    assert stats["runs"] == 3 and stats["kept"] == 2


def test_claudes_summarised_thinking_becomes_reasoning_content():
    ep = {"system": "s", "history": [], "prompt": "go", "tools": [],
          "transcript": [{"assistant": "Checking.", "reasoning": "See where the rename stands.", "calls": [{"name": "Bash", "arguments": {}}], "results": ["ok"]},
                         {"assistant": "Done.", "calls": []}]}
    m = distill.example(ep)["messages"]
    assert m[2]["reasoning_content"] == "See where the rename stands." and "reasoning_content" not in m[4]


def test_usd_matches_opus_prices_and_spent_sums_rows(tmp_path):
    assert distill.usd({"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
                        "cache_creation_input_tokens": 1_000_000}) == 4 + 20 + 0.2 + 5
    f = tmp_path / "runs-x.jsonl"
    f.write_text('{"id": "a", "usd": 1.5}\n{"id": "b"}\n{"id": "c", "usd": 2.0}\n')
    assert distill.spent(f) == 3.5 and distill.spent(tmp_path / "none.jsonl") == 0


def test_api_worker_stops_before_the_budget(tmp_path, monkeypatch):
    import json
    import pytest
    from forge import anthropic_backend, backends, bench_c
    snap = tmp_path / "v0002"
    snap.mkdir()
    (snap / "tools.json").write_text("[]")
    monkeypatch.setattr(distill, "OUT", tmp_path / "out")
    monkeypatch.setattr(distill, "train_trajectories", lambda s: [{"meta": {"file": "f"}, "steps": []}])
    tasks = [{"id": f"t{i}", "file": "f"} for i in range(10)]
    monkeypatch.setattr(bench_c, "build_tasks", lambda *a, **k: tasks)
    monkeypatch.setattr(backends, "chat", lambda spec: None)
    monkeypatch.setitem(distill.VIA, "api", ("anthropic:{m}", lambda m: None))
    monkeypatch.setattr(anthropic_backend, "USAGE", {"calls": 0, "output_tokens": 0})

    def fake_task(task, *a, **k):  # each attempt costs $1 (50k output tokens)
        anthropic_backend.USAGE["output_tokens"] += 50_000
        return {"id": task["id"], "status": "ok", "grade": {"overall": 4}, "steps": 1, "seconds": 1}
    monkeypatch.setattr(bench_c, "run_task", fake_task)
    with pytest.raises(SystemExit):
        distill.run(snap, "m", 10, 0, "v2", tag="b", via="api")  # no budget given
    runs = distill.run(snap, "m", 10, 0, "v2", tag="b", via="api", budget_usd=6.0)
    rows = [json.loads(l) for l in open(runs)]
    assert len(rows) == 4 and distill.spent(runs) == 4.0  # 3 spent + 2.5 reserve fits 6, then 4 + 2.5 does not
    anthropic_backend.USAGE["output_tokens"] = 0  # a restart keeps counting what the file already holds
    distill.run(snap, "m", 10, 0, "v2", tag="b", via="api", budget_usd=6.0)
    assert len(open(runs).readlines()) == 4
