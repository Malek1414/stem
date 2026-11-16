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
