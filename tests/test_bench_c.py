from forge import bench_c


def traj():
    steps = [
        {"kind": "user_prompt", "text": "find three competitors and their prices", "ts": "2026-11-01T10:00:00Z"},
        {"kind": "tool_call", "id": "1", "name": "WebSearch", "input": {"query": "ball tracking camera price"}},
        {"kind": "tool_result", "id": "1", "output": "..."},
        {"kind": "assistant_text", "text": "Pixellot $1k, Veo $1.2k, Trace $0.6k."},
        {"kind": "user_prompt", "text": "ok", "ts": "2026-11-01T10:05:00Z"},
    ]
    return {"meta": {"file": "s.jsonl", "subagent": False, "touched_root": None, "git_root": None}, "steps": steps}


def test_tasks_take_non_edit_turns_with_a_reply():
    tasks = bench_c.build_tasks([traj()])
    assert len(tasks) == 1  # "ok" is too short and has no reply
    assert tasks[0]["reference"]["final_reply"].startswith("Pixellot")
    assert tasks[0]["reference"]["actions"][0].startswith("WebSearch")


def test_run_task_and_judge(tmp_path):
    t = traj()
    task = bench_c.build_tasks([t])[0]
    replies = iter([{"role": "assistant", "content": "I would search the web for X. Pixellot, Veo, Trace.",
                     "tool_calls": [{"function": {"name": "done", "arguments": {}}}]}])
    chat = lambda msgs, tools: next(replies)
    seen = {}
    def ask(prompt):
        seen["p"] = prompt
        return {"understood": 5, "approach": 4, "correctness": 4, "completeness": 9, "overall": 4, "reason": "fine"}
    r = bench_c.run_task(task, t["steps"], chat, ask, root=tmp_path)
    assert r["grade"]["completeness"] == 5 and r["grade"]["overall"] == 4  # clamped to 1..5
    assert "overall" in r["null_grade"]
    assert "Pixellot" in seen["p"] and r["attempt"]["final"].startswith("I would search")


def test_bad_judge_output_is_recorded():
    task = bench_c.build_tasks([traj()])[0]
    g = bench_c.judge(lambda p: {"nope": 1}, task, {"actions": [], "final": ""})
    assert "error" in g


def test_unreachable_model_is_not_judged(tmp_path):
    t = traj()
    task = bench_c.build_tasks([t])[0]

    def down(msgs, tools):
        raise ConnectionRefusedError(61, "Connection refused")

    judged = []
    r = bench_c.run_task(task, t["steps"], down, lambda p: judged.append(p) or {}, root=tmp_path)
    assert r["status"] == "infra_error" and judged == []
