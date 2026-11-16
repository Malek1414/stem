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


def test_budget_exhaustion_forces_a_final_reply(tmp_path):
    from forge import harness
    seen_tools = []

    def chat(msgs, tools):
        seen_tools.append(len(tools))
        if not tools:  # the forced final turn has no tools
            return {"role": "assistant", "content": "Final: here is what I found."}
        return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Bash", "arguments": {"command": "true"}}}]}

    r = harness.run_agent(chat, "sys", [], "do it", tmp_path, max_steps=3, force_final=True)
    assert r["transcript"][-1]["assistant"] == "Final: here is what I found." and seen_tools[-1] == 0
    assert r["forced_final"] is True


def test_missing_tools_are_named_in_the_system_prompt():
    steps = [{"kind": "tool_call", "name": "mcp__chrome-real__click", "input": {}},
             {"kind": "tool_call", "name": "Bash", "input": {}}, {"kind": "tool_call", "name": "WebFetch", "input": {}}]
    note = bench_c.missing_tools_note(steps)
    assert "mcp__chrome-real__click" in note and "WebFetch" in note and "Bash" not in note


def test_forced_final_retries_a_textual_tool_call(tmp_path):
    from forge import harness
    replies = iter([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Bash", "arguments": {"command": "ls"}}}]},
                    {"role": "assistant", "content": "<tool_call>\n<function=Bash>\n<parameter=command>ls</parameter>"},
                    {"role": "assistant", "content": "I listed the folder; it was empty."}])
    run = harness.run_agent(lambda m, t: next(replies), "sys", [], "do it", tmp_path, max_steps=1, force_final=True)
    assert run["transcript"][-1]["assistant"] == "I listed the folder; it was empty."
    assert harness.strip_tool_markup("ok <tool_call><function=X></tool_call> done") == "ok  done"


def test_replay_tools_answer_from_the_recording():
    from forge import bench_c
    turn = [{"kind": "user_prompt", "text": "group my tabs"},
            {"kind": "tool_call", "id": "a", "name": "mcp__chrome__list_pages", "input": {}},
            {"kind": "tool_result", "id": "a", "output": "1: rossmann cart\n2: zalando cart"},
            {"kind": "tool_call", "id": "b", "name": "mcp__chrome__evaluate_script", "input": {"function": "group tabs 1 2"}},
            {"kind": "tool_result", "id": "b", "output": "grouped 2 tabs"},
            {"kind": "tool_call", "id": "c", "name": "mcp__chrome__evaluate_script", "input": {"function": "read title"}},
            {"kind": "tool_result", "id": "c", "output": "Rossmann"},
            {"kind": "tool_call", "id": "d", "name": "Bash", "input": {"command": "ls"}},
            {"kind": "tool_result", "id": "d", "output": "x"}]
    assert bench_c.external_tools(turn) == ["mcp__chrome__evaluate_script", "mcp__chrome__list_pages"]
    tools, h = bench_c.replay_tools(turn, {"mcp__chrome__list_pages": {"type": "object", "properties": {}}})
    assert {t["function"]["name"] for t in tools} == {"mcp__chrome__evaluate_script", "mcp__chrome__list_pages"}
    assert h["mcp__chrome__list_pages"]({}) == "1: rossmann cart\n2: zalando cart"
    assert h["mcp__chrome__evaluate_script"]({"function": "group the tabs 1 and 2"}) == "grouped 2 tabs"  # most similar
    assert h["mcp__chrome__evaluate_script"]({"function": "group again"}) == "Rossmann"  # unused ones first


def test_offline_subset():
    from forge import bench_c
    rows = [{"grade": {"overall": 4}, "external_tools": []}, {"grade": {"overall": 2}, "external_tools": []}]
    assert bench_c._subset(rows) == {"tasks": 2, "overall": 3.0, "ci95_overall": bench_c.mean_ci([4, 2])}


def test_orphan_tool_result_becomes_user_text():
    from forge.backends import to_openai
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "why did you stop"},
            {"role": "tool", "content": "late output"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Bash", "arguments": {"command": "ls"}}}]},
            {"role": "tool", "content": "a.txt"}]
    out = to_openai(msgs)
    assert out[2] == {"role": "user", "content": "[tool output]\nlate output"}
    assert out[4]["role"] == "tool" and out[4]["tool_call_id"] == out[3]["tool_calls"][0]["id"]


def test_context_full_ends_the_run_without_infra_error(tmp_path):
    from forge import harness

    def chat(messages, tools):
        raise harness.ContextFull("request (67771 tokens) exceeds the available context size (65536 tokens)")
    run = harness.run_agent(chat, "sys", [], "do it", tmp_path)
    assert run.get("infra_error") is None
    assert "context full" in run["transcript"][-1]["error"]


def test_context_full_still_gets_a_forced_final_answer(tmp_path):
    from forge import harness
    calls = []

    def chat(messages, tools):
        calls.append(len(messages))
        if len(calls) == 1 or len(calls) == 2:
            raise harness.ContextFull("exceeds the available context size")
        return {"role": "assistant", "content": "here is what I did"}
    history = [{"role": "user", "content": f"h{i}"} for i in range(30)]
    run = harness.run_agent(chat, "sys", history, "do it", tmp_path, force_final=True)
    assert run["infra_error"] is None and run["forced_final"]
    assert run["transcript"][-1]["assistant"] == "here is what I did"
    assert calls[2] < calls[1]
