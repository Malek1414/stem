import json

from forge import sft


def seg(n_turns=3, out="ok"):
    steps = []
    for i in range(n_turns):
        steps += [{"kind": "user_prompt", "text": f"task {i}"},
                  {"kind": "thinking", "text": ""},
                  {"kind": "assistant_text", "text": f"doing {i}"},
                  {"kind": "tool_call", "id": f"t{i}", "name": "Bash", "input": {"command": f"echo {i}"}},
                  {"kind": "tool_result", "id": f"t{i}", "output": out},
                  {"kind": "assistant_text", "text": f"done {i}"}]
    return {"agent": "coder", "file": "f", "segment": 0, "steps": steps}


def test_messages_pair_calls_with_results():
    ex = sft.examples(seg(), {"Bash": {"type": "object", "properties": {"command": {"type": "string"}}}})
    assert len(ex) == 1
    m = ex[0]["messages"]
    assert m[0]["role"] == "system" and m[1] == {"role": "user", "content": "task 0"}
    call = m[2]["tool_calls"][0]
    assert m[2]["content"] == "doing 0" and json.loads(call["function"]["arguments"]) == {"command": "echo 0"}
    assert m[3] == {"role": "tool", "tool_call_id": call["id"], "content": "ok"}
    assert m[4] == {"role": "assistant", "content": "done 0"}
    assert [t["function"]["name"] for t in ex[0]["tools"]][:6] == sft.CORE


def test_windows_keep_the_prompt_and_never_split_a_call():
    ex = sft.examples(seg(6, out="x" * 900), {}, max_chars=1500)
    assert len(ex) > 1
    for e in ex:
        msgs = e["messages"][1:]
        assert msgs[0]["role"] == "user"
        ids = [c["id"] for m in msgs for c in m.get("tool_calls", [])]
        assert ids == [m["tool_call_id"] for m in msgs if m["role"] == "tool"]


def test_secrets_are_redacted():
    text = "key sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345 and ghp_" + "a" * 36 + " password=hunter2secret"
    r = sft.redact(text)
    assert "abcdefghij" not in r and "aaaa" not in r and "hunter2" not in r and r.count("[REDACTED") == 3


def test_forbidden_spark_ssh_windows_are_dropped():
    s = seg(1)
    s["steps"][3]["input"] = {"command": "ssh spark 'nvidia-smi'"}
    assert sft.examples(s, {}) == [None]
    s["steps"][3]["input"] = {"command": "echo spark is fine"}
    assert sft.examples(s, {})[0] is not None
