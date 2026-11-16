"""The Claude backend: harness history (Ollama/OpenAI shape) -> Messages API shape, and replies back."""
from types import SimpleNamespace as NS

from forge import anthropic_backend as ab

TOOLS = [{"type": "function", "function": {"name": "Bash", "description": "shell",
                                           "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]


def test_tools_convert():
    assert ab.to_tools(TOOLS) == [{"name": "Bash", "description": "shell",
                                   "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}]


def test_history_converts_with_system_tool_ids_and_orphans():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "assistant", "content": "old reply"},          # history may start with the assistant
            {"role": "tool", "content": "orphan output"},           # a result with no call before it
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Bash", "arguments": {"command": "ls"}}},
                                                                {"function": {"name": "Bash", "arguments": "{\"command\": \"pwd\"}"}}]},
            {"role": "tool", "content": "a.py"}, {"role": "tool", "content": "/x"}]
    system, out = ab.to_messages(msgs, {})
    assert system == "sys"
    assert out[0]["role"] == "user"
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "user", "assistant", "user"]  # same-role turns merge
    assert out[2]["content"][0]["text"].startswith("[tool output]") and out[2]["content"][1]["text"] == "do it"
    uses = [b for b in out[3]["content"] if b["type"] == "tool_use"]
    assert [u["input"] for u in uses] == [{"command": "ls"}, {"command": "pwd"}]
    res = out[4]["content"]
    assert [r["tool_use_id"] for r in res] == [u["id"] for u in uses] and res[1]["content"] == "/x"


def test_reply_keeps_raw_blocks_for_replay():
    raw = {}
    blocks = [NS(type="thinking", model_dump=lambda **k: {"type": "thinking", "thinking": "", "signature": "s"}),
              NS(type="text", text="Listing.", model_dump=lambda **k: {"type": "text", "text": "Listing."}),
              NS(type="tool_use", id="toolu_1", name="Bash", input={"command": "ls"},
                 model_dump=lambda **k: {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}})]
    reply = ab.from_response(NS(content=blocks, stop_reason="tool_use"), raw)
    assert reply["content"] == "Listing." and reply["tool_calls"][0]["id"] == "toolu_1"
    msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": reply["content"], "tool_calls": reply["tool_calls"]},
            {"role": "tool", "content": "a.py"}]
    _, out = ab.to_messages(msgs, raw)
    assert out[1]["content"][0]["type"] == "thinking"  # thinking block replayed unchanged
    assert out[2]["content"][0]["tool_use_id"] == "toolu_1"


def test_unanswered_call_gets_a_result_before_the_next_prompt():
    msgs = [{"role": "user", "content": "a"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Bash", "arguments": {}}}]},
            {"role": "user", "content": "b"}]
    _, out = ab.to_messages(msgs, {})
    assert out[2]["content"][0]["type"] == "tool_result" and out[2]["content"][1]["text"] == "b"


def test_partially_answered_calls_get_a_result_before_the_next_turn():
    # two calls, one result, then the next assistant turn: the second call still needs a tool_result (bench C 400)
    from forge.anthropic_backend import to_messages
    calls = [{"id": "a", "function": {"name": "Bash", "arguments": "{}"}},
             {"id": "b", "function": {"name": "Bash", "arguments": "{}"}}]
    _, out = to_messages([{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": calls},
                          {"role": "tool", "content": "ok"}, {"role": "assistant", "content": "done"}], {})
    results = [b for b in out[2]["content"] if b["type"] == "tool_result"]
    assert [r["tool_use_id"] for r in results] == ["toolu_h1", "toolu_h2"]
    assert results[1]["content"] == "(not run)" and out[3]["role"] == "assistant"
