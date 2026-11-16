import json

from forge import backends, claude_p_backend as cp

TOOLS = [{"type": "function", "function": {"name": "Bash", "description": "run", "parameters": {"type": "object"}}}]


def test_render_puts_tools_in_the_system_prompt_and_the_conversation_in_order():
    msgs = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Bash", "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "content": "a.py"}]
    system, convo = cp.render(msgs, TOOLS)
    assert system.startswith("You are an agent.") and "- Bash: run" in system
    assert convo.index("list files") < convo.index('[call] Bash {"command": "ls"}') < convo.index("a.py")


def test_parse_reads_structured_output_and_drops_calls_when_no_tools_are_offered():
    out = json.dumps({"is_error": False, "structured_output": {"text": "hi", "tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}})
    r = cp.parse(out, TOOLS, 3)
    assert r["tool_calls"][0] == {"id": "cp3", "function": {"name": "Bash", "arguments": {"command": "ls"}}}
    assert "tool_calls" not in cp.parse(out, [])


def test_errors_are_infrastructure_errors():
    try:
        cp.parse(json.dumps({"is_error": True, "result": "usage limit reached"}), TOOLS)
    except RuntimeError as e:
        assert "usage limit" in str(e)
    else:
        raise AssertionError


def test_label_keeps_claude_p_runs_apart_from_api_runs():
    assert backends.label("claude-p:claude-opus-5-5") == "claude-p_claude-opus-5-5"
    assert backends.sampling("claude-p:claude-opus-5-5") == {"effort": "high"}
