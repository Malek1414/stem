import io
import json

from forge import backends, net


def test_messages_get_ids_and_string_arguments():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Bash", "arguments": {"command": "ls"}}},
                                                               {"function": {"name": "Read", "arguments": {"file_path": "a"}}}]},
            {"role": "tool", "content": "a.py"}, {"role": "tool", "content": "x = 1"}]
    out = backends.to_openai(msgs)
    calls = out[1]["tool_calls"]
    assert [c["type"] for c in calls] == ["function", "function"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"}
    assert [m["tool_call_id"] for m in out[2:]] == [calls[0]["id"], calls[1]["id"]]
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}  # input untouched


def test_reply_is_normalised_to_the_harness_shape():
    reply = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "Edit", "arguments": "{\"file_path\": \"a\"}"}}]}}]}
    m = backends.from_openai(reply)
    assert m["content"] == "" and m["tool_calls"][0]["function"] == {"name": "Edit", "arguments": {"file_path": "a"}}


def test_sync_port_resolution(monkeypatch):
    monkeypatch.setattr(net, "sync_port", lambda remote_port=11434: {11434: 50001, 8081: 50002}[remote_port])
    assert net._resolve("sync://ollama/api/tags") == "http://localhost:50001/api/tags"
    assert net._resolve("sync://8081/v1/chat/completions") == "http://localhost:50002/v1/chat/completions"


def test_model_spec_parsing():
    assert backends.parse("llamacpp:8081") == ("llamacpp", "sync://8081")
    assert backends.parse("qwen3-coder-next-abliterated-128k") == ("ollama", net.SYNC_OLLAMA)


def test_vision_message_uses_image_url():
    seen = {}

    def fake(req, timeout):
        seen["body"] = json.loads(req.data)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": '{"x": 1, "y": 2}'}}]}).encode())

    import forge.net as n
    n_urlopen = n.urlopen
    try:
        n.urlopen = fake
        ask = backends.vision("llamacpp:8081")
        assert ask("where?", "QUJD") == '{"x": 1, "y": 2}'
    finally:
        n.urlopen = n_urlopen
    part = seen["body"]["messages"][0]["content"][1]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/")
