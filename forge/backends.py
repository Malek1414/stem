"""Model backends behind one interface: Ollama (default) or a llama.cpp server (OpenAI-compatible API).

Model specs: a plain name is an Ollama model; "llamacpp:<remote port>" is a llama-server on the Spark, reached
through an NVIDIA Sync tunnel for that port (forge.net resolves sync://<port> on every attempt)."""
from __future__ import annotations

import json
import os
import urllib.request

from . import harness, net


def parse(spec: str) -> tuple[str, str]:
    if spec.startswith("llamacpp:"):
        return "llamacpp", f"sync://{spec.split(':', 1)[1].split('@')[0]}"
    return "ollama", net.SYNC_OLLAMA


def label(spec: str) -> str:
    """Name for results folders: 'llamacpp:8081@qwen36-ablit' -> 'qwen36-ablit'."""
    return spec.split("@", 1)[1] if spec.startswith("llamacpp:") and "@" in spec else spec


def to_openai(msgs: list[dict]) -> list[dict]:
    """Ollama-shaped history -> OpenAI shape: tool calls get ids and JSON-string arguments, tool results get the
    matching tool_call_id (results follow their calls in order)."""
    out, pending, n = [], [], 0
    for m in msgs:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                n += 1
                fn = c["function"]
                args = fn.get("arguments")
                calls.append({"id": c.get("id") or f"call_{n}", "type": "function",
                              "function": {"name": fn["name"],
                                           "arguments": args if isinstance(args, str) else json.dumps(args or {})}})
            m["tool_calls"] = calls
            m["content"] = m.get("content") or ""
            pending = [c["id"] for c in calls]
        elif m.get("role") == "tool":
            m["tool_call_id"] = pending.pop(0) if pending else f"call_orphan_{n}"
        out.append(m)
    return out


def from_openai(reply: dict) -> dict:
    msg = reply["choices"][0]["message"]
    calls = []
    for c in msg.get("tool_calls") or []:
        args = c["function"].get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else (args or {})
        except json.JSONDecodeError:
            args = {}
        calls.append({"id": c.get("id"), "function": {"name": c["function"]["name"], "arguments": args}})
    out = {"role": "assistant", "content": msg.get("content") or ""}
    if calls:
        out["tool_calls"] = calls
    return out


# Sampling per model card. The Qwen3.x cards (Holo4 is Qwen3.6-based) say not to decode greedily: it causes endless
# repetition, which we saw (the same command 16 times in a row at temperature 0). A fixed seed keeps runs reproducible.
SAMPLING = {
    "qwen3-coder": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05},  # Qwen3-Coder(-Next) card
    "qwen3": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},  # Qwen3.x cards
}
# One call once generated 29,400 tokens (thinking ran away) and filled the 64k context; cap every call.
MAX_TOKENS = 4096
GREEDY = {"temperature": 0}


def sampling(spec: str) -> dict:
    """AF_SAMPLING=greedy restores temperature 0 (harness v2 runs)."""
    if os.environ.get("AF_SAMPLING") == "greedy":
        return dict(GREEDY)
    s = spec.lower()
    return dict(SAMPLING["qwen3-coder"] if ("qwen3-coder" in s or "coder-next" in s) else SAMPLING["qwen3"])


def openai_chat(base_url: str, seed: int = 0, timeout: int = 900, params: dict | None = None) -> harness.ChatFn:
    params = dict(GREEDY if params is None else params)

    def chat(messages: list[dict], tools: list[dict]) -> dict:
        body = {"messages": to_openai(messages), **params, "seed": seed, "stream": False, "max_tokens": MAX_TOKENS}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with net.urlopen(req, timeout=timeout) as r:
            return from_openai(json.loads(r.read()))
    return chat


def chat(spec: str, seed: int = 0) -> harness.ChatFn:
    kind, base = parse(spec)
    params = sampling(spec)
    if kind == "llamacpp":
        return openai_chat(base, seed, params=params)
    return harness.ollama_chat(spec, base, seed=seed, params=params)


def vision(spec: str, seed: int = 0, timeout: int = 300):
    kind, base = parse(spec)
    if kind == "ollama":
        from .bench_d import ollama_vision
        return ollama_vision(spec, base, seed)

    def ask(prompt: str, image_b64: str) -> str:
        mt = "image/png" if image_b64.startswith("iVBOR") else "image/jpeg"
        body = {**sampling(spec), "seed": seed, "stream": False, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{image_b64}"}}]}]}
        req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with net.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"].get("content") or ""
    return ask


def record(spec: str) -> dict:
    """What exactly served the model: Ollama digest, or llama-server's model file and build."""
    kind, base = parse(spec)
    if kind == "ollama":
        from .bench_a import model_digest
        return model_digest(base, spec)
    with net.urlopen(f"{base}/props", timeout=120) as r:
        p = json.loads(r.read())
    return {"runtime": "llama.cpp", "model_path": p.get("model_path"), "build": p.get("build_info"),
            "n_ctx": (p.get("default_generation_settings") or {}).get("n_ctx")}
