"""Claude through the Claude Code CLI on Malek's plan ("claude-p:<model>"), for A3: trajectories on train-split tasks
through this harness (Malek, 16.11.2026: "ok to spend through claude -p so through the plan").

`claude -p` serves only as the model. Its own tools are off (--tools ""), it starts in an empty folder with no
settings, MCP servers or slash commands, and it returns the next step as JSON (--json-schema). The harness runs
every call in its sandbox and the replayed tools, exactly as for the local models, so nothing touches the real Mac.
Each step resends the whole conversation as text (the CLI caches the repeated prefix). Claude's thinking is not
returned; the visible text and calls are what the local model learns from."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

EFFORT = "high"
TIMEOUT = 900
SCHEMA = {"type": "object", "required": ["text", "tool_calls"], "properties": {
    "text": {"type": "string"},
    "tool_calls": {"type": "array", "items": {"type": "object", "required": ["name", "arguments"], "properties": {
        "name": {"type": "string"}, "arguments": {"type": "object"}}}}}}
FORMAT = ("\n\nYou act only through the tools below; the harness runs your calls and shows you the results in the "
          "next message. Each reply is JSON: \"text\" is what you say to the user (may be empty while you work), "
          "\"tool_calls\" the calls to run now, in order. A reply without calls ends your turn, so make your final "
          "message to the user complete.\n\nTools:\n")


def _args(a) -> dict:
    if isinstance(a, dict):
        return a
    try:
        v = json.loads(a or "{}")
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def render_tools(tools: list[dict]) -> str:
    if not tools:
        return "(none in this step: reply with text only and an empty tool_calls list)"
    return "\n".join(f"- {t['function']['name']}: {t['function'].get('description') or ''} "
                     f"Arguments: {json.dumps(t['function'].get('parameters') or {}, ensure_ascii=False)}" for t in tools)


def render(msgs: list[dict], tools: list[dict]) -> tuple[str, str]:
    """Harness history -> (system prompt, conversation as text)."""
    system, parts = [], []
    for m in msgs:
        role, content = m.get("role"), str(m.get("content") or "")
        if role == "system":
            system.append(content)
        elif role == "assistant":
            calls = "".join(f"\n[call] {c['function']['name']} {json.dumps(_args(c['function'].get('arguments')), ensure_ascii=False)}"
                            for c in m.get("tool_calls") or [])
            parts.append(f"=== assistant ===\n{content}{calls}")
        elif role == "tool":
            parts.append(f"=== tool result ===\n{content or '(no output)'}")
        else:
            parts.append(f"=== user ===\n{content}")
    convo = "\n\n".join(parts) + "\n\n=== your next reply (JSON) ==="
    return "\n\n".join(system) + FORMAT + render_tools(tools), convo


def parse(out: str, tools: list[dict], n0: int = 0) -> dict:
    d = json.loads(out)
    if d.get("is_error"):
        raise RuntimeError(f"claude -p: {str(d.get('result'))[:300]}")
    s = d.get("structured_output") or json.loads(d.get("result") or "{}")
    allowed = {t["function"]["name"] for t in tools}
    calls = [{"id": f"cp{n0 + i}", "function": {"name": c["name"], "arguments": _args(c.get("arguments"))}}
             for i, c in enumerate(s.get("tool_calls") or []) if allowed]  # no tools offered: calls are dropped
    reply = {"role": "assistant", "content": str(s.get("text") or "")}
    if calls:
        reply["tool_calls"] = calls
    return reply


def chat(model: str):
    empty = Path(tempfile.mkdtemp(prefix="claude-p-"))  # no CLAUDE.md, no project settings
    n = 0

    def call(messages: list[dict], tools: list[dict]) -> dict:
        nonlocal n
        system, convo = render(messages, tools)
        r = subprocess.run(["claude", "-p", "--model", model, "--effort", EFFORT, "--tools", "", "--no-session-persistence",
                            "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
                            "--output-format", "json", "--system-prompt", system, "--json-schema", json.dumps(SCHEMA)],
                           input=convo, capture_output=True, text=True, cwd=empty, timeout=TIMEOUT)
        if r.returncode != 0 and not r.stdout.strip():
            raise RuntimeError(f"claude -p exit {r.returncode}: {r.stderr[-300:]}")
        reply = parse(r.stdout, tools, n)
        n += len(reply.get("tool_calls", []))
        return reply
    return call


def record(model: str) -> dict:
    v = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    return {"runtime": "claude-cli", "cli": v, "model": model, "effort": EFFORT, "tools": "off (harness runs them)"}
