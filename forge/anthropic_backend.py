"""Claude as a reference agent, through the same harness and judge as the local models ("anthropic:<model id>").

It measures the ceiling of the benchmarks (what the model the sessions came from can reach under this harness);
it is never a candidate for the operator or coder roles. Credentials come from the Anthropic SDK's usual sources
(ANTHROPIC_API_KEY or the `ant auth login` profile).

- The harness keeps only role/content/tool_calls of a reply. Claude's thinking blocks must go back unchanged, so the
  raw content of every reply is kept here, keyed by its first tool_use id, and replayed in place of the summary.
- Prompt caching: top-level automatic cache_control, so each step re-reads the growing prefix from cache.
- Thinking is always on for Claude Opus 5.5; effort is set explicitly (its default is medium)."""
from __future__ import annotations

import json

EFFORT = "high"
MAX_TOKENS = 16_000  # thinking + a tool call; the local models' 4096 cap would cut Claude off mid-thought
THINKING = {"type": "adaptive", "display": "summarized"}  # summaries come back as text (default: omitted); for A3
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


def to_tools(tools: list[dict]) -> list[dict]:
    return [{"name": t["function"]["name"], "description": t["function"].get("description") or t["function"]["name"],
             "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}} for t in tools]


def _args(a) -> dict:
    if isinstance(a, dict):
        return a
    try:
        v = json.loads(a or "{}")
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def to_messages(msgs: list[dict], raw: dict) -> tuple[str, list[dict]]:
    """Harness history -> (system, messages). Tool results of one turn become one user message of tool_result
    blocks, in call order; a result whose call is not right before it becomes plain text."""
    system, out, pending, n = [], [], [], 0

    def add(role: str, blocks: list[dict]):
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    def flush():  # every call needs a result before anything else follows it, also when only some results came
        nonlocal pending
        if pending and out:
            add("user", [{"type": "tool_result", "tool_use_id": p, "content": "(not run)"} for p in pending])
        pending = []

    for m in msgs:
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            system.append(str(content))
        elif role == "assistant":
            flush()
            calls = m.get("tool_calls") or []
            first = calls[0].get("id") if calls else None
            if first and first in raw:
                blocks = raw[first]
                pending = [b["id"] for b in blocks if b["type"] == "tool_use"]
            else:
                blocks = [{"type": "text", "text": str(content)}] if str(content).strip() else []
                pending = []
                for c in calls:
                    n += 1
                    cid = f"toolu_h{n}"
                    pending.append(cid)
                    blocks.append({"type": "tool_use", "id": cid, "name": c["function"]["name"],
                                   "input": _args(c["function"].get("arguments"))})
                blocks = blocks or [{"type": "text", "text": "(no reply)"}]
            if not out:
                out.append({"role": "user", "content": [{"type": "text", "text": "(conversation so far)"}]})
            add("assistant", blocks)
        elif role == "tool":
            if pending:
                add("user", [{"type": "tool_result", "tool_use_id": pending.pop(0), "content": str(content) or "(no output)"}])
            else:
                add("user", [{"type": "text", "text": "[tool output]\n" + str(content)}])
        else:
            flush()
            add("user", [{"type": "text", "text": str(content) or "(empty)"}])
    flush()
    return "\n\n".join(system), out


def from_response(resp, raw: dict) -> dict:
    blocks = [b.model_dump(exclude_none=True, by_alias=True) for b in resp.content]  # by alias: fallback.from (SDK from_)
    text = "\n".join(b.text for b in resp.content if b.type == "text")
    thinking = "\n".join(getattr(b, "thinking", "") or "" for b in resp.content if b.type == "thinking").strip()
    calls = [{"id": b.id, "function": {"name": b.name, "arguments": dict(b.input or {})}}
             for b in resp.content if b.type == "tool_use"]
    out = {"role": "assistant", "content": text}
    if thinking:
        out["reasoning"] = thinking
    if calls:
        out["tool_calls"] = calls
        raw[calls[0]["id"]] = blocks
    return out


def chat(model: str):
    import anthropic
    client = anthropic.Anthropic(max_retries=5, timeout=600)
    raw: dict = {}

    def call(messages: list[dict], tools: list[dict]) -> dict:
        system, msgs = to_messages(messages, raw)
        kw = {"tools": to_tools(tools)} if tools else {}
        resp = client.beta.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=system or anthropic.NOT_GIVEN, messages=msgs,
            output_config={"effort": EFFORT}, thinking=THINKING, cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        USAGE["calls"] += 1
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            USAGE[k] += getattr(resp.usage, k, 0) or 0
        return from_response(resp, raw)
    return call


def record(model: str) -> dict:
    return {"runtime": "anthropic-api", "model": model, "effort": EFFORT, "max_tokens": MAX_TOKENS, "thinking": THINKING,
            "fallbacks": "default"}
