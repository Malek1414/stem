"""A minimal, sandboxed agent loop: a model gets bash / read / write / edit tools inside one folder.

The model under test may be abliterated (no refusals), so safety lives here, not in the model:
every shell command runs under macOS sandbox-exec with no network and no writes outside the work
folder, and the file tools refuse any path that resolves outside it."""
from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.request

from . import net
from pathlib import Path
from typing import Callable

ChatFn = Callable[[list[dict], list[dict]], dict]


class ContextFull(Exception):
    """The request no longer fits the model's context window. The agent has used up its context: the run ends
    there like a run out of steps or wall time, and is graded, not counted as an infrastructure error."""

# Harness v2 (2026-11-15): fixed context window for every model (HF GGUFs otherwise get Ollama's small default),
# longer rendered history, and an optional forced final reply when the step budget runs out.
HARNESS_VERSION = 2
NUM_CTX = 65_536
HISTORY_CHARS = 96_000
FINAL_PROMPT = ("Your step budget is used up. Reply to the user now with your final answer: what you did, what you "
                "found, and what you would do next. No more tool calls.")

# Claude Code's own tool names and argument names: the session history the model sees uses them, and
# so does the training data. Lower-case aliases are accepted too.
TOOLS = [
    {"type": "function", "function": {
        "name": "Bash", "description": "Run a shell command in the work folder (no network). Returns exit code and output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "Read", "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {
        "name": "Write", "description": "Create or overwrite a text file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["file_path", "content"]}}},
    {"type": "function", "function": {
        "name": "Edit", "description": "Replace one exact occurrence of old_string with new_string in a file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"},
                                                        "new_string": {"type": "string"}},
                       "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "done", "description": "Call when the task is finished.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}}}},
]
ALIASES = {"bash": "Bash", "read": "Read", "read_file": "Read", "write": "Write", "write_file": "Write",
           "edit": "Edit", "edit_file": "Edit"}


def sandbox_profile(workdir: Path) -> str:
    w = str(Path(workdir).resolve())
    # no network, no writes outside the work folder, and no signals to processes outside the sandbox (a replayed
    # `pkill -f remotion` must not kill the user's real apps)
    return ("(version 1)(allow default)(deny network*)(deny signal)(allow signal (target same-sandbox))"
            f'(deny file-write* (require-not (require-any (subpath "{w}") (subpath "/private/var/folders") '
            '(subpath "/dev"))))')


def _clip(text: str, limit: int = 8_000) -> str:
    return text if len(text) <= limit else text[: limit // 2] + f"\n[... {len(text) - limit} chars cut ...]\n" + text[-limit // 2:]


class Workspace:
    def __init__(self, root: Path, bash_timeout: int = 120):
        self.root = Path(root).resolve()
        self.bash_timeout = bash_timeout

    def _path(self, p: str) -> Path:
        q = (self.root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
        if q != self.root and self.root not in q.parents:
            raise PermissionError(f"{p} is outside the work folder {self.root}")
        return q

    def call(self, name: str, args: dict) -> str:
        name = ALIASES.get(name, ALIASES.get(name.lower(), name))
        path = args.get("file_path") or args.get("path")
        try:
            if name == "Bash":
                r = subprocess.run(["sandbox-exec", "-p", sandbox_profile(self.root), "bash", "-c", str(args.get("command", ""))],
                                   cwd=self.root, capture_output=True, text=True, timeout=self.bash_timeout)
                return _clip(f"exit {r.returncode}\n{r.stdout}{r.stderr}")
            if name == "Read":
                return _clip(self._path(path).read_text(errors="replace"), 20_000)
            if name == "Write":
                p = self._path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(args.get("content", "")))
                return f"wrote {p.relative_to(self.root)}"
            if name == "Edit":
                p = self._path(path)
                text = p.read_text(errors="replace")
                old = str(args.get("old_string", ""))
                if not old or text.count(old) != 1:
                    return f"error: old_string found {text.count(old) if old else 0} times, need exactly 1"
                p.write_text(text.replace(old, str(args.get("new_string", "")), 1))
                return f"edited {p.relative_to(self.root)}"
            return (f"error: {name} is not available in this run. Available tools: Bash, Read, Write, Edit, "
                    "done. Use Bash for anything else (no network).")
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {self.bash_timeout}s"
        except Exception as e:
            return f"error: {e}"


class Budget:
    """Steps and wall time shared by every agent working on one task (equal-budget comparisons)."""

    def __init__(self, max_steps: int, wall_seconds: float):
        self.steps_left, self.deadline, self.t0 = max_steps, time.time() + wall_seconds, time.time()

    def ok(self) -> bool:
        return self.steps_left > 0 and time.time() < self.deadline


def run_agent(chat: ChatFn, system: str, history: list[dict], prompt: str, workdir: Path,
              max_steps: int = 30, wall_seconds: int = 900, tools: list[dict] | None = None,
              handlers: dict[str, Callable[[dict], str]] | None = None, budget: Budget | None = None,
              force_final: bool = False, self_check: str | None = None) -> dict:
    """tools: the tool list this agent may use (default: all of TOOLS). handlers: extra tools implemented by the
    caller (e.g. handoff). budget: shared with other agents on the same task; default is a private one.
    self_check: the first time the agent says it is finished (done, or a reply without calls) while budget is left,
    it gets this message instead and goes on; the second time ends the run. The agent checks its own work: it sees
    nothing of a grader."""
    tools = TOOLS if tools is None else tools
    allowed = {t["function"]["name"] for t in tools}
    handlers = handlers or {}
    budget = budget or Budget(max_steps, wall_seconds)
    ws = Workspace(workdir)
    msgs = [{"role": "system", "content": system}, *history, {"role": "user", "content": prompt}]
    transcript, t0, steps, finished = [], time.time(), 0, False
    infra_error = None
    while budget.ok():
        try:
            reply = chat(msgs, tools)
        except ContextFull as e:
            transcript.append({"error": f"context full: {e}", "infra": False})
            break
        except Exception as e:  # the model never answered: an infrastructure failure, not a model failure
            transcript.append({"error": repr(e), "infra": True})
            infra_error = repr(e)
            break
        msgs.append({k: v for k, v in reply.items() if k in ("role", "content", "tool_calls")} | {"role": "assistant"})
        calls = reply.get("tool_calls") or []
        transcript.append({"assistant": reply.get("content", ""), "calls": [c["function"] for c in calls]})
        if not calls:
            if self_check and budget.ok():
                msgs.append({"role": "user", "content": self_check})
                transcript[-1]["self_check"] = True
                self_check = None
                continue
            break
        for c in calls:
            fn = c["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            steps += 1
            budget.steps_left -= 1
            name = ALIASES.get(fn["name"], fn["name"])
            if name == "done":
                if self_check and budget.ok():
                    msgs.append({"role": "tool", "content": self_check})
                    transcript[-1]["self_check"] = True
                    self_check = None
                    break
                finished = True
                transcript[-1]["done"] = str(args.get("summary", ""))
                break
            if name not in allowed:
                out = (f"error: {fn['name']} is not available to you. Your tools: "
                       + ", ".join(sorted(allowed)))
            elif name in handlers:
                out = handlers[name](args)
            else:
                out = ws.call(name, args)
            transcript[-1].setdefault("results", []).append(out[:2_000])
            msgs.append({"role": "tool", "content": out})
            if not budget.ok():
                break
        if finished:
            break
    forced = False
    has_text = any(t.get("assistant") for t in transcript[-1:])
    if force_final and infra_error is None and not has_text:
        try:  # one tool-free turn so a model that ran out of steps still answers the user
            ask = msgs + [{"role": "user", "content": FINAL_PROMPT}]
            try:
                reply = chat(ask, [])
            except ContextFull:  # keep the task prompt and the last turns, drop the middle
                p = len(history) + 1
                ask = msgs[: 1] + msgs[p: p + 1] + msgs[-10:] + [{"role": "user", "content": FINAL_PROMPT}]
                reply = chat(ask, [])
            text = reply.get("content", "") or ""
            if "<tool_call>" in text:  # some models write a tool call as text when tools are withdrawn: ask once more
                reply = chat(ask + [{"role": "assistant", "content": text},
                                    {"role": "user", "content": "That was a tool call. Tools are off. Answer in plain text."}], [])
                text = strip_tool_markup(reply.get("content", "") or "")
            transcript.append({"assistant": text, "calls": [], "forced_final": True})
            forced = True
        except Exception as e:
            transcript.append({"error": repr(e), "infra": True})
            infra_error = repr(e)
    return {"finished": finished, "steps": steps, "seconds": round(time.time() - t0, 1), "transcript": transcript,
            "infra_error": infra_error, "forced_final": forced}


def strip_tool_markup(text: str) -> str:
    return re.sub(r"<tool_call>.*?(</tool_call>|$)", "", text, flags=re.S).strip()


def ollama_chat(model: str, base_url: str = net.SYNC_OLLAMA, seed: int = 0,
                num_ctx: int | None = NUM_CTX, timeout: int = 900, params: dict | None = None) -> ChatFn:
    opts = dict(params or {"temperature": 0})

    def chat(messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": model, "messages": messages, "tools": tools, "stream": False,
                "options": opts | {"seed": seed, "num_predict": 4096} | ({"num_ctx": num_ctx} if num_ctx else {})}
        req = urllib.request.Request(f"{base_url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with net.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"]
    return chat
