"""A minimal, sandboxed agent loop: a model gets bash / read / write / edit tools inside one folder.

The model under test may be abliterated (no refusals), so safety lives here, not in the model:
every shell command runs under macOS sandbox-exec with no network and no writes outside the work
folder, and the file tools refuse any path that resolves outside it."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request

from . import net
from pathlib import Path
from typing import Callable

ChatFn = Callable[[list[dict], list[dict]], dict]

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
    return ("(version 1)(allow default)(deny network*)"
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


def run_agent(chat: ChatFn, system: str, history: list[dict], prompt: str, workdir: Path,
              max_steps: int = 30, wall_seconds: int = 900) -> dict:
    ws = Workspace(workdir)
    msgs = [{"role": "system", "content": system}, *history, {"role": "user", "content": prompt}]
    transcript, t0, steps, finished = [], time.time(), 0, False
    infra_error = None
    while steps < max_steps and time.time() - t0 < wall_seconds:
        try:
            reply = chat(msgs, TOOLS)
        except Exception as e:  # the model never answered: an infrastructure failure, not a model failure
            transcript.append({"error": repr(e), "infra": True})
            infra_error = repr(e)
            break
        msgs.append({k: v for k, v in reply.items() if k in ("role", "content", "tool_calls")} | {"role": "assistant"})
        calls = reply.get("tool_calls") or []
        transcript.append({"assistant": reply.get("content", ""), "calls": [c["function"] for c in calls]})
        if not calls:
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
            if fn["name"] == "done":
                finished = True
                break
            out = ws.call(fn["name"], args)
            transcript[-1].setdefault("results", []).append(out[:2_000])
            msgs.append({"role": "tool", "content": out})
        if finished:
            break
    return {"finished": finished, "steps": steps, "seconds": round(time.time() - t0, 1), "transcript": transcript,
            "infra_error": infra_error}


def ollama_chat(model: str, base_url: str = net.SYNC_OLLAMA, seed: int = 0,
                num_ctx: int | None = None, timeout: int = 900) -> ChatFn:
    def chat(messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": model, "messages": messages, "tools": tools, "stream": False,
                "options": {"temperature": 0, "seed": seed} | ({"num_ctx": num_ctx} if num_ctx else {})}
        req = urllib.request.Request(f"{base_url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with net.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"]
    return chat
