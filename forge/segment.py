"""Label tool calls coder / operator / neutral and cut a trajectory into per-agent segments."""
from __future__ import annotations

import re

CODER_TOOLS = {
    "Edit", "Write", "MultiEdit", "NotebookEdit",
    "mcp__plugin_mozcode_mozcode__code_edit", "mcp__mozcode__code_edit",
}
OPERATOR_PREFIXES = (
    "mcp__computer-use__", "mcp__peekaboo__", "mcp__claude-in-chrome__", "mcp__chrome-devtools__",
    "mcp__chrome-real__", "mcp__social-chrome__", "mcp__nutcracker__socialchrome_",
    "mcp__blenderwright__", "mcp__remote-devices__", "mcp__Claude_Browser__",
)
_CODER_BASH = re.compile(
    r"(?:^|[\s;&|(])(?:pytest|vitest|jest|tsc|cargo|make|xcodebuild|swift\s+(?:build|test)|"
    r"go\s+(?:build|test|run)|npm\s+(?:run|test|install|i|ci)|pnpm|yarn|bun\s+(?:run|test|install|add)|"
    r"uv\s+(?:run|add|sync|pip)|pip3?\s+install|python3?\s+-m\s+(?:pytest|unittest)|"
    r"git\s+(?:add|commit|diff|status|checkout|switch|merge|rebase|push|pull|stash|log|restore))\b"
)
_OPERATOR_BASH = re.compile(
    r"(?:^|[\s;&|(])(?:osascript|screencapture|cliclick|shortcuts\s+run|pbcopy|pbpaste|open)\s"
)

CODER, OPERATOR, NEUTRAL = "coder", "operator", "neutral"


def classify(name: str | None, tool_input: dict | None) -> str:
    name = name or ""
    if name in CODER_TOOLS:
        return CODER
    if name.startswith(OPERATOR_PREFIXES):
        return OPERATOR
    if name == "Bash":
        cmd = str((tool_input or {}).get("command", ""))
        if _OPERATOR_BASH.search(cmd):
            return OPERATOR
        if _CODER_BASH.search(cmd):
            return CODER
    return NEUTRAL


def label_steps(steps: list[dict]) -> list[str]:
    """One label per step. Tool results take their call's label; everything else is neutral."""
    by_id: dict[str, str] = {}
    labels = []
    for s in steps:
        if s["kind"] == "tool_call":
            lab = classify(s.get("name"), s.get("input"))
            if s.get("id"):
                by_id[s["id"]] = lab
        elif s["kind"] == "tool_result":
            lab = by_id.get(s.get("id"), NEUTRAL)
        else:
            lab = NEUTRAL
        labels.append(lab)
    return labels


def segments(steps: list[dict]) -> list[dict]:
    """Maximal runs of one agent. Neutral steps join the next labeled run (the reading before an edit
    belongs to the edit); trailing neutral steps join the previous run. A trajectory with no coder or
    operator steps at all is one operator segment, since the operator is the general tool caller."""
    labels = label_steps(steps)
    resolved: list[str] = [""] * len(labels)
    nxt = ""
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] != NEUTRAL:
            nxt = labels[i]
        resolved[i] = labels[i] if labels[i] != NEUTRAL else nxt
    prev = ""
    for i, lab in enumerate(resolved):
        if lab:
            prev = lab
        else:
            resolved[i] = prev or OPERATOR
    out: list[dict] = []
    for i, lab in enumerate(resolved):
        if out and out[-1]["agent"] == lab:
            out[-1]["end"] = i + 1
        else:
            out.append({"agent": lab, "start": i, "end": i + 1})
    return out


def handoffs(steps: list[dict], segs: list[dict]) -> list[dict]:
    """One example per agent switch: what was asked, what the outgoing agent said last, and the
    incoming agent's first action."""
    out = []
    for a, b in zip(segs, segs[1:]):
        prompt = next(
            (s["text"] for s in reversed(steps[: b["start"]]) if s["kind"] == "user_prompt" and not s.get("meta")),
            "",
        )
        said = [s["text"] for s in steps[a["start"]: a["end"]] if s["kind"] == "assistant_text"][-3:]
        first = next((s for s in steps[b["start"]: b["end"]] if s["kind"] == "tool_call"), None)
        out.append({
            "from": a["agent"], "to": b["agent"], "at_step": b["start"],
            "prompt": prompt, "outgoing_said": said,
            "incoming_first_call": {"name": first["name"], "input": first["input"]} if first else None,
        })
    return out
