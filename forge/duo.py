"""Operator + coder on one task, the way the link connects them, with one shared budget.

The operator explores (Read, Bash) and hands work to the coder with a typed task card (link.cards.TaskCard):
goal, done criteria, constraints, context, plus a free-text field. The coder sees only the card, never the
session history, and edits with Read/Write/Edit/Bash. Steps and wall time are shared, so operator + coder get
exactly the single agent's budget (equal-budget comparison: splitting agents must win on quality, not compute).
Transport is in-process here; the link's measured overhead (17-63 ms median per message) is negligible next to
model calls and is reported separately."""
from __future__ import annotations

from pathlib import Path

from link import cards

from . import harness

_by_name = {t["function"]["name"]: t for t in harness.TOOLS}
HANDOFF = {"type": "function", "function": {
    "name": "handoff",
    "description": ("Give a coding task to the coder agent. It sees ONLY this card, not the conversation, so put "
                    "every fact it needs in context. Returns the coder's report."),
    "parameters": {"type": "object", "required": ["goal", "done_criteria"], "properties": {
        "goal": {"type": "string"},
        "done_criteria": {"type": "object", "required": ["description"], "properties": {
            "description": {"type": "string"}, "check_command": {"type": "string"}}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "context": {"type": "object", "properties": {
            "files": {"type": "array", "items": {"type": "string"}},
            "findings": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"}}},
        "freeform": {"type": "string"}}}}}
OPERATOR_TOOLS = [_by_name["Bash"], _by_name["Read"], HANDOFF, _by_name["done"]]
CODER_TOOLS = [_by_name[n] for n in ("Bash", "Read", "Write", "Edit", "done")]


def card_prompt(card: cards.TaskCard) -> str:
    ctx = card.context or {}
    parts = [f"Task from the operator: {card.goal}", f"Done when: {card.done_criteria.get('description', '')}"]
    if card.done_criteria.get("check_command"):
        parts.append(f"Check with: {card.done_criteria['check_command']}")
    parts += [f"Constraint: {c}" for c in card.constraints or []]
    if ctx.get("files"):
        parts.append("Relevant files: " + ", ".join(ctx["files"]))
    parts += [f"Finding: {f}" for f in ctx.get("findings") or []]
    if ctx.get("notes"):
        parts.append(f"Notes: {ctx['notes']}")
    if card.freeform:
        parts.append(card.freeform)
    return "\n".join(parts)


def run(operator: harness.ChatFn, coder: harness.ChatFn, system: str, history: list[dict], prompt: str,
        workdir: Path, max_steps: int = 30, wall_seconds: int = 900) -> dict:
    budget = harness.Budget(max_steps, wall_seconds)
    coder_runs: list[dict] = []
    coder_system = (f"You are a coding agent. The repository is checked out at {workdir}. Work only inside that "
                    "folder. There is no network. Make the change the task card asks for, then call done with a "
                    "short report of what you changed.")

    def handoff(args: dict) -> str:
        try:
            card = cards.TaskCard.from_dict(args)
            card.validate()
        except (ValueError, TypeError, KeyError) as e:
            return f"error: invalid card ({e}). Required: goal (text), done_criteria.description (text)."
        r = harness.run_agent(coder, coder_system, [], card_prompt(card), workdir, tools=CODER_TOOLS, budget=budget)
        coder_runs.append(r | {"card": args})
        if r["infra_error"]:
            raise RuntimeError(r["infra_error"])
        report = next((t["done"] for t in reversed(r["transcript"]) if t.get("done")), None)
        last = next((t["assistant"] for t in reversed(r["transcript"]) if t.get("assistant")), "")
        return (f"coder {'finished' if r['finished'] else 'stopped without finishing'} after {r['steps']} steps. "
                f"Report: {report or last or '(none)'}")

    op_system = system + (" You are the operator: you can read and run commands but you cannot edit files. "
                          "Hand every code change to the coder with the handoff tool, then check the result.")
    try:
        r = harness.run_agent(operator, op_system, history, prompt, workdir, tools=OPERATOR_TOOLS,
                              handlers={"handoff": handoff}, budget=budget)
    except RuntimeError as e:  # the coder's model was unreachable
        r = {"finished": False, "steps": 0, "seconds": 0, "transcript": [{"error": str(e), "infra": True}],
             "infra_error": str(e)}
    infra = r["infra_error"] or next((c["infra_error"] for c in coder_runs if c["infra_error"]), None)
    return {"finished": r["finished"], "steps": r["steps"] + sum(c["steps"] for c in coder_runs),
            "seconds": round(time_used(budget), 1), "transcript": r["transcript"], "infra_error": infra,
            "handoffs": len(coder_runs), "coder_runs": coder_runs}


def time_used(budget: harness.Budget) -> float:
    import time
    return time.time() - budget.t0
