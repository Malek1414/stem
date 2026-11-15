"""Task cards and progress events, and how they travel inside standard A2A messages.

A card is a typed handoff: goal, constraints, the context the receiver needs, and what counts as done,
plus a free-text field for anything that doesn't fit the schema. Typed handoffs cut tokens 2-3x but
need that escape field (Routed Graph Handoff, arXiv 2608.25277; docs/research/2026-11-15-agent-communication.md).
On the wire a card is an ordinary A2A Message with one data part, so any A2A agent can receive it."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field

from a2a import types as t
from google.protobuf import json_format, struct_pb2

CARD_TYPE = "application/vnd.agent-forge.task-card+json"
PROGRESS_TYPE = "application/vnd.agent-forge.progress+json"
PROGRESS_KINDS = ("accepted", "rejected", "started", "progress", "blocked", "done", "failed")
MAX_PROGRESS_TEXT = 280  # progress stays short; detail goes in data or the workspace

CARD_SCHEMA = {  # also passed to Ollama as `format`, so a local model must emit exactly this
    "type": "object",
    "required": ["goal", "done_criteria"],
    "properties": {
        "goal": {"type": "string"},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "context": {"type": "object", "properties": {
            "files": {"type": "array", "items": {"type": "string"}},
            "findings": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"}}},
        "done_criteria": {"type": "object", "required": ["description"], "properties": {
            "description": {"type": "string"},
            "check_command": {"type": "string"}}},
        "freeform": {"type": "string"},
    },
}


@dataclass
class TaskCard:
    goal: str
    done_criteria: dict
    constraints: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    freeform: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def validate(self) -> None:
        if not self.goal.strip():
            raise ValueError("card needs a goal")
        if not isinstance(self.done_criteria, dict) or not str(self.done_criteria.get("description", "")).strip():
            raise ValueError("card needs done_criteria.description")
        if not all(isinstance(c, str) for c in self.constraints):
            raise ValueError("constraints must be strings")

    @classmethod
    def from_dict(cls, d: dict) -> "TaskCard":
        card = cls(goal=str(d.get("goal", "")), done_criteria=d.get("done_criteria") or {},
                   constraints=list(d.get("constraints") or []), context=dict(d.get("context") or {}),
                   freeform=str(d.get("freeform") or ""), id=str(d.get("id") or uuid.uuid4().hex))
        card.validate()
        return card


def _data_part(payload: dict, media_type: str) -> t.Part:
    s = struct_pb2.Value()
    json_format.ParseDict(payload, s)
    return t.Part(data=s, media_type=media_type)


def text_message(text: str, context_id: str = "") -> t.Message:
    return t.Message(message_id=uuid.uuid4().hex, role=t.Role.ROLE_USER, parts=[t.Part(text=text)],
                     context_id=context_id)


def card_message(card: TaskCard) -> t.Message:
    card.validate()
    return t.Message(message_id=uuid.uuid4().hex, role=t.Role.ROLE_USER, task_id=card.id,
                     parts=[_data_part(asdict(card), CARD_TYPE)])


def progress_message(card_id: str, kind: str, text: str = "", data: dict | None = None) -> t.Message:
    if kind not in PROGRESS_KINDS:
        raise ValueError(f"progress kind must be one of {PROGRESS_KINDS}")
    payload = {"card_id": card_id, "kind": kind, "text": text[:MAX_PROGRESS_TEXT], "data": data or {}}
    return t.Message(message_id=uuid.uuid4().hex, role=t.Role.ROLE_AGENT, task_id=card_id,
                     parts=[_data_part(payload, PROGRESS_TYPE)])


def decode(msg: t.Message) -> dict:
    """An A2A message -> {"type": "card"|"progress"|"text", ...}. Unknown data parts become text."""
    for p in msg.parts:
        if p.WhichOneof("content") == "data":
            payload = json_format.MessageToDict(p.data)
            if p.media_type == CARD_TYPE:
                return {"type": "card", "card": asdict(TaskCard.from_dict(payload))}
            if p.media_type == PROGRESS_TYPE:
                if payload.get("kind") not in PROGRESS_KINDS:
                    raise ValueError("bad progress kind")
                return {"type": "progress", **payload}
    text = "\n".join(p.text for p in msg.parts if p.WhichOneof("content") == "text")
    return {"type": "text", "text": text, "context_id": msg.context_id}
