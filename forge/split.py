"""Projects, eligibility for the held-out draw, and the seeded random draw itself."""
from __future__ import annotations

import random
from collections import defaultdict

DEFAULT_SEED = 20261115
MIN_SESSIONS = 5
MIN_EDIT_SESSIONS = 3
RULES = "v2: project = repo of most-touched files; >=5 sessions, >=3 editing code, coder+operator steps"


def project_key(meta: dict, parents: dict[str, dict]) -> str:
    """The git repo whose files the session touched most (edits count triple); else the repo it started
    in; else the working folder. Subagents belong to their parent's project."""
    if meta.get("subagent") and meta.get("parent_session") in parents:
        meta = parents[meta["parent_session"]]
    return meta.get("touched_root") or meta.get("git_root") or meta.get("cwd") or "?"


def eligible_projects(infos: list[dict]) -> list[str]:
    """infos: one per non-automated main session: {project, has_git, edits, agents: labels seen}."""
    count = defaultdict(int)
    agents = defaultdict(set)
    git = defaultdict(bool)
    edits = defaultdict(int)
    for i in infos:
        edits[i["project"]] += bool(i.get("edits"))
        p = i["project"]
        count[p] += 1
        agents[p] |= set(i["agents"])
        git[p] = git[p] or i["has_git"]
    return sorted(
        p for p in count
        if git[p] and count[p] >= MIN_SESSIONS and edits[p] >= MIN_EDIT_SESSIONS
        and {"coder", "operator"} <= agents[p]
    )


def draw(eligible: list[str], seed: int = DEFAULT_SEED) -> str:
    if not eligible:
        raise ValueError("no eligible project to hold out")
    return random.Random(seed).choice(sorted(eligible))
