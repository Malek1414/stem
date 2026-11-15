"""Paired, task-level comparison of two models on one benchmark of one snapshot."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .stats import sign_test

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _scores(model: str, snapshot: str, bench: str) -> dict[str, float]:
    r = json.loads((RESULTS / re.sub(r"[^A-Za-z0-9._-]", "_", model) / snapshot / f"bench_{bench}.json").read_text())
    if bench == "a":
        return {i["id"]: i["score"]["exact"] for i in r["items"]}
    if bench == "b":
        return {t["id"]: t["score"] for t in r["tasks"]}
    return {t["id"]: t["grade"]["overall"] for t in r["tasks"] if "overall" in t["grade"]}


def compare(snapshot: str, bench: str, model_a: str, model_b: str) -> dict:
    a, b = _scores(model_a, snapshot, bench), _scores(model_b, snapshot, bench)
    shared = sorted(set(a) & set(b))
    test = sign_test([a[k] for k in shared], [b[k] for k in shared])
    return {"bench": bench, "snapshot": snapshot, "a": model_a, "b": model_b, "paired_tasks": len(shared),
            "mean_a": round(sum(a[k] for k in shared) / len(shared), 4) if shared else None,
            "mean_b": round(sum(b[k] for k in shared) / len(shared), 4) if shared else None,
            "a_better_on": test["wins"], "b_better_on": test["losses"], "ties": test["ties"],
            "p_value": test["p_value"]}
