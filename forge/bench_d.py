"""Benchmark D: click grounding on the user's own screens.

Items come from real GUI actions in the snapshot's raw sessions (computer-use left_click, clicks inside
computer_batch, Chrome-extension left_click). Each item is the last screenshot before the click, the agent's own
words just before it (what it meant to click), and the click point. The model sees the screenshot and the intent
and must answer with a point; it passes if the point is within max(30 px, 3% of the image width) of the original.

Caveat: the original click was made by a frontier model and may itself have been off, so this measures agreement
with frontier clicks on these screens, not ground truth. Items are frozen per snapshot for reproducibility."""
from __future__ import annotations

import base64
import io
import json
import math
import re
import struct
import tarfile
import time
from pathlib import Path

from . import net
from .stats import wilson

RESULTS = Path("results")
CLICK_TOOLS = {"mcp__computer-use__left_click", "mcp__computer-use__computer_batch", "mcp__claude-in-chrome__computer"}


def image_size(data: bytes) -> tuple[int, int]:
    """(width, height) of a PNG or JPEG without extra dependencies."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    raise ValueError("unknown image format")


def _clicks(name: str, inp: dict) -> list[list[float]]:
    if name == "mcp__computer-use__left_click":
        return [inp["coordinate"]] if inp.get("coordinate") else []
    if name == "mcp__computer-use__computer_batch":
        acts = inp.get("actions") or []
        return [a["coordinate"] for a in acts[:1] if a.get("action") == "left_click" and a.get("coordinate")]
    if name == "mcp__claude-in-chrome__computer" and inp.get("action") == "left_click" and inp.get("coordinate"):
        return [inp["coordinate"]]
    return []


def items_from_session(lines: list[str], source: str) -> list[dict]:
    """Only the first click after each screenshot: later clicks in a batch happen on a screen we do not have."""
    out, img, intent = [], None, ""
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (r.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            t = b.get("type")
            if t == "text" and r.get("type") == "assistant":
                intent = b.get("text", "")
            elif t == "tool_result" and isinstance(b.get("content"), list):
                imgs = [x for x in b["content"] if x.get("type") == "image" and (x.get("source") or {}).get("data")]
                if imgs:
                    img = imgs[-1]["source"]
            elif t == "tool_use" and b.get("name") in CLICK_TOOLS and img is not None:
                pts = _clicks(b["name"], b.get("input") or {})
                if not pts or not intent.strip():
                    continue
                raw = base64.b64decode(img["data"])
                w, h = image_size(raw)
                x, y = pts[0]
                if 0 <= x < w and 0 <= y < h:  # click must be in the screenshot's own coordinate space
                    out.append({"id": f"{source}#{len(out)}", "media_type": img.get("media_type", "image/png"),
                                "image": img["data"], "width": w, "height": h, "intent": intent.strip()[-600:],
                                "target": [x, y], "tool": b["name"]})
                img = None  # the next click needs a fresh screenshot
    return out


def build_items(snapshot: Path) -> list[dict]:
    items = []
    with tarfile.open(Path(snapshot) / "raw.tar.gz") as tar:
        for m in sorted(tar.getmembers(), key=lambda m: m.name):
            if not m.isfile() or not m.name.endswith(".jsonl"):
                continue
            data = tar.extractfile(m).read()
            if b"computer-use__" not in data and b"claude-in-chrome__computer" not in data:
                continue
            items += items_from_session(data.decode("utf-8", "replace").splitlines(), m.name)
    return items


def items_for(snapshot: Path) -> list[dict]:
    frozen = RESULTS / "bench_d_items" / f"{Path(snapshot).name}.jsonl"
    if frozen.exists():
        return [json.loads(x) for x in frozen.read_text().splitlines() if x]
    items = build_items(snapshot)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text("".join(json.dumps(i) + "\n" for i in items))
    return items


PROMPT = ("This is a screenshot of a macOS screen, {w}x{h} pixels. The agent controlling the computer said, just "
          "before clicking:\n\"{intent}\"\nWhere should it click? Answer with only JSON: {{\"x\": <int>, \"y\": <int>}} "
          "in this image's pixel coordinates.")


def parse_point(text: str) -> tuple[float, float] | None:
    m = re.search(r'"?x"?\s*[:=]\s*(-?\d+(?:\.\d+)?)\D+?"?y"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def score(item: dict, point) -> dict:
    if point is None:
        return {"pass": False, "distance": None}
    d = math.dist(point, item["target"])
    return {"pass": d <= max(30.0, 0.03 * item["width"]), "distance": round(d, 1)}


def ollama_vision(model: str, base_url: str = net.SYNC_OLLAMA, seed: int = 0, timeout: int = 300):
    def ask(prompt: str, image_b64: str) -> str:
        body = {"model": model, "stream": False, "options": {"temperature": 0, "seed": seed},
                "messages": [{"role": "user", "content": prompt, "images": [image_b64]}]}
        req = net.urllib.request.Request(f"{base_url}/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with net.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"].get("content", "")
    return ask


def run(snapshot: Path, model: str, base_url: str = net.SYNC_OLLAMA, limit: int | None = None, ask=None) -> Path:
    from .bench_a import model_digest
    items = items_for(snapshot)[:limit] if limit else items_for(snapshot)
    use_backend = ask is None
    if ask is None:
        from .backends import vision
        ask = vision(model)
    rows = []
    for it in items:
        t = time.time()
        try:
            text = ask(PROMPT.format(w=it["width"], h=it["height"], intent=it["intent"]), it["image"])
            rows.append({"id": it["id"], "answer": text[:300], **score(it, parse_point(text)),
                         "seconds": round(time.time() - t, 2)})
        except Exception as e:  # unreachable model: not scored
            rows.append({"id": it["id"], "status": "infra_error", "error": repr(e)})
    ok = [r for r in rows if r.get("status") != "infra_error"]
    k = sum(r["pass"] for r in ok)
    report = {"benchmark": "D", "model": model, "snapshot": Path(snapshot).name,
              "model_record": __import__("forge.backends", fromlist=["record"]).record(model) if use_backend else None,
              "settings": {"temperature": 0, "radius": "max(30px, 3% width)"},
              "overall": {"items": len(ok), "infra_errors": len(rows) - len(ok), "accuracy": round(k / len(ok), 4) if ok else None,
                          "ci95": wilson(k, len(ok)) if ok else None,
                          "mean_seconds": round(sum(r["seconds"] for r in ok) / len(ok), 2) if ok else None},
              "rows": rows}
    from .backends import label
    out = RESULTS / re.sub(r"[^A-Za-z0-9._+-]", "_", label(model)) / Path(snapshot).name / "bench_d.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    return out
