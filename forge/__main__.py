"""python -m forge <command>

snapshot [--limit N] [--no-raw]      build data/snapshots/vNNNN from ~/.claude/projects
verify <snapshot> [--rebuild]       re-hash a snapshot; --rebuild also rebuilds it from raw.tar.gz
bench-a <snapshot> --model M        run benchmark A against an Ollama endpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bench_a, snapshot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--limit", type=int)
    s.add_argument("--no-raw", action="store_true")
    s.add_argument("--out")
    v = sub.add_parser("verify")
    v.add_argument("snapshot")
    v.add_argument("--rebuild", action="store_true")
    b = sub.add_parser("bench-a")
    b.add_argument("snapshot")
    b.add_argument("--model", required=True)
    b.add_argument("--base-url", default="http://localhost:11434")
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--limit", type=int)
    b.add_argument("--num-ctx", type=int, default=32_768)
    a = ap.parse_args(argv)

    if a.cmd == "snapshot":
        out = snapshot.build(out_dir=Path(a.out) if a.out else None, raw=not a.no_raw, limit=a.limit)
        m = json.loads((out / "manifest.json").read_text())
        print(out)
        print(json.dumps({"heldout": m["heldout_project"], **m["counts"]}, indent=1))
        return 0
    if a.cmd == "verify":
        problems = snapshot.verify(Path(a.snapshot), rebuild=a.rebuild)
        print("\n".join(problems) or "ok: every hash matches")
        return 1 if problems else 0
    if a.cmd == "bench-a":
        out = bench_a.run(Path(a.snapshot), a.model, a.base_url, n=a.n, limit=a.limit, num_ctx=a.num_ctx)
        r = json.loads(out.read_text())
        print(out)
        print(json.dumps({"overall": r["overall"], "by_agent": r["by_agent"]}, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
