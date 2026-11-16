"""python -m forge <command>

snapshot [--limit N] [--no-raw]      build data/snapshots/vNNNN from ~/.claude/projects
verify <snapshot> [--rebuild]       re-hash a snapshot; --rebuild also rebuilds it from raw.tar.gz
bench-a <snapshot> --model M        run benchmark A against an Ollama endpoint
bench-b <snapshot> --model M        run benchmark B (sandboxed task replay)
bench-b-tasks <snapshot>            build and freeze benchmark B tasks, print a summary
bench-c <snapshot> --model M        run benchmark C (judged outcomes, pinned local judge)
publish <snapshot> [--repo R]       copy a snapshot to a private HF dataset repo and verify hashes
compare <snapshot> <a|b|c> M1 M2    paired task-level comparison with an exact sign test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import net, bench_a, bench_b, bench_c, bench_d, snapshot


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
    b.add_argument("--base-url", default=net.SYNC_OLLAMA)
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--limit", type=int)
    b.add_argument("--num-ctx", type=int)
    bb = sub.add_parser("bench-b")
    bb.add_argument("snapshot")
    bb.add_argument("--model", required=True)
    bb.add_argument("--base-url", default=net.SYNC_OLLAMA)
    bb.add_argument("--limit", type=int)
    bb.add_argument("--match", help="only tasks whose prompt contains this text")
    bb.add_argument("--coder", help="run as operator (--model) + coder (this model) with task-card handoffs")
    sf = sub.add_parser("sft", help="training examples (chat + tool calls) from a snapshot's train split")
    sf.add_argument("snapshot")
    sf.add_argument("--out", required=True)
    sf.add_argument("--max-chars", type=int, default=40_000)
    bc = sub.add_parser("bench-c")
    bc.add_argument("snapshot")
    bc.add_argument("--model", required=True)
    bc.add_argument("--judge", default=bench_c.DEFAULT_JUDGE)
    bc.add_argument("--agent", default="v1", choices=sorted(bench_c.AGENT_PROMPTS),
                    help="agent-side instructions (the benchmark itself is the same for every version)")
    bc.add_argument("--base-url", default=net.SYNC_OLLAMA)
    bc.add_argument("--limit", type=int)
    bc.add_argument("--replay-tools", action="store_true", help="give the agent the original turn's app/web tools, answered from the recording")
    bd = sub.add_parser("bench-d", help="click grounding on the user's own screens (vision models)")
    bd.add_argument("snapshot")
    bd.add_argument("--model", required=True)
    bd.add_argument("--base-url", default=net.SYNC_OLLAMA)
    bd.add_argument("--limit", type=int)
    cp = sub.add_parser("compare")
    cp.add_argument("snapshot")
    cp.add_argument("bench", choices=["a", "b", "c"])
    cp.add_argument("model_a")
    cp.add_argument("model_b")
    pb = sub.add_parser("publish")
    pb.add_argument("snapshot")
    pb.add_argument("--repo")
    bt = sub.add_parser("bench-b-tasks")
    bt.add_argument("snapshot")
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
    if a.cmd == "bench-b":
        out = bench_b.run(Path(a.snapshot), a.model, a.base_url, limit=a.limit, match=a.match, coder=a.coder)
        print(out)
        print(json.dumps(json.loads(out.read_text())["overall"], indent=1))
        return 0
    if a.cmd == "bench-d":
        out = bench_d.run(Path(a.snapshot), a.model, a.base_url, limit=a.limit)
        print(out)
        print(json.dumps(json.loads(out.read_text())["overall"], indent=1))
        return 0
    if a.cmd == "sft":
        from . import sft
        print(json.dumps(sft.build(Path(a.snapshot), Path(a.out), a.max_chars), indent=1))
        return 0
    if a.cmd == "bench-c":
        out = bench_c.run(Path(a.snapshot), a.model, a.judge, a.base_url, limit=a.limit, replay=a.replay_tools, agent=a.agent)
        print(out)
        print(json.dumps(json.loads(out.read_text())["overall"], indent=1))
        return 0
    if a.cmd == "compare":
        from .compare import compare
        print(json.dumps(compare(Path(a.snapshot).name, a.bench, a.model_a, a.model_b), indent=1))
        return 0
    if a.cmd == "publish":
        from . import publish
        r = publish.publish(Path(a.snapshot), a.repo)
        log = Path(__file__).resolve().parent.parent / "results" / "published.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as fh:
            fh.write(json.dumps({"snapshot": Path(a.snapshot).name, **r}) + "\n")
        print(json.dumps(r, indent=1))
        return 1 if r["mismatches"] else 0
    if a.cmd == "bench-b-tasks":
        for t in bench_b.tasks_for(Path(a.snapshot)):
            files = ", ".join(f"{r}{'' if f['expected'] is not None else ' (changed-only)'}" for r, f in t["files"].items())
            print(f"{t['ts'][:16]} {t['commit'][:8]} | {t['prompt'][:60]!r} | {files}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
