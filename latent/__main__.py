"""python -m latent bench --model Qwen/Qwen3-0.6B [--n 60] [--conditions a,b] [--mbps 100 --rtt-ms 17]"""
import argparse
import json

from .bench import CONDITIONS, run

ap = argparse.ArgumentParser(prog="latent")
sub = ap.add_subparsers(dest="cmd", required=True)
b = sub.add_parser("bench")
b.add_argument("--model", default="Qwen/Qwen3-0.6B")
b.add_argument("--n", type=int, default=60)
b.add_argument("--seed", type=int, default=20261115)
b.add_argument("--conditions", default=",".join(CONDITIONS))
b.add_argument("--mbps", type=float, default=100.0)
b.add_argument("--rtt-ms", type=float, default=17.0)
b.add_argument("--ratio", type=float, default=0.3)
b.add_argument("--latent-steps", type=int, default=20)
b.add_argument("--device")
b.add_argument("--ctx-chars", default="0", help="comma list for a long-context sweep, e.g. 8000,32000,64000,120000")
a = ap.parse_args()
from .bench import Runner
R = None
for cc in [int(x) for x in a.ctx_chars.split(",")]:
    R = R or Runner(a.model, a.device)  # load the model once for the whole sweep
    res = run(a.model, a.n, a.seed, tuple(a.conditions.split(",")), mbps=a.mbps, rtt_ms=a.rtt_ms, ratio=a.ratio,
              latent_steps=a.latent_steps, device=a.device, ctx_chars=cc, runner=R)
    print(json.dumps({k: res[k] for k in ("model", "device", "n", "ctx_chars", "path")} | {"summary": res["summary"]}, indent=1), flush=True)
