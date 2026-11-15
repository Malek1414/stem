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
a = ap.parse_args()
res = run(a.model, a.n, a.seed, tuple(a.conditions.split(",")), mbps=a.mbps, rtt_ms=a.rtt_ms, ratio=a.ratio,
          latent_steps=a.latent_steps, device=a.device)
print(json.dumps({k: res[k] for k in ("model", "device", "n", "path")} | {"summary": res["summary"]}, indent=1))
