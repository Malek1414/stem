# stem

Two agents trained on your own Claude Code sessions, and a link that lets Claudes talk to each other.

- **Operator**: low-latency, abliterated, precise at computer use and tool calls. Runs next to you (on a DGX Spark here) and drives the Mac.
- **Coder**: writes code from the context the operator hands it.
- **Link**: an encrypted peer-to-peer channel (iroh QUIC + A2A messages) between the operator and the coder, and between your Claude and other people's Claudes. Text, typed task cards with human approval, shared workspaces, and (between your own machines) KV-cache handoff.

Order of work: dataset and benchmarks → link → choosing the operator and the coder → training. Live status is in [docs/STATUS.md](docs/STATUS.md), and every measured number is in [docs/RESULTS.md](docs/RESULTS.md).

## Install the link in your Claude Code

Needs [uv](https://docs.astral.sh/uv/). While the repo is private you also need read access to it (`gh auth login`, then `gh auth setup-git`).

```
/plugin marketplace add Malek1414/stem
/plugin install stem@stem
```

Restart Claude Code. On first start the plugin launches a link daemon in the background. Run `/stem:link-setup` to make it start at login. Then, to connect two Claudes:

1. One side: "make a link invite". This gives a single-use ticket, valid for 72 hours. Send it privately.
2. Other side: "join this invite: <ticket>".

Cards from other people wait for your approval before anything runs. Only `role=self` peers (your own machines) are trusted without approval. See [the link skill](plugins/stem/skills/link/SKILL.md).

## Layout

| path | what |
|---|---|
| `link/` | daemon (`python -m link run`), CLI, MCP server, task cards, KV-blob channel |
| `forge/` | dataset snapshots from Claude Code sessions, benchmarks A-D, agent harness, operator+coder duo, model backends (Ollama, llama.cpp) |
| `latent/` | latent communication: KV handoff (fp16/int8), KVComm-style layer selection, LatentMAS, and its benchmark (`--level hard` for multi-hop tasks) |
| `plugins/stem/` | the Claude Code plugin (MCP server, skills, daemon installer) |
| `docs/` | status, results, design specs, research notes |

## Development

```
uv sync                      # dev + latent (torch, transformers) groups
uv run pytest -q
uv run python -m link run --name me --mode relay
uv run python -m forge --help          # snapshot, bench-a/b/c/d, compare, ...
uv run python -m latent bench --model Qwen/Qwen3-0.6B --n 20 --level hard
```

Benchmarks are reproducible: snapshots are frozen and versioned, every result records the model digest or GGUF build, the seed and the harness version, and infrastructure errors are excluded from scores and rerun. Session data (`data/`), results with transcripts (`results/`) and scratch files (`work/`) stay out of git.

Models in use are abliterated only: Holo4-35B-A3B-heretic (operator), Huihui-Qwen3.6-35B-A3B-abliterated (coder), Huihui-Qwen3.8-27B-abliterated (dense alternative), served with llama.cpp. The reasoning is in [docs/research/2026-11-15-model-selection.md](docs/research/2026-11-15-model-selection.md).
