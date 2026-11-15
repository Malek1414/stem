# agent-forge, sub-project 2: the link

Date: 2026-11-15. Status: design approved ("keep going", 2026-11-15); v1 built in `link/`, 13 tests.

## Goal

A real-time channel that (1) connects the operator and the coder with low latency, and (2) connects
Malek's Claude to other people's Claudes and agents. Every design choice must cite evidence
(user request: "scientific base for everything").

## Requirements (from the interview)

| Topic | Decision |
|---|---|
| Topology | Both modes must exist: pure peer-to-peer (direct, e.g. Tailscale/WebRTC) and local daemon + relay (relay only forwards end-to-end-encrypted messages). |
| Protocol | A2A-compatible, so agents we didn't build can join. Exposed to Claude Code as an MCP server. |
| Handoff | Own fine-tuned open models: structured task cards (goal, constraints, context, done-criteria) with live progress streaming. With other people: shared live workspace, free chat as fallback. |
| Trust | Others can message and send task requests; anything that would run on Malek's machine waits for his yes. Invite links, one revocable key per person. |
| Not | Not built on Nutcracker. |

## Design (evidence in docs/research/2026-11-15-*.md, spikes in 2026-11-15-link-spikes.md)

| Layer | v1 as built | Evidence / reason |
|---|---|---|
| Messages | A2A 1.0 `SendMessage` with SDK protobuf types; cards and progress are A2A Messages with a typed data part (`application/vnd.agent-forge.task-card+json`, `...progress+json`) | Any A2A agent can read them; spike 1 |
| Operator <-> coder | Typed task card (goal, constraints, context, done criteria + check command, freeform escape field); progress capped at 280 chars; `CARD_SCHEMA` doubles as Ollama `format` | Routed Graph Handoff arXiv 2608.25277 |
| Latent channel | Not in v1 | Ollama exposes no KV/hidden states; papers test small dense models on QA, not tool use or hybrid MoE; needs more pair data than 712 handoffs |
| Transport | iroh 1.2 QUIC, one request per bidirectional stream. `p2p` mode: no discovery, no relay. `relay` mode: n0 discovery + relays or a custom relay URL; relays only see encrypted QUIC | Spike 1: 19 ms connect, 5.7 ms request/reply on loopback |
| Mac <-> Spark | Existing NVIDIA Sync tunnel (spike 3: 30 ms median, 384 ms p95), so streaming, never blocking per round trip | CLAUDE.md: never plain ssh |
| Identity | Each node's iroh Ed25519 key; QUIC/TLS authenticates the endpoint id on every connection | iroh |
| Invites and trust | Signed (node key), single-use (nonce), expiring invite `afl1.…` carrying role (`self` = Malek's machines, `peer` = others) and scopes (message, request, workspace). Revocation is immediate (checked per request) | Interview |
| Approval | Cards from role `peer` wait in `approvals` until Malek approves or denies; the sender gets `accepted`/`rejected` progress. Role `self` cards are accepted at once | Interview |
| Workspace | Append-only log hosted by its creator; members append/read through the host; every entry records author and `trusted` (false for other people) | XAMT arXiv 2512.15790 |
| Claude Code | MCP server `python -m link mcp` (14 tools) talking to the daemon over a 0600 Unix control socket; `link_wait` long-polls | MCP SDK 2.3 MCPServer |

Deviations from the proposal, with reasons:
- Biscuit tokens deferred: authenticated endpoint ids + per-peer scopes + revocation already give one revocable key per person; Biscuit's gain is offline delegation chains, which nothing needs yet.
- Live push into a session via the Claude Code inbox socket (spike 2) is not wired yet: the socket's message line format is undocumented and must be tested on a throwaway `claude -p` session first. Until then Claude reads the link with `link_wait`.

## Deployment (2026-11-15, approved by Malek: "all of them")

- Mac: launchd agent `com.agent-forge.link` (`~/Library/LaunchAgents/com.agent-forge.link.plist`), node `malek`,
  UDP 47100, relay mode, log `~/Library/Logs/agent-forge-link.log`. MCP server `link` registered at user scope.
- Spark: node `spark`, systemd user unit `agent-forge-link.service`, code in `~/agent-forge`, venv `~/agent-forge/.venv`.
  Reached only through the NVIDIA Sync tunnel to the dashboard's JupyterLab (kernel API driven from Malek's
  logged-in browser tab), never plain ssh. Linger could not be enabled without sudo, so the unit lives as long as
  Malek's user session on the Spark (JupyterLab keeps one open).
- p2p mode failed between the two (the Mac reaches the Spark's subnet through a VPN interface, the Spark cannot dial
  back to 192.168.178.x). Relay mode works and iroh then upgrades to a direct UDP path: measured bare round trips
  min 10 ms, median 17-63 ms, with tunnel jitter episodes of ~0.9-1.1 s median (same p95 pattern as spike 3).
- Throughput Mac -> Spark measured with KV-sized blobs while a model download and the baseline ran: 0.6-2.4 Mbit/s.
  At that rate a KV cache (MBs) loses to a text card (KBs) across machines; latent handoff belongs co-located.

## Latent channel (added on Malek's request, overriding the research recommendation to defer)

- `latent/` implements three training-free same-backbone methods: full KV handoff, KVComm-style selected layers
  (arXiv 2510.03346), LatentMAS latent thoughts (arXiv 2511.20639). Binary `AFB1` frames carry KV blobs over the
  link; only peers with role `self` may send them, checked before the body is read, because latent content cannot be
  audited (research doc, point 5) and other people's closed models cannot take part anyway.
- Placement: the operator must fit on the Spark, so operator and coder run side by side there and share KV in GPU
  memory; the Mac <-> Spark link carries tool calls and text; links to other people stay text (A2A).
- Benchmark `python -m latent bench` (7 conditions, accuracy with Wilson CI, time and bytes, e2e per link profile).
- Not yet: hybrid Qwen3-Next caches (Gated DeltaNet recurrent state is not a KV cache; needs its own serialiser).

## Still open

- Transport, relay and encryption choice; A2A SDK maturity.
- Shared workspace model (blackboard) and authorization model (capability tokens vs plain keys).
