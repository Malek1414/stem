# agent-forge, sub-project 2: the link

Date: 2026-11-15. Status: requirements fixed by interview; design waits on the research in `docs/research/`.

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

## Open until research returns

- Whether operator and coder can talk through latent channels (KV cache / hidden states) instead of
  text, and what that needs from the serving stack (Ollama exposes neither).
- Transport, relay and encryption choice; A2A SDK maturity.
- Shared workspace model (blackboard) and authorization model (capability tokens vs plain keys).
