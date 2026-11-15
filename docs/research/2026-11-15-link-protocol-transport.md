# Agent link: A2A state and transport options (research, 2026-11-15)

Legend: [V] verified against a primary source today (PyPI JSON, GitHub API, spec text); [S] from a secondary source or web-search summary; [U] unverified or my inference.

## 1. A2A protocol

- Version: spec **1.0.0** is the "Latest Released Version" on a2a-protocol.org/latest/specification (earlier: 0.3.0, 0.2.6, 0.1.0). GitHub repo a2aproject/A2A has tags v1.0.0 (2026-03-12) and v1.0.1 (2026-05-28), v0.3.0 (2025-07-30). 26.5k stars, Apache-2.0, pushed 2026-11-13. [V]
- Governance: donated by Google to the Linux Foundation (June 2025) [S: linuxfoundation.org press release]. On **2026-08-17** it moved into the Agentic AI Foundation (AAIF, an LF project that also hosts MCP). [V: aaif.io/blog/a2a-joins-aaif] GOVERNANCE.md: a 9-seat Technical Steering Committee (Google, Microsoft, Cisco, AWS, Salesforce, ServiceNow, SAP, IBM, Bloomberg). [V]
- Transports (spec sections 9-10 plus bindings): JSON-RPC 2.0 over HTTP, HTTP+JSON/REST, **gRPC**; streaming via SSE (SendStreamingMessage / SubscribeToTask); push notifications = server-initiated webhook POST to a client URL (Create/Get/List/Delete push config). Custom protocol bindings are allowed via the Agent Card `protocolBinding` field plus an extension and binding governance process. [V: spec text]
- Agent Card: served at `/.well-known/agent-card.json`; lists `supportedInterfaces` (URL, protocolBinding, optional `tenant`), skills, capabilities (streaming, pushNotifications, `extendedAgentCard`), `securitySchemes`. Authenticated `GetExtendedAgentCard` returns a fuller card. v1.0 adds **signed Agent Cards (JWS, RFC 7515)**. Discovery is well-known URI, registries or direct config; no built-in registry. [V]
- Task states (v1.0 enum): SUBMITTED, WORKING, INPUT_REQUIRED, AUTH_REQUIRED (interrupted), COMPLETED, FAILED, CANCELED, REJECTED (terminal). Streams start with the Task and then TaskStatusUpdateEvent / TaskArtifactUpdateEvent, and close on a terminal state. Messages to terminal tasks return UnsupportedOperationError. [V]
- Auth: schemes follow OpenAPI 3.2: API key, HTTP (bearer/basic), OAuth2, OpenID Connect, **mutual TLS**. HTTPS is required in production. A `tenant` field provides multi-tenant routing. v1.0 breaking changes: unified Part type, enum renames, google.rpc.Status errors, cursor pagination (a2a-protocol.org/latest/whats-new-v1). [V]
- Python SDK: **`a2a-sdk` 1.3.0** (PyPI, 2026-10-27), Apache-2.0, Python >=3.10, repo a2aproject/a2a-python (2.2k stars, pushed 2026-11-14, v1.3.0 / v1.2.3 / v1.2.2 in Oct 2026). Implements spec 1.0 over JSON-RPC, REST and gRPC, plus a 0.3 compatibility mode. Extras: http-server (starlette), grpc, sql (sqlite/postgres/mysql task stores), signing (pyjwt, card signatures), encryption (cryptography, which I read as at-rest encryption for the task store [U]), telemetry. Active, fast releases. Note: the old PyPI name `a2a-python` is a dead 0.0.1 stub. [V] JS SDK `@a2a-js/sdk` 1.3.0, Apache-2.0. [V]
- **Known gaps** (what the spec does not give you):
  - No peer-to-peer or NAT traversal. It assumes a reachable HTTPS endpoint per agent. [V by absence]
  - No application-layer E2E encryption: grepping the 1.0 spec text for "encrypt" and "end-to-end" finds nothing; security is TLS/mTLS per hop, so a relay terminating TLS sees plaintext. [V by absence]
  - No relay or store-and-forward concept, and no invite or pairing flow.
  - Authorization is delegated to the HTTP layer. There is no capability or delegation model, and no human-approval concept beyond the INPUT_REQUIRED / AUTH_REQUIRED states.
  - The base protocol has no shared workspace. Artifacts are per task.

## 2. A2A and MCP

- Official position: complementary layers. MCP = agent-to-tool/data (vertical), A2A = agent-to-agent (horizontal). The docs page a2a-and-mcp also suggests representing A2A agents as MCP resources. Both now sit under AAIF. [V]
- Bridges exist but are small and immature [V via `gh search`]: regismesquita/MCP_A2A (22 stars, last push 2025-05), a2anet/a2a-utils (3), shufflethis/a2a-router (3), jonwraymond/metatools-a2a (1). IBM/mcp-context-forge (4.8k stars, active) is an MCP gateway that I believe also federates A2A agents [U, not read]. None is worth depending on. Writing the MCP facade ourselves over `a2a-sdk` is small: roughly 6 tools (send_message, list_inbox, await_message, get_task, cancel_task, approve). Python `mcp` 2.3.0 (MIT, 2026-10-02), npm @modelcontextprotocol/sdk 1.32.2. [V]
- Claude Code **Channels** (research preview since 2026-03-20): an MCP server declares the `claude/channel` capability and emits `notifications/claude/channel` events into a running session. It can expose a reply tool, has **sender gating** against prompt injection, and has **permission relay** (forward tool-approval prompts to a remote channel). Custom channels need `--dangerously-load-development-channels` until allowlisted; Team/Enterprise orgs must enable them. The reference SDK is Node (`@modelcontextprotocol/sdk`). [V: code.claude.com/docs/en/channels-reference]. This is the right primitive for inbound messages that wake a Claude. Plain polling tools (await_message) are the fallback. Claude Code is at v2.1.325. [V]

## 3. P2P and relay building blocks

| Option | State (verified) | NAT traversal | Python/Node | Fit |
|---|---|---|---|---|
| **Tailscale** (WireGuard, DERP relays) | tailscale/tailscale 39k stars, BSD-3, active. libtailscale (C library) 354 stars. [V] | Best in class, direct when possible, DERP fallback | tsnet is Go only. Python means running `tailscaled` (userspace) and using 100.x addresses or MagicDNS, or a Go sidecar. libtailscale ships in-repo Python bindings (built from source, not on PyPI; tailscale/tailscale#7920 still open as of 2026-06; a build failure is reported in #10764) [V: search results, not tested] | Excellent for the user's own MacBook <-> Spark. Does not cover "other people" unless they join the tailnet or are shared a node. |
| **iroh** (QUIC, dial by public key, hole punching, relay fallback) | **1.0 stable (2026-06)**, iroh v1.4.1 (2026-10-22), 13k stars, Apache-2.0/MIT. PyPI `iroh` 1.2.0 (2026-10-22), npm `@number0/iroh` 1.2.0, uniffi bindings in iroh-ffi (Py/Swift/Kotlin/Node). Bindings cover the stable 1.0 surface only (endpoints, connections, tickets, relays), not blobs/docs/gossip. [V] | Hole punching, fallback to open relay ecosystem or self-hosted iroh-relay. Relays only forward QUIC packets that are already end-to-end TLS encrypted. [S: iroh README/docs] | Python and Node both | Strong fit for both modes: same endpoint model direct (P2P) or via relay, public-key identity built in, E2E by construction. |
| WebRTC / **aiortc** | aiortc 1.16.0 (2026-11-06), BSD-3, 5.1k stars. [V] | ICE/STUN, needs a signalling server and TURN for hard NATs | Python (and any browser) | Workable, but you build signalling, TURN and reconnect yourself. Less natural for daemons. Good only if browsers must join. |
| **libp2p** | py-libp2p 0.8.1 (2026-10-22), 0.x, 629 stars, license listed NOASSERTION on GitHub. [V] | AutoNAT, hole punching and circuit relay, but the Python implementation is immature | Python weak, JS stronger [U] | Not recommended in Python. |
| **PyNaCl / libsodium** | PyNaCl 1.6.2 (Apache-2.0). `cryptography` 50.0.2. [V] | n/a (crypto only) | Python, Node (libsodium-wrappers) | Sealed boxes, X25519 + XChaCha20-Poly1305, Ed25519 signatures. Good for E2E over a dumb relay. |
| **Noise** | Python `noiseprotocol` 0.3.1 last released **2020-03-03**, effectively abandoned. [V] | n/a | Rust `snow` is the real implementation [U] | Skip in Python. iroh already gives you Noise-like QUIC TLS 1.3. |
| **Matrix** | matrix-nio 0.26.0 (2026-07), 710 stars, ISC. vodozemac 0.10.0 (Olm/Megolm in Rust). [V] | Federation over HTTPS, a homeserver required | Python | Gives E2E rooms, invites, device keys and revocation for free, but heavy (homeserver, key verification UX) and latency is chat-grade. Good fallback for the human-style "free chat" mode only. [U on fitness] |

Latency: I found no first-party numbers I could verify. Reasoning only [U]: Tailscale direct and iroh direct paths are both UDP at roughly LAN/WAN RTT with around 1 ms overhead. Relayed paths add the relay's RTT (tens of ms). iroh publishes continuous perf measurements ("iroh-perf") that I did not open. Measure it on the real machines before committing.

Effort ranking (low to high): Tailscale-only (days) < iroh + own framing (about 1-2 weeks) < aiortc (2-3 weeks) < libp2p-py or Matrix (longest, most risk).

## 4. Shared workspace and agent identity

- Blackboard for LLM MAS: arXiv 2510.01285, "LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science" (Salemi et al., 2025). A central agent posts requests to a shared blackboard, subordinate agents self-select, and the paper reports 13-57% relative improvement over RAG and master-slave baselines. arXiv 2507.01701, "Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture" (Han and Zhang, 2025). [S: search summaries, papers not read in full] Practical takeaway: an append-only, typed, versioned log (events plus artifacts) that agents subscribe to. Build it as a CRDT-free single-writer log per workspace owner, replicated through the link. Note iroh-docs and iroh-gossip exist, but the Python/Node bindings do not cover them. [V]
- Identity and authorization (2026 literature, all [S]): arXiv 2603.24775 "AIP: Agent Identity Protocol" (Invocation-Bound Capability Tokens, attenuable chains; reports a Knostic scan finding no authentication on about 2,000 MCP servers). IETF drafts: draft-niyikiza-oauth-attenuating-agent-tokens-01 (June 2026, JWT-based, cites Macaroons and Biscuit), draft-asor-wimse-agent-delegation-chain-01 (Sept 2026). Google DeepMind "Intelligent AI Delegation" proposes delegation capability tokens built on macaroons. All of it is early and draft-grade.
- Libraries: **Biscuit** (eclipse-biscuit/biscuit, Apache-2.0, 1.2k stars). It is Ed25519-based, supports offline attenuation, and has Datalog policies. `biscuit-python` 0.4.0 (2025-09-26, repo pushed 2026-11-14, 18 stars) is thin but maintained. [V] `pymacaroons` 0.13.0 last released **2018**, so skip it. [V] Macaroons need a shared secret per verifier, which suits a multi-party mesh worse than Biscuit's public-key model.
- Signed Agent Cards (A2A 1.0, JWS) cover the identity-publication part only. They are not delegation.

## Recommended stack (one choice per layer)

| Layer | Choice | Reasoning |
|---|---|---|
| Protocol | **A2A 1.0** (task cards, SendStreamingMessage, signed cards), plus a small extension for "workspace" and "free chat" parts | Only real standard. LF/AAIF governance, 150+ orgs, and it maps to the structured-task and live-progress requirement. |
| SDK | **`a2a-sdk` 1.3.0** (Python) | Official, Apache-2.0, active, speaks 1.0 and 0.3, supports custom task stores. Wrap it in an MCP server (`mcp` 2.3.0) and add a Claude Code channel for inbound wake-ups, with polling tools as fallback. |
| P2P transport | **iroh 1.x** (PyPI `iroh`) carrying A2A JSON-RPC and SSE frames as QUIC streams (custom binding). Tailscale 100.x HTTP stays an allowed fast path between the user's own MacBook and Spark. | Dial-by-key, hole punching and relay fallback in one API. 1.0 is stable. Bindings are generated and may be rough, so run a spike. A cheaper Plan B for phase 1 is plain A2A over HTTPS on the Tailscale addresses. |
| Relay | **Self-hosted iroh-relay** | Forwards only already-encrypted QUIC, so it fits "relay forwards E2E messages only" without extra work. Daemon on each machine keeps a connection to it. |
| Encryption | **iroh's QUIC/TLS 1.3 with endpoint keys** for transport E2E. Add **PyNaCl sealed-box or signature envelopes** only for store-and-forward messages when the peer is offline. | Avoids Python Noise (abandoned) and avoids inventing a protocol. |
| Identity/authz | **Ed25519 per-person key (the iroh endpoint ID), signed Agent Cards, Biscuit tokens** for scoped capabilities (e.g. "may send task requests, may not run anything"). Revoke by revocation ID or by removing the key from the daemon allowlist. Invite link = a Biscuit or signed blob holding a relay address, a node ID and a single-use scope. Local-execution actions map to A2A INPUT_REQUIRED / AUTH_REQUIRED plus a local approval queue. | Public-key attenuation fits a multi-party mesh. Macaroons and pymacaroons do not. The IETF and AIP drafts are too early to build on, so copy their ideas only. |
| Workspace | **Append-only event log per workspace** (typed entries, artifacts by hash, SQLite in the daemon), replicated over the link. Exposed through MCP tools and A2A artifacts. | Matches the blackboard literature. It avoids CRDT and Matrix complexity, and you can later swap in iroh-docs once bindings cover it. |

Biggest risks to spike first: (1) the iroh Python binding's stream API and its async fit with a2a-sdk, (2) the Channels research-preview restriction (needs the dev flag), (3) the real MacBook <-> Spark latency over Tailscale vs iroh. Matrix is the fallback for free chat with other people only if the custom path proves too costly.

## Sources
- https://a2a-protocol.org/latest/specification/ , /latest/whats-new-v1/ , /latest/topics/a2a-and-mcp/
- https://github.com/a2aproject/A2A (GOVERNANCE.md, MAINTAINERS.md, releases)
- https://github.com/a2aproject/a2a-python , https://pypi.org/project/a2a-sdk/
- https://aaif.io/blog/a2a-joins-aaif (2026-08-17)
- https://github.com/n0-computer/iroh , https://github.com/n0-computer/iroh-ffi , https://pypi.org/project/iroh/ , npm @number0/iroh
- https://github.com/aiortc/aiortc , https://github.com/libp2p/py-libp2p , https://github.com/tailscale/libtailscale
- https://pypi.org/project/{pynacl,noiseprotocol,pymacaroons,biscuit-python,matrix-nio,vodozemac,mcp}/
- https://code.claude.com/docs/en/channels-reference
- https://arxiv.org/abs/2510.01285 , https://arxiv.org/abs/2507.01701 , https://arxiv.org/abs/2603.24775
- https://datatracker.ietf.org/doc/html/draft-niyikiza-oauth-attenuating-agent-tokens-00 , draft-asor-wimse-agent-delegation-chain

## Addendum (late search results, titles/snippets only, not read in full) [S]
- Claude Code documents **cross-session messaging** (https://code.claude.com/docs/en/cross-session-messaging): Claude delivers a message from one of your sessions to another. Check it before building local inbound delivery, since it may cover the same-machine case.
- Further blackboard evidence: SE-Blackboard, a shared-state architecture for multi-agent software engineering (IEEE, https://ieeexplore.ieee.org/abstract/document/11527206); Google Research page for the data-science blackboard paper (research.google/pubs/blackboard-multi-agent-systems-for-information-discovery-in-data-science/).
- Risk for a shared workspace: arXiv 2512.15790 (XAMT) studies covert memory tampering in multi-agent systems. Treat other people's workspace entries as untrusted input, and gate by sender and scope.
- Capability-token overviews (macaroons, Biscuit, UCAN) agree that attenuation without contacting the issuer fits delegation chains, which supports the Biscuit choice. UCAN was not evaluated here.
