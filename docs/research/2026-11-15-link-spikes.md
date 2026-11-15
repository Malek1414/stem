# Link spikes (2026-11-15)

## Spike 1: iroh + a2a-sdk on the Mac — PASS

Environment: throwaway uv venv, Python 3.12, PyPI `iroh` 1.2.0, `a2a-sdk` 1.3.0. Script:
`docs/research/spikes/spike1_iroh_a2a.py`. Relays and n0 discovery disabled (`apply_minimal`,
`RelayMode.disabled()`), bound to 127.0.0.1, so nothing left the machine.

What ran: two iroh endpoints; endpoint A connected to B with ALPN `agent-forge/a2a/1`, opened a
bidirectional QUIC stream and sent a JSON-RPC envelope whose params are an A2A `SendMessageRequest`
built with the SDK's protobuf types (`a2a.types.Message`, `Part(text=...)`, `Role.ROLE_USER`). B parsed
it back into `SendMessageRequest` (schema-validated) and replied with an A2A `Message` (`ROLE_AGENT`).

| Measure (loopback, same Mac) | Value |
|---|---|
| iroh connect (QUIC + TLS handshake) | 19.2 ms |
| first request -> reply on an open connection | 5.7 ms |
| reply | "ack: task card: set x to 2" |

Findings that shape the build:
- iroh's Python API is fully async (`EndpointBuilder.bind`, `Endpoint.connect/accept_next`,
  `Connection.open_bi/accept_bi`, `SendStream.write_all/finish`, `RecvStream.read_to_end`) and fits asyncio.
- a2a-sdk 1.x types are protobuf (`a2a_pb2`); JSON mapping via `google.protobuf.json_format`.
- The SDK separates transport from logic: server side `RequestHandler` / `DefaultRequestHandlerV2`
  take protobuf requests (`on_message_send(SendMessageRequest, ServerCallContext)`), client side
  accepts any `ClientTransport` (abstract: send_message, send_message_streaming, get_task, list_tasks,
  cancel_task, subscribe, push-config methods, get_extended_agent_card, close). So the link needs an
  `IrohClientTransport` plus a server loop that dispatches stream frames to `DefaultRequestHandlerV2`;
  task lifecycle, streaming and stores come from the SDK.
- Unverified: the JSON-RPC method name in the A2A 1.0 spec ("SendMessage" was assumed for the envelope).
  It does not matter for our own iroh framing, but must match the spec for the HTTP binding.
- Not yet tested: NAT traversal and relays (needs network), streaming responses, many concurrent streams.

## Spike 2: getting messages into a running Claude Code session — two paths, one needs no flag

Sources: code.claude.com/docs/en/cross-session-messaging and /channels (read 2026-11-15). This Mac:
Claude Code 2.1.323; this session's inbox socket exists (`CLAUDE_CODE_MESSAGING_SOCKET`, mode 0600).

1. **Inbox socket (no flag, chosen as the default delivery path).** Every interactive and `-p` session
   (not `--bare`) binds a Unix socket restricted to the OS user; its path is in
   `CLAUDE_CODE_MESSAGING_SOCKET`, with a per-session token in `CLAUDE_CODE_MESSAGING_TOKEN`
   (optional `{"type":"auth","token":...}` first line on macOS). The link daemon, running as Malek's
   user, can post a message into a session; it is delivered between tool calls or starts a new turn
   when idle. Claude Code's own guards apply on top of ours: a delivered message can't approve
   prompts, change configuration or run slash commands. Inbound control `crossSessionInbound`
   (accept / hold / refuse). Default when unset: a session that bypasses permissions (Malek's usual
   mode) HOLDS messages from non-child senders behind an approval dialog that expires after
   `dialogExpiry` (5 min). Plain text only, ~1M-char cap, burst and loop throttling, 100 held max.
   Own sessions on other machines are reachable only via Anthropic's Remote Control, and never other
   people's sessions, so cross-person traffic must come through the link.
   Not yet verified: the exact JSON line format for a message on the socket (the page documents the
   auth line only). Test against a throwaway `claude -p` session, never this live one.
2. **Channels (richer, but a dev flag for custom ones).** An MCP server that pushes events into the
   session, with sender allowlists and optional permission relay (approve tool prompts remotely).
   Pro/Max users without an org skip the org gate, but during the research preview `--channels` only
   accepts allowlisted plugins; a custom channel needs `--dangerously-load-development-channels
   server:<name>`. Use later for remote approval of held requests; not required for delivery.

Design consequence: the link daemon delivers inbound messages through the inbox socket by default;
Channels is an optional add-on for remote approvals. Inbound messages from other people stay subject
to both our approval gate and Claude Code's hold/accept controls.

## Spike 3: Mac -> Spark latency over the existing NVIDIA Sync tunnel

20 sequential `GET /api/version` calls to Ollama on localhost:11434 (tunnel), 2026-11-15, Spark idle:
min 18.1 ms, median 30.4 ms, p95 384.1 ms, max 520.3 ms. The median is fine for task cards and progress
events; the tail is large (tunnel or Wi-Fi), so the link must stream and not block on each round trip.
One sample of 20 calls: repeat on wired network and over iroh direct before drawing a firm conclusion.
