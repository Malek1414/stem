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
