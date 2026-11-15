"""One link node: an iroh endpoint that speaks A2A messages plus a few link operations.

Wire format: one request per QUIC bidirectional stream, JSON, then one JSON response. Operations:
  join       redeem a signed, single-use invite (adds the caller's authenticated endpoint id as a peer)
  a2a        an A2A SendMessage request (protobuf JSON mapping): text, task card or progress event
  ws_append  append to a workspace this node hosts (append-only; authors recorded)
  ws_read    read a workspace this node hosts
Every operation except join requires a known, unrevoked peer with the matching scope. Task cards from
anyone but Malek's own nodes (role "self") wait for his approval before they count as accepted.
Workspace entries from others are marked untrusted (memory tampering risk: arXiv 2512.15790)."""
from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid

import iroh
from a2a import types as t
from google.protobuf import json_format

from . import cards
from .state import SCOPES, State

ALPN = b"agent-forge/link/1"
MAX_FRAME = 8 << 20


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


class LinkError(Exception):
    pass


class Node:
    def __init__(self, state: State, bind: str = "0.0.0.0:0", on_event=None):
        self.state = state
        self.bind = bind
        self.on_event = on_event or (lambda entry: None)
        self.ep: iroh.Endpoint | None = None
        self._conns: dict[str, iroh.Connection] = {}
        self._accept_task: asyncio.Task | None = None

    # --- lifecycle ---------------------------------------------------------------------------
    async def start(self) -> None:
        b = iroh.EndpointBuilder()
        b.secret_key(self.state.secret.to_bytes())
        mode, relay = self.state.settings.get("mode", "p2p"), self.state.settings.get("relay_url")
        if mode == "relay":
            b.apply_n0()  # discovery + relays; relays only ever see end-to-end encrypted QUIC
            if relay:
                b.relay_mode(iroh.RelayMode.custom_from_urls([relay]))
        else:
            b.apply_minimal()  # peer-to-peer only: no discovery service, no relay
            b.relay_mode(iroh.RelayMode.disabled())
        b.alpns([ALPN])
        b.bind_addr(self.bind)
        self.ep = await b.bind()
        self._accept_task = asyncio.create_task(self._accept_loop())

    async def close(self) -> None:
        if self._accept_task:
            self._accept_task.cancel()
        if self.ep:
            await self.ep.close()

    @property
    def id(self) -> str:
        return self.state.id

    def ticket(self) -> str:
        return str(iroh.EndpointTicket.from_addr(self.ep.addr()))

    # --- invites -----------------------------------------------------------------------------
    def make_invite(self, role: str = "peer", scopes: list[str] | None = None, ttl_hours: float = 72) -> str:
        if role not in ("self", "peer"):
            raise ValueError("role is self (your own machines) or peer (other people)")
        scopes = list(scopes or (SCOPES if role == "self" else ["message", "request", "workspace"]))
        payload = {"v": 1, "inviter": self.id, "inviter_name": self.state.settings["name"], "ticket": self.ticket(),
                   "role": role, "scopes": sorted(scopes), "expires": time.time() + ttl_hours * 3600,
                   "nonce": uuid.uuid4().hex}
        sig = self.state.secret.sign(_canon(payload)).to_bytes()
        return "afl1." + _b64(_canon({"p": payload, "s": _b64(sig)}))

    @staticmethod
    def read_invite(invite: str) -> tuple[dict, bytes]:
        if not invite.startswith("afl1."):
            raise LinkError("not an agent-forge link invite")
        outer = json.loads(_unb64(invite[5:]))
        return outer["p"], _unb64(outer["s"])

    async def join(self, invite: str) -> dict:
        payload, _ = self.read_invite(invite)
        res = await self._request_ticket(payload["ticket"], {
            "op": "join", "invite": invite, "name": self.state.settings["name"], "ticket": self.ticket()})
        scopes = list(SCOPES)  # what we let the inviter do here; their cards still need approval unless role self
        return self.state.add_peer(payload["inviter"], res["name"], payload["ticket"], payload["role"], scopes)

    # --- outgoing ----------------------------------------------------------------------------
    async def _connect(self, peer_id: str, ticket: str) -> iroh.Connection:
        conn = self._conns.get(peer_id)
        if conn is None:
            addr = iroh.EndpointTicket.from_string(ticket).endpoint_addr()
            conn = await self.ep.connect(addr, ALPN)
            self._conns[peer_id] = conn
        return conn

    async def _roundtrip(self, conn: iroh.Connection, req: dict) -> dict:
        bi = await conn.open_bi()
        s = bi.send()
        await s.write_all(json.dumps(req).encode())
        await s.finish()
        res = json.loads(await bi.recv().read_to_end(MAX_FRAME))
        if not res.get("ok"):
            raise LinkError(res.get("error", "request failed"))
        return res.get("result", {})

    async def _request_ticket(self, ticket: str, req: dict) -> dict:
        addr = iroh.EndpointTicket.from_string(ticket).endpoint_addr()
        conn = await self.ep.connect(addr, ALPN)
        return await self._roundtrip(conn, req)

    async def _request(self, peer_name: str, req: dict) -> dict:
        peer = self.state.peer_by_name(peer_name) or self.state.peer(peer_name)
        if not peer:
            raise LinkError(f"no active peer called {peer_name}")
        for attempt in (1, 2):
            try:
                return await self._roundtrip(await self._connect(peer["id"], peer["ticket"]), req)
            except LinkError:
                raise
            except Exception:
                self._conns.pop(peer["id"], None)  # stale connection: reconnect once
                if attempt == 2:
                    raise

    async def ping(self, peer_name: str, n: int = 10) -> dict:
        """Round trips over the open connection, with no disk writes on either side."""
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            await self._request(peer_name, {"op": "ping"})
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return {"n": n, "median_ms": round(ts[n // 2], 2), "p95_ms": round(ts[min(n - 1, int(0.95 * n))], 2),
                "min_ms": round(ts[0], 2), "paths": self.paths(peer_name)}

    def paths(self, peer_name: str) -> list[dict]:
        """Network paths of the open connection to a peer: direct UDP or relay, and their RTT."""
        peer = self.state.peer_by_name(peer_name) or self.state.peer(peer_name)
        conn = self._conns.get(peer["id"]) if peer else None
        if conn is None:
            return []
        return [{"kind": "relay" if p.is_relay else "direct", "remote": str(p.remote_addr), "selected": p.is_selected,
                 "rtt_ms": p.rtt_ms} for p in conn.paths()]

    async def _send_a2a(self, peer_name: str, msg: t.Message) -> dict:
        params = json_format.MessageToDict(t.SendMessageRequest(message=msg))
        return await self._request(peer_name, {"op": "a2a", "method": "SendMessage", "params": params})

    async def send_text(self, peer: str, text: str, context_id: str = "") -> dict:
        return await self._send_a2a(peer, cards.text_message(text, context_id))

    async def send_card(self, peer: str, card: cards.TaskCard) -> dict:
        return await self._send_a2a(peer, cards.card_message(card))

    async def send_progress(self, peer: str, card_id: str, kind: str, text: str = "", data: dict | None = None) -> dict:
        return await self._send_a2a(peer, cards.progress_message(card_id, kind, text, data))

    async def ws_append(self, ws: str, kind: str, body, host: str | None = None) -> dict:
        if host is None:
            return self.state.ws_append(ws, self.id, self.state.settings["name"], True, kind, body)
        return await self._request(host, {"op": "ws_append", "ws": ws, "kind": kind, "body": body})

    async def ws_read(self, ws: str, after: int = 0, host: str | None = None) -> list[dict]:
        if host is None:
            return self.state.ws_read(ws, after)
        return (await self._request(host, {"op": "ws_read", "ws": ws, "after": after}))["entries"]

    # --- approvals ---------------------------------------------------------------------------
    async def approve(self, card_id: str) -> dict:
        a = self.state.approvals().get(card_id)
        if not a:
            raise LinkError(f"nothing waiting for approval with id {card_id}")
        self.state.set_approval(card_id, None)
        entry = self._event({"type": "card", "state": "accepted", "card": a["card"], "from": a["from"],
                             "from_name": a["from_name"], "trusted": False, "approved": True})
        await self.send_progress(a["from"], card_id, "accepted", "approved by the owner")
        return entry

    async def deny(self, card_id: str, reason: str = "") -> None:
        a = self.state.approvals().get(card_id)
        if not a:
            raise LinkError(f"nothing waiting for approval with id {card_id}")
        self.state.set_approval(card_id, None)
        await self.send_progress(a["from"], card_id, "rejected", reason or "declined by the owner")

    # --- incoming ----------------------------------------------------------------------------
    def _event(self, entry: dict) -> dict:
        e = self.state.inbox_add(entry)
        self.on_event(e)
        return e

    async def _accept_loop(self) -> None:
        while True:
            inc = await self.ep.accept_next()
            if inc is None:
                return
            asyncio.create_task(self._serve(inc))

    async def _serve(self, inc) -> None:
        try:
            conn = await (await inc.accept()).connect()
        except Exception:
            return
        remote = str(conn.remote_id())
        while True:
            try:
                bi = await conn.accept_bi()
            except Exception:
                return
            asyncio.create_task(self._stream(remote, bi))

    async def _stream(self, remote: str, bi) -> None:
        try:
            req = json.loads(await bi.recv().read_to_end(MAX_FRAME))
            res = {"ok": True, "result": await self._handle(remote, req)}
        except Exception as e:
            res = {"ok": False, "error": str(e) or e.__class__.__name__}
        s = bi.send()
        await s.write_all(json.dumps(res).encode())
        await s.finish()

    async def _handle(self, remote: str, req: dict) -> dict:
        op = req.get("op")
        if op == "join":
            return self._handle_join(remote, req)
        peer = self.state.peer(remote)
        if not peer:
            raise LinkError("unknown or revoked peer")
        if op == "ping":
            return {"pong": True}
        need = {"a2a": "message", "ws_append": "workspace", "ws_read": "workspace"}.get(op)
        if need is None:
            raise LinkError(f"unknown op {op}")
        if need not in peer["scopes"]:
            raise LinkError(f"peer lacks the {need} scope")
        trusted = peer["role"] == "self"
        if op == "a2a":
            if req.get("method") != "SendMessage":
                raise LinkError("only SendMessage is supported over the link")
            msg = json_format.ParseDict(req["params"], t.SendMessageRequest()).message
            d = cards.decode(msg)
            who = {"from": peer["id"], "from_name": peer["name"], "trusted": trusted}
            if d["type"] == "card":
                if "request" not in peer["scopes"]:
                    raise LinkError("peer lacks the request scope")
                if trusted:
                    self._event({"type": "card", "state": "accepted", "card": d["card"], **who})
                    return {"status": "accepted", "card_id": d["card"]["id"]}
                self.state.set_approval(d["card"]["id"], {"card": d["card"], **who, "ts": time.time()})
                self._event({"type": "approval_needed", "card": d["card"], **who})
                return {"status": "waiting_for_approval", "card_id": d["card"]["id"]}
            self._event({**d, **who})
            return {"status": "delivered"}
        if op == "ws_append":
            return self.state.ws_append(req["ws"], peer["id"], peer["name"], trusted, str(req.get("kind", "note")),
                                        req.get("body"))
        return {"entries": self.state.ws_read(req["ws"], int(req.get("after", 0)))}

    def _handle_join(self, remote: str, req: dict) -> dict:
        payload, sig = self.read_invite(req["invite"])
        if payload["inviter"] != self.id:
            raise LinkError("invite was not issued by this node")
        try:
            self.state.secret.public().verify(_canon(payload), iroh.Signature.from_bytes(sig))
        except Exception:
            raise LinkError("invite signature is invalid")
        if payload["expires"] < time.time():
            raise LinkError("invite expired")
        if not self.state.use_nonce(payload["nonce"]):
            raise LinkError("invite already used")
        self.state.add_peer(remote, str(req.get("name") or "peer"), str(req["ticket"]), payload["role"], payload["scopes"])
        self._event({"type": "peer_joined", "from": remote, "from_name": req.get("name"), "role": payload["role"],
                     "trusted": payload["role"] == "self"})
        return {"name": self.state.settings["name"]}
