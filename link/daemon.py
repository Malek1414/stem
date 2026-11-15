"""The link daemon: runs one Node and serves a local control socket for the CLI and the MCP server.

The control socket is a Unix socket in the node's home folder (mode 0600, owner only). One JSON request
per line, one JSON response per line."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from . import cards
from .node import Node
from .state import DEFAULT_HOME, State


def control_path(home: Path = DEFAULT_HOME) -> Path:
    return Path(home) / "control.sock"


class Daemon:
    def __init__(self, home: Path = DEFAULT_HOME, bind: str = "0.0.0.0:0", name: str | None = None):
        self.home = Path(home)
        self.changed = asyncio.Condition()
        self.node = Node(State(self.home, name=name), bind=bind, on_event=self._notify)

    def _notify(self, entry: dict) -> None:
        async def wake():
            async with self.changed:
                self.changed.notify_all()
        asyncio.get_event_loop().create_task(wake())

    async def wait(self, after: int, timeout: float) -> list[dict]:
        """Long-poll: return inbox entries after `after`, waiting up to `timeout` seconds for one."""
        new = self.node.state.inbox(after)
        if new or timeout <= 0:
            return new
        try:
            async with self.changed:
                await asyncio.wait_for(self.changed.wait_for(lambda: bool(self.node.state.inbox(after))), timeout)
        except asyncio.TimeoutError:
            pass
        return self.node.state.inbox(after)

    async def handle(self, req: dict):
        n, cmd = self.node, req.get("cmd")
        if cmd == "status":
            s = n.state.settings
            return {"name": s["name"], "id": n.id, "mode": s["mode"], "ticket": n.ticket(),
                    "peers": len([p for p in n.state.peers().values() if not p["revoked"]]),
                    "waiting_for_approval": len(n.state.approvals())}
        if cmd == "ping":
            return await n.ping(req["peer"], int(req.get("n", 10)))
        if cmd == "send_kv":  # blobs go by file path, the control socket carries one JSON line
            return await n.send_kv(req["peer"], Path(req["path"]).read_bytes(), req.get("meta", {}))
        if cmd == "paths":
            return n.paths(req["peer"])
        if cmd == "invite":
            return {"invite": n.make_invite(req.get("role", "peer"), req.get("scopes"), float(req.get("ttl_hours", 72)))}
        if cmd == "join":
            return await n.join(req["invite"])
        if cmd == "peers":
            return list(n.state.peers().values())
        if cmd == "revoke":
            return {"revoked": n.state.revoke(req["peer"])}
        if cmd == "send":
            return await n.send_text(req["peer"], req["text"], req.get("context_id", ""))
        if cmd == "card":
            return await n.send_card(req["peer"], cards.TaskCard.from_dict(req["card"]))
        if cmd == "progress":
            return await n.send_progress(req["peer"], req["card_id"], req["kind"], req.get("text", ""), req.get("data"))
        if cmd == "inbox":
            return n.state.inbox(int(req.get("after", 0)))
        if cmd == "wait":
            return await self.wait(int(req.get("after", 0)), float(req.get("timeout", 60)))
        if cmd == "approvals":
            return n.state.approvals()
        if cmd == "approve":
            return await n.approve(req["card_id"])
        if cmd == "deny":
            await n.deny(req["card_id"], req.get("reason", ""))
            return {"denied": req["card_id"]}
        if cmd == "ws_append":
            return await n.ws_append(req["ws"], req.get("kind", "note"), req.get("body"), req.get("host"))
        if cmd == "ws_read":
            return await n.ws_read(req["ws"], int(req.get("after", 0)), req.get("host"))
        raise ValueError(f"unknown command {cmd}")

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                try:
                    res = {"ok": True, "result": await self.handle(json.loads(line))}
                except Exception as e:
                    res = {"ok": False, "error": str(e) or e.__class__.__name__}
                writer.write((json.dumps(res, ensure_ascii=False) + "\n").encode())
                await writer.drain()
        finally:
            writer.close()

    async def serve(self) -> None:
        await self.node.start()
        sock = control_path(self.home)
        sock.unlink(missing_ok=True)
        server = await asyncio.start_unix_server(self._client, path=str(sock), limit=16 << 20)
        os.chmod(sock, 0o600)
        print(json.dumps({"listening": str(sock), "id": self.node.id, "ticket": self.node.ticket()}), flush=True)
        async with server:
            await server.serve_forever()


async def call(req: dict, home: Path = DEFAULT_HOME, timeout: float = 120):
    """Send one control request to a running daemon."""
    reader, writer = await asyncio.open_unix_connection(str(control_path(home)), limit=16 << 20)
    try:
        writer.write((json.dumps(req) + "\n").encode())
        await writer.drain()
        res = json.loads(await asyncio.wait_for(reader.readline(), timeout))
    finally:
        writer.close()
    if not res["ok"]:
        raise RuntimeError(res["error"])
    return res["result"]
