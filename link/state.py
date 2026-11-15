"""On-disk state of one link node: its key, peers, inbox, approvals and workspaces.

Identity is the node's iroh secret key (Ed25519). Every iroh connection proves which endpoint id is on
the other end, so a peer is simply an endpoint id we have accepted, with scopes and a revoked flag."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import iroh

DEFAULT_HOME = Path.home() / ".agent-forge" / "link"
SCOPES = ("message", "request", "workspace")  # request = send task cards that run on this machine


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


class State:
    def __init__(self, home: Path = DEFAULT_HOME, name: str | None = None):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        os.chmod(self.home, 0o700)
        (self.home / "workspaces").mkdir(exist_ok=True)
        key = self.home / "secret.key"
        if not key.exists():
            key.write_bytes(iroh.SecretKey.generate().to_bytes())
            os.chmod(key, 0o600)
        self.secret = iroh.SecretKey.from_bytes(key.read_bytes())
        fresh = not (self.home / "settings.json").exists()
        self.settings = self._load("settings.json", {"name": name or "me", "mode": "p2p", "relay_url": None})
        if fresh or (name and self.settings.get("name") != name):
            self.settings["name"] = name or self.settings["name"]
            _write(self.home / "settings.json", self.settings)

    # --- small JSON stores -------------------------------------------------------------------
    def _load(self, fname: str, default):
        p = self.home / fname
        return json.loads(p.read_text()) if p.exists() else default

    def _save(self, fname: str, obj) -> None:
        _write(self.home / fname, obj)

    @property
    def id(self) -> str:
        return str(self.secret.public())

    # --- peers -------------------------------------------------------------------------------
    def peers(self) -> dict[str, dict]:
        return self._load("peers.json", {})

    def peer(self, endpoint_id: str) -> dict | None:
        p = self.peers().get(endpoint_id)
        return None if (p is None or p.get("revoked")) else p

    def peer_by_name(self, name: str) -> dict | None:
        hits = [p for p in self.peers().values() if p["name"] == name and not p.get("revoked")]
        return hits[0] if len(hits) == 1 else None

    def add_peer(self, endpoint_id: str, name: str, ticket: str, role: str, scopes: list[str]) -> dict:
        bad = set(scopes) - set(SCOPES)
        if bad:
            raise ValueError(f"unknown scopes {bad}")
        peers = self.peers()
        peers[endpoint_id] = {"id": endpoint_id, "name": name, "ticket": ticket, "role": role,
                              "scopes": sorted(scopes), "revoked": False, "added": time.time()}
        self._save("peers.json", peers)
        return peers[endpoint_id]

    def revoke(self, name_or_id: str) -> bool:
        peers = self.peers()
        for p in peers.values():
            if name_or_id in (p["id"], p["name"]):
                p["revoked"] = True
                self._save("peers.json", peers)
                return True
        return False

    # --- invites (single use) ------------------------------------------------------------------
    def use_nonce(self, nonce: str) -> bool:
        used = self._load("used_nonces.json", [])
        if nonce in used:
            return False
        used.append(nonce)
        self._save("used_nonces.json", used)
        return True

    # --- inbox and approvals -------------------------------------------------------------------
    def inbox_add(self, entry: dict) -> dict:
        entry = {"seq": self._next_seq(), "ts": time.time(), **entry}
        with open(self.home / "inbox.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def inbox(self, after: int = 0) -> list[dict]:
        p = self.home / "inbox.jsonl"
        if not p.exists():
            return []
        return [e for e in (json.loads(x) for x in p.read_text().splitlines() if x) if e["seq"] > after]

    def _next_seq(self) -> int:
        n = self._load("seq.json", 0) + 1
        self._save("seq.json", n)
        return n

    def approvals(self) -> dict[str, dict]:
        return self._load("approvals.json", {})

    def set_approval(self, key: str, entry: dict | None) -> None:
        a = self.approvals()
        if entry is None:
            a.pop(key, None)
        else:
            a[key] = entry
        self._save("approvals.json", a)

    # --- latent KV blobs from your own machines ----------------------------------------------
    def latent_store(self, blob: bytes) -> str:
        d = self.home / "latent"
        d.mkdir(exist_ok=True)
        bid = hashlib.sha256(blob).hexdigest()[:24]
        (d / f"{bid}.kv").write_bytes(blob)
        return bid

    def latent_blob(self, bid: str) -> bytes:
        if not bid.isalnum():
            raise ValueError("bad blob id")
        return (self.home / "latent" / f"{bid}.kv").read_bytes()

    # --- workspaces: append-only logs hosted by their creator ----------------------------------
    def ws_path(self, ws: str) -> Path:
        if not ws.replace("-", "").replace("_", "").isalnum():
            raise ValueError("workspace names are letters, digits, - and _")
        return self.home / "workspaces" / f"{ws}.jsonl"

    def ws_append(self, ws: str, author: str, author_name: str, trusted: bool, kind: str, body) -> dict:
        p = self.ws_path(ws)
        n = sum(1 for _ in open(p)) if p.exists() else 0
        entry = {"ws": ws, "seq": n + 1, "ts": time.time(), "author": author, "author_name": author_name,
                 "trusted": trusted, "kind": kind, "body": body}
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def ws_read(self, ws: str, after: int = 0) -> list[dict]:
        p = self.ws_path(ws)
        if not p.exists():
            return []
        return [e for e in (json.loads(x) for x in p.read_text().splitlines() if x) if e["seq"] > after]
