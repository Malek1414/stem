"""python -m link <command>

run [--name N] [--bind ADDR] [--mode p2p|relay] [--relay-url URL]   start the daemon
mcp                                                                 MCP server for Claude Code (stdio)
status | peers | approvals | inbox [--after N]
invite [--role peer|self] [--ttl-hours H]                           single-use invite
join <invite> | revoke <peer> | approve <card_id> | deny <card_id> [--reason R]
send <peer> <text>                                                  free-text A2A message
Environment: STEM_LINK_HOME (default ~/.stem/link)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .state import DEFAULT_HOME, State


def main(argv=None) -> int:
    home = Path(os.environ.get("STEM_LINK_HOME", DEFAULT_HOME))
    ap = argparse.ArgumentParser(prog="link")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name")
    r.add_argument("--bind", default="0.0.0.0:0")
    r.add_argument("--mode", choices=["p2p", "relay"])
    r.add_argument("--relay-url")
    sub.add_parser("mcp")
    for c in ("status", "peers", "approvals"):
        sub.add_parser(c)
    i = sub.add_parser("inbox")
    i.add_argument("--after", type=int, default=0)
    iv = sub.add_parser("invite")
    iv.add_argument("--role", default="peer", choices=["peer", "self"])
    iv.add_argument("--ttl-hours", type=float, default=72)
    for c in ("join", "revoke", "approve"):
        sub.add_parser(c).add_argument("arg")
    d = sub.add_parser("deny")
    d.add_argument("arg")
    d.add_argument("--reason", default="")
    s = sub.add_parser("send")
    s.add_argument("peer")
    s.add_argument("text")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        st = State(home, name=a.name)
        if a.mode or a.relay_url:
            st.settings.update({k: v for k, v in (("mode", a.mode), ("relay_url", a.relay_url)) if v})
            st._save("settings.json", st.settings)
        from .daemon import Daemon
        asyncio.run(Daemon(home, a.bind).serve())
        return 0
    if a.cmd == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()
        return 0

    from .daemon import call
    req = {"status": {"cmd": "status"}, "peers": {"cmd": "peers"}, "approvals": {"cmd": "approvals"},
           "inbox": {"cmd": "inbox", "after": getattr(a, "after", 0)},
           "invite": {"cmd": "invite", "role": getattr(a, "role", "peer"), "ttl_hours": getattr(a, "ttl_hours", 72)},
           "join": {"cmd": "join", "invite": getattr(a, "arg", "")},
           "revoke": {"cmd": "revoke", "peer": getattr(a, "arg", "")},
           "approve": {"cmd": "approve", "card_id": getattr(a, "arg", "")},
           "deny": {"cmd": "deny", "card_id": getattr(a, "arg", ""), "reason": getattr(a, "reason", "")},
           "send": {"cmd": "send", "peer": getattr(a, "peer", ""), "text": getattr(a, "text", "")}}[a.cmd]
    try:
        print(json.dumps(asyncio.run(call(req, home)), indent=1, ensure_ascii=False))
    except (FileNotFoundError, ConnectionRefusedError):
        print("the link daemon is not running: start it with `python -m link run`", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
