import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from link.daemon import Daemon, call, control_path


@pytest.fixture
async def two_daemons():
    homes = [Path(tempfile.mkdtemp(prefix="afl-", dir="/tmp")) for _ in range(2)]  # short: unix socket path limit
    ds = [Daemon(h, bind="127.0.0.1:0", name=n) for h, n in zip(homes, ("malek", "friend"))]
    tasks = [asyncio.create_task(d.serve()) for d in ds]
    for h in homes:
        for _ in range(100):
            if control_path(h).exists():
                break
            await asyncio.sleep(0.05)
    yield homes
    for t in tasks:
        t.cancel()
    for d in ds:
        await d.node.close()
    for h in homes:
        shutil.rmtree(h, ignore_errors=True)


async def test_daemon_end_to_end(two_daemons):
    me, friend = two_daemons
    assert (await call({"cmd": "status"}, me))["name"] == "malek"
    invite = (await call({"cmd": "invite", "role": "peer"}, me))["invite"]
    assert (await call({"cmd": "join", "invite": invite}, friend))["name"] == "malek"

    waiter = asyncio.create_task(call({"cmd": "wait", "after": 1, "timeout": 10}, me))  # seq 1 = peer_joined
    await asyncio.sleep(0.2)
    await call({"cmd": "send", "peer": "malek", "text": "hello"}, friend)
    got = await waiter
    assert got[0]["type"] == "text" and got[0]["text"] == "hello" and got[0]["trusted"] is False

    card = {"goal": "review my PR", "done_criteria": {"description": "comments posted"}}
    res = await call({"cmd": "card", "peer": "malek", "card": card}, friend)
    assert res["status"] == "waiting_for_approval"
    assert res["card_id"] in await call({"cmd": "approvals"}, me)
    await call({"cmd": "approve", "card_id": res["card_id"]}, me)
    assert (await call({"cmd": "inbox"}, friend))[-1]["kind"] == "accepted"

    with pytest.raises(RuntimeError, match="unknown command"):
        await call({"cmd": "format_disk"}, me)


async def test_wait_times_out_empty(two_daemons):
    me, _ = two_daemons
    assert await call({"cmd": "wait", "after": 999, "timeout": 0.3}, me) == []


async def test_mcp_server_lists_link_tools():
    from link.mcp_server import make_server
    names = {t.name for t in await make_server().list_tools()}
    assert {"link_send", "link_send_card", "link_wait", "link_approve", "link_workspace_read"} <= names
