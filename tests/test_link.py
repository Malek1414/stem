import asyncio

import pytest

from link import cards
from link.node import LinkError, Node
from link.state import State


async def make(tmp_path, name):
    n = Node(State(tmp_path / name, name=name), bind="127.0.0.1:0")
    await n.start()
    return n


@pytest.fixture
async def trio(tmp_path):
    me, friend, coder = [await make(tmp_path, n) for n in ("malek", "friend", "coder")]
    await friend.join(me.make_invite("peer"))
    await coder.join(me.make_invite("self"))
    yield me, friend, coder
    for n in (me, friend, coder):
        await n.close()


def card(goal="set x to 2"):
    return cards.TaskCard(goal=goal, done_criteria={"description": "x == 2", "check_command": "pytest -q"})


async def test_text_from_a_friend_is_delivered_untrusted(trio):
    me, friend, _ = trio
    assert (await friend.send_text("malek", "hi from friend"))["status"] == "delivered"
    last = me.state.inbox()[-1]
    assert last["type"] == "text" and last["text"] == "hi from friend"
    assert last["from_name"] == "friend" and last["trusted"] is False


async def test_friends_card_waits_for_approval_then_progress_flows_back(trio):
    me, friend, _ = trio
    c = card()
    assert (await friend.send_card("malek", c))["status"] == "waiting_for_approval"
    assert me.state.inbox()[-1]["type"] == "approval_needed"
    assert c.id in me.state.approvals()
    await me.approve(c.id)
    assert me.state.inbox()[-1]["state"] == "accepted" and c.id not in me.state.approvals()
    back = friend.state.inbox()[-1]
    assert back["type"] == "progress" and back["kind"] == "accepted" and back["card_id"] == c.id


async def test_deny_sends_rejected(trio):
    me, friend, _ = trio
    c = card()
    await friend.send_card("malek", c)
    await me.deny(c.id, "not now")
    assert friend.state.inbox()[-1]["kind"] == "rejected"


async def test_own_machine_card_is_accepted_at_once(trio):
    me, _, coder = trio
    c = card()
    assert (await coder.send_card("malek", c))["status"] == "accepted"
    last = me.state.inbox()[-1]
    assert last["type"] == "card" and last["trusted"] is True and last["card"]["goal"] == "set x to 2"


async def test_progress_from_coder(trio):
    me, _, coder = trio
    await coder.send_progress("malek", "abc", "progress", "x" * 1000)
    last = me.state.inbox()[-1]
    assert last["kind"] == "progress" and len(last["text"]) == cards.MAX_PROGRESS_TEXT


async def test_invite_is_single_use_and_tamper_proof(tmp_path, trio):
    me, _, _ = trio
    other = await make(tmp_path, "other")
    try:
        inv = me.make_invite("peer")
        await other.join(inv)
        with pytest.raises(LinkError, match="already used"):
            await other.join(inv)
        payload, sig = Node.read_invite(me.make_invite("peer"))
        payload["role"] = "self"  # try to promote yourself to an owner machine
        import base64, json
        forged = "afl1." + base64.urlsafe_b64encode(json.dumps(
            {"p": payload, "s": base64.urlsafe_b64encode(sig).decode().rstrip("=")}).encode()).decode().rstrip("=")
        with pytest.raises(LinkError, match="signature"):
            await other.join(forged)
        with pytest.raises(LinkError, match="expired"):
            await other.join(me.make_invite("peer", ttl_hours=-1))
    finally:
        await other.close()


async def test_revoked_peer_is_refused(trio):
    me, friend, _ = trio
    assert me.state.revoke("friend")
    with pytest.raises(LinkError, match="unknown or revoked"):
        await friend.send_text("malek", "still there?")


async def test_scopes_are_enforced(tmp_path, trio):
    me, _, _ = trio
    limited = await make(tmp_path, "limited")
    try:
        await limited.join(me.make_invite("peer", scopes=["message"]))
        await limited.send_text("malek", "ok")
        with pytest.raises(LinkError, match="request scope"):
            await limited.send_card("malek", card())
        with pytest.raises(LinkError, match="workspace scope"):
            await limited.ws_append("hack", "note", "x", host="malek")
    finally:
        await limited.close()


async def test_workspace_marks_other_peoples_entries_untrusted(trio):
    me, friend, _ = trio
    await me.ws_append("hack", "note", "plan: ship the demo")
    await friend.ws_append("hack", "finding", "API key rotates hourly", host="malek")
    entries = await friend.ws_read("hack", host="malek")
    assert [e["author_name"] for e in entries] == ["malek", "friend"]
    assert [e["trusted"] for e in entries] == [True, False]
    assert [e["seq"] for e in await me.ws_read("hack", after=1)] == [2]


def test_card_validation():
    with pytest.raises(ValueError):
        cards.TaskCard(goal="", done_criteria={"description": "x"}).validate()
    with pytest.raises(ValueError):
        cards.TaskCard.from_dict({"goal": "g", "done_criteria": {}})
    with pytest.raises(ValueError):
        cards.progress_message("id", "teleported")


def test_name_persists_across_restarts(tmp_path):
    State(tmp_path / "n", name="malek")
    assert State(tmp_path / "n").settings["name"] == "malek"


async def test_ping_and_paths(trio):
    me, _, coder = trio
    r = await coder.ping("malek", n=5)
    assert r["n"] == 5 and r["median_ms"] >= 0 and r["p95_ms"] >= r["median_ms"]
    paths = coder.paths("malek")
    assert paths and any(p["selected"] for p in paths) and paths[0]["kind"] in ("direct", "relay")
