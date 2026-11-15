import asyncio, json, time, uuid
import iroh
from google.protobuf import json_format
from a2a import types as t

ALPN = b"agent-forge/a2a/1"

async def endpoint():
    b = iroh.EndpointBuilder()
    b.apply_minimal()                      # no n0 discovery, no relays: nothing leaves this machine
    b.relay_mode(iroh.RelayMode.disabled())
    b.alpns([ALPN])
    b.bind_addr("127.0.0.1:0")
    return await b.bind()

def a2a_request(text):
    msg = t.Message(message_id=str(uuid.uuid4()), role=t.Role.ROLE_USER, parts=[t.Part(text=text)])
    params = json_format.MessageToDict(t.SendMessageRequest(message=msg))
    return {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": params}

async def serve(ep, ready):
    ready.set()
    inc = await ep.accept_next()
    conn = await (await inc.accept()).connect()
    bi = await conn.accept_bi()
    req = json.loads(await bi.recv().read_to_end(1 << 20))
    got = json_format.ParseDict(req["params"], t.SendMessageRequest())     # validates against the A2A schema
    reply = t.Message(message_id=str(uuid.uuid4()), role=t.Role.ROLE_AGENT,
                      parts=[t.Part(text="ack: " + got.message.parts[0].text)])
    out = {"jsonrpc": "2.0", "id": req["id"], "result": {"message": json_format.MessageToDict(reply)}}
    s = bi.send(); await s.write_all(json.dumps(out).encode()); await s.finish()
    await asyncio.sleep(0.5)

async def main():
    a, b = await endpoint(), await endpoint()
    ready = asyncio.Event(); task = asyncio.create_task(serve(b, ready)); await ready.wait()
    t0 = time.perf_counter()
    conn = await a.connect(b.addr(), ALPN)
    t1 = time.perf_counter()
    bi = await conn.open_bi()
    s = bi.send(); await s.write_all(json.dumps(a2a_request("task card: set x to 2")).encode()); await s.finish()
    resp = json.loads(await bi.recv().read_to_end(1 << 20))
    t2 = time.perf_counter()
    rtts = []
    for i in range(20):  # 20 more round trips on fresh streams of the same connection
        x = time.perf_counter(); bi2 = await conn.open_bi(); s2 = bi2.send(); await s2.write_all(b"p"); await s2.finish()
        rtts.append(time.perf_counter() - x)
    print(json.dumps({"connect_ms": round((t1-t0)*1e3,2), "first_request_reply_ms": round((t2-t1)*1e3,2),
                      "reply_text": resp["result"]["message"]["parts"][0]["text"], "remote_id": str(conn.remote_id())[:16],
                      "quic_rtt": conn.rtt()}, indent=1))
    await task; await a.close(); await b.close()

asyncio.run(main())
