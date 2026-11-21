"""Talk to the DGX Spark through the NVIDIA Sync JupyterLab tunnel (localhost:11012), never plain SSH.

  .venv/bin/python work/spark.py sh "<bash command>" [timeout_s]   run a command on the Spark, print its output
  .venv/bin/python work/spark.py put <local file> <remote path>     upload (chunked, sha256 checked); remote is relative to ~

Token: work/.jupyter_token (chmod 600, git-ignored; read from the JupyterLab page config on 22.11).
One call = one action; callers must not loop this (the Spark is shared).
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import sys
import uuid
from pathlib import Path

import httpx
import websocket

BASE = "localhost:11012"
TOKEN = (Path(__file__).parent / ".jupyter_token").read_text().strip()
H = {"Authorization": f"token {TOKEN}"}
CHUNK = 600_000  # base64 chars per message; a 3 MB websocket message was dropped on 21.11


def kernel() -> str:
    ks = httpx.get(f"http://{BASE}/api/kernels", headers=H, timeout=15).json()
    k = next((k["id"] for k in ks if k["name"] == "python3"), None)
    return k or httpx.post(f"http://{BASE}/api/kernels", headers=H, json={"name": "python3"}, timeout=30).json()["id"]


def run_py(code: str, timeout: int = 120) -> str:
    ws = websocket.create_connection(f"ws://{BASE}/api/kernels/{kernel()}/channels", header=[f"Authorization: token {TOKEN}"],
                                     timeout=timeout)
    mid = uuid.uuid4().hex
    ws.send(json.dumps({"header": {"msg_id": mid, "username": "af", "session": uuid.uuid4().hex, "msg_type": "execute_request",
                                   "version": "5.3"}, "parent_header": {}, "metadata": {}, "channel": "shell", "buffers": [],
                        "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {},
                                    "allow_stdin": False, "stop_on_error": True}}))
    out = ""
    try:
        while True:
            m = json.loads(ws.recv())
            if m.get("parent_header", {}).get("msg_id") != mid:
                continue
            t = m["header"]["msg_type"]
            if t == "stream":
                out += m["content"]["text"]
            elif t == "error":
                out += f"\nERROR {m['content']['ename']}: {m['content']['evalue']}"
            elif t == "execute_result":
                out += m["content"]["data"].get("text/plain", "") + "\n"
            elif t == "status" and m["content"]["execution_state"] == "idle":
                break
    except websocket.WebSocketTimeoutException:
        out += "\n[timeout]"
    finally:
        ws.close()
    return out


def sh(cmd: str, timeout: int = 120) -> str:
    code = ("import subprocess, os\nos.environ.pop('CODER27_ADAPTER', None)\n"
            f"r = subprocess.run({cmd!r}, shell=True, executable='/bin/bash', capture_output=True, text=True, timeout={timeout})\n"
            "print(r.stdout[-6000:], r.stderr[-2000:])")
    return run_py(code, timeout + 30)


def put(local: str, remote: str) -> str:
    data = Path(local).read_bytes()
    want = hashlib.sha256(data).hexdigest()
    b64 = base64.b64encode(gzip.compress(data)).decode()
    tmp = f"/tmp/af_put_{uuid.uuid4().hex}.b64"
    for i in range(0, len(b64), CHUNK):
        r = run_py(f"open({tmp!r}, 'a').write({b64[i:i + CHUNK]!r}); print('ok')", 120)
        if "ok" not in r:
            raise SystemExit(f"chunk {i // CHUNK} failed: {r[-300:]}")
    r = run_py(f"""import base64, gzip, hashlib, os
p = os.path.expanduser('~/' + {remote!r}); os.makedirs(os.path.dirname(p), exist_ok=True)
d = gzip.decompress(base64.b64decode(open({tmp!r}).read())); os.remove({tmp!r})
h = hashlib.sha256(d).hexdigest()
if h == {want!r}:
    open(p + '.part', 'wb').write(d); os.replace(p + '.part', p)
print(h, len(d), sum(1 for _ in open(p)) if h == {want!r} else 'MISMATCH, not written')""", 300)
    if want not in r:
        raise SystemExit(f"sha256 mismatch: {r[-300:]}")
    return r.strip()


if __name__ == "__main__":
    if sys.argv[1] == "sh":
        print(sh(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 120))
    elif sys.argv[1] == "py":
        print(run_py(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 120))
    elif sys.argv[1] == "put":
        print(put(sys.argv[2], sys.argv[3]))
