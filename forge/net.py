"""HTTP to Ollama through the NVIDIA Sync tunnel, surviving tunnel drops.

The Sync tunnel restarts on its own when the VPN path flaps (seen 2026-11-15: benchmark C died on
"Connection refused" while Sync showed RESTARTING). Connection-level failures are retried with backoff for
up to `wait_total` seconds. HTTP errors and read timeouts are not retried: they are real answers."""
from __future__ import annotations

import http.client
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


NVSYNC = "/Applications/NVIDIA Sync.app/Contents/Resources/bin/nvsync-arm64"
SYNC_ALIAS = "malekhassan"
SYNC_OLLAMA = "sync://ollama"  # resolved on every attempt: Sync may move the local port when it reconnects


def sync_port(remote_port: int = 11434) -> int:
    """Local port NVIDIA Sync currently forwards to `remote_port` on the Spark (falls back to the same number)."""
    try:
        out = subprocess.run([NVSYNC, "status", SYNC_ALIAS], capture_output=True, text=True, timeout=20).stdout
        return int(json.loads(out)["ports"][str(remote_port)]["local_port"]) or remote_port
    except Exception:
        return remote_port


def _resolve(req):
    url = req.full_url if isinstance(req, urllib.request.Request) else req
    if not url.startswith(SYNC_OLLAMA):
        return req
    url = f"http://localhost:{sync_port()}" + url[len(SYNC_OLLAMA):]
    if isinstance(req, urllib.request.Request):
        return urllib.request.Request(url, data=req.data, headers=dict(req.header_items()), method=req.get_method())
    return url


def _dropped(e: Exception) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return False
    if isinstance(e, urllib.error.URLError):
        return isinstance(e.reason, (ConnectionError, OSError)) and not isinstance(e.reason, TimeoutError)
    return isinstance(e, (ConnectionError, http.client.RemoteDisconnected, http.client.IncompleteRead))


def urlopen(req, timeout: float, wait_total: float = 900):
    start, delay = time.monotonic(), 5.0
    while True:
        try:
            return urllib.request.urlopen(_resolve(req), timeout=timeout)
        except Exception as e:
            if not _dropped(e) or time.monotonic() - start >= wait_total:
                raise
            print(f"[net] tunnel down ({e}); retrying in {delay:.0f}s", file=sys.stderr, flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
