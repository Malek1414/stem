import io
import urllib.error

import pytest

from forge import net


def test_retries_a_dropped_tunnel_then_succeeds(monkeypatch):
    calls = []

    def fake(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        return io.BytesIO(b"ok")

    monkeypatch.setattr(net.urllib.request, "urlopen", fake)
    monkeypatch.setattr(net.time, "sleep", lambda s: None)
    with net.urlopen("http://x/", timeout=5) as r:
        assert r.read() == b"ok"
    assert len(calls) == 3


def test_gives_up_after_the_budget(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(net.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(net.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))

    def down(req, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    monkeypatch.setattr(net.urllib.request, "urlopen", down)
    with pytest.raises(urllib.error.URLError):
        net.urlopen("http://x/", timeout=5, wait_total=120)
    assert clock[0] >= 120


def test_http_errors_are_not_retried(monkeypatch):
    calls = []

    def bad(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError("u", 500, "boom", {}, None)

    monkeypatch.setattr(net.urllib.request, "urlopen", bad)
    with pytest.raises(urllib.error.HTTPError):
        net.urlopen("http://x/", timeout=5)
    assert len(calls) == 1


def test_sync_base_follows_the_current_tunnel_port(monkeypatch):
    seen = []
    monkeypatch.setattr(net, "sync_port", lambda remote_port=11434: 65171)

    def fake(req, timeout):
        seen.append(req.full_url if hasattr(req, "full_url") else req)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(net.urllib.request, "urlopen", fake)
    net.urlopen(net.urllib.request.Request(net.SYNC_OLLAMA + "/api/chat", data=b"{}"), timeout=5)
    net.urlopen(net.SYNC_OLLAMA + "/api/tags", timeout=5)
    assert seen == ["http://localhost:65171/api/chat", "http://localhost:65171/api/tags"]
