import base64
import json
import struct
import zlib

from forge import bench_d


def png(w, h):
    raw = b"".join(b"\x00" + b"\x00" * (w * 3) for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def jpeg_header(w, h):
    return b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9 + \
        b"\xff\xc0" + struct.pack(">HBHH", 17, 8, h, w) + b"\x00" * 10


def test_image_size():
    assert bench_d.image_size(png(40, 30)) == (40, 30)
    assert bench_d.image_size(jpeg_header(1280, 800)) == (1280, 800)


def session(click_name, click_input, with_image=True):
    img = base64.b64encode(png(400, 300)).decode()
    lines = [
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "1", "content":
            ([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}}] if with_image else
             [{"type": "text", "text": "ok"}])}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "I'll click the blue Send button."},
                                                      {"type": "tool_use", "name": click_name, "input": click_input}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": click_name, "input": click_input}]}},
    ]
    return [json.dumps(l) for l in lines]


def test_items_take_first_click_after_a_screenshot():
    items = bench_d.items_from_session(session("mcp__computer-use__left_click", {"coordinate": [120, 80]}), "s")
    assert len(items) == 1  # the second click has no fresh screenshot
    it = items[0]
    assert it["target"] == [120, 80] and it["width"] == 400 and "Send button" in it["intent"]
    batch = {"actions": [{"action": "left_click", "coordinate": [10, 20]}, {"action": "left_click", "coordinate": [30, 40]}]}
    assert bench_d.items_from_session(session("mcp__computer-use__computer_batch", batch), "s")[0]["target"] == [10, 20]
    assert bench_d.items_from_session(session("mcp__computer-use__left_click", {"coordinate": [9999, 1]}), "s") == []
    assert bench_d.items_from_session(session("mcp__computer-use__left_click", {"coordinate": [1, 1]}, False), "s") == []


def test_parse_and_score():
    it = {"target": [100, 100], "width": 400}
    assert bench_d.parse_point('{"x": 110, "y": 95}') == (110, 95)
    assert bench_d.parse_point("click at (12, 34)") == (12, 34)
    assert bench_d.parse_point("no idea") is None
    assert bench_d.score(it, (120, 100))["pass"] and not bench_d.score(it, (140, 100))["pass"]
    assert bench_d.score(it, None) == {"pass": False, "distance": None}
