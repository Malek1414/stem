"""KV caches as messages: pull per-layer tensors out of a transformers cache, pack them for the wire
(optionally only some layers, optionally int8), and rebuild a cache on the receiving side.

Wire format (one blob): b"AFKV1" + u32 header length + JSON header + raw tensor bytes.
The header names the model, the sequence length, which layers are present and every tensor's
dtype, shape and byte range, so a receiver can refuse a blob made by a different model."""
from __future__ import annotations

import json
import struct

import torch
from transformers import DynamicCache

MAGIC = b"AFKV1"


def layers_of(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "layers"):  # transformers >= 4.56
        return [(l.keys, l.values) for l in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def build(kvs: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    c = DynamicCache()
    for i, (k, v) in enumerate(kvs):
        c.update(k, v, layer_idx=i)
    return c


def seq_len(cache) -> int:
    return layers_of(cache)[0][0].shape[-2]


def _q8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 per (batch, head, token) vector. Error is at most half a step of absmax/127."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(x / scale).to(torch.int8), scale.to(torch.float16)


def pack(cache, model_id: str, layers: list[int] | None = None, quant: str = "fp16", extra: dict | None = None) -> bytes:
    kvs = layers_of(cache)
    layers = sorted(range(len(kvs)) if layers is None else layers)
    header = {"model": model_id, "n_layers": len(kvs), "seq_len": kvs[0][0].shape[-2], "layers": layers,
              "quant": quant, "tensors": [], "extra": extra or {}}
    chunks, off = [], 0
    for i in layers:
        for name, t in (("k", kvs[i][0]), ("v", kvs[i][1])):
            parts = [("data", t.to(torch.float16))] if quant == "fp16" else list(zip(("data", "scale"), _q8(t.float())))
            for part, x in parts:
                b = x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                header["tensors"].append({"layer": i, "name": name, "part": part, "dtype": str(x.dtype).split(".")[-1],
                                          "shape": list(x.shape), "offset": off, "nbytes": len(b)})
                chunks.append(b)
                off += len(b)
    h = json.dumps(header).encode()
    return MAGIC + struct.pack(">I", len(h)) + h + b"".join(chunks)


def read_header(blob: bytes) -> tuple[dict, int]:
    if not blob.startswith(MAGIC):
        raise ValueError("not an agent-forge KV blob")
    n = struct.unpack(">I", blob[len(MAGIC):len(MAGIC) + 4])[0]
    start = len(MAGIC) + 4
    return json.loads(blob[start:start + n]), start + n


def unpack(blob: bytes, model_id: str, device, dtype=torch.float32, like: list[tuple[torch.Tensor, torch.Tensor]] | None = None):
    """Rebuild a full-depth cache. Layers missing from the blob are zero-filled; the caller must mask
    those positions (see latent.attn), otherwise zeros act as real keys. Returns (cache, header)."""
    header, base = read_header(blob)
    if header["model"] != model_id:
        raise ValueError(f"KV blob is for {header['model']}, receiver runs {model_id}")
    got: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for t in header["tensors"]:
        raw = bytearray(blob[base + t["offset"]: base + t["offset"] + t["nbytes"]])
        x = torch.frombuffer(raw, dtype=torch.uint8).view(getattr(torch, t["dtype"])).reshape(t["shape"])
        got.setdefault((t["layer"], t["name"]), {})[t["part"]] = x
    kvs, shape = [], None
    for i in range(header["n_layers"]):
        pair = []
        for name in ("k", "v"):
            p = got.get((i, name))
            if p is None:
                pair.append(None)
                continue
            x = p["data"].float() * p["scale"].float() if "scale" in p else p["data"].float()
            shape = shape or x.shape
            pair.append(x.to(device=device, dtype=dtype))
        kvs.append(pair)
    for i, pair in enumerate(kvs):
        if pair[0] is None:
            z = torch.zeros(shape, device=device, dtype=dtype)
            kvs[i] = (z, z.clone())
        else:
            kvs[i] = tuple(pair)
    return build(kvs), header
