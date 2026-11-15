"""Model state as a message: pull per-layer tensors out of a transformers cache, pack them for the wire
(optionally only some attention layers, optionally int8), and rebuild a cache on the receiving side.

Two layer kinds:
  kv      attention layers: keys and values, grow with the sequence
  linear  linear-attention layers (Qwen3-Next Gated DeltaNet): conv state and recurrent state, fixed size.
          A recurrent state summarises the whole prefix and cannot be partly hidden, so it always travels.

Wire format (one blob): b"AFKV1" + u32 header length + JSON header + raw tensor bytes.
The header names the model, the sequence length, the kind of every layer, which layers are present and
every tensor's dtype, shape and byte range, so a receiver can refuse a blob made by a different model."""
from __future__ import annotations

import json
import struct

import torch
from transformers import DynamicCache

MAGIC = b"AFKV1"


def kind(layer) -> str:
    return "linear" if hasattr(layer, "recurrent_states") else "kv"


def _layers(cache):
    if hasattr(cache, "layers"):  # transformers >= 4.56
        return list(cache.layers)
    raise TypeError("this transformers version has no per-layer cache objects; need >= 4.56")


def layers_of(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(keys, values) of the attention layers only."""
    return [(l.keys, l.values) for l in _layers(cache) if kind(l) == "kv"]


def build(kvs: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    c = DynamicCache()
    for i, (k, v) in enumerate(kvs):
        c.update(k, v, layer_idx=i)
    return c


def seq_len(cache) -> int:
    return cache.get_seq_length()


def _q8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 per (batch, head, token) vector. Error is at most half a step of absmax/127."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(x / scale).to(torch.int8), scale.to(torch.float16)


def _tensors(layer) -> list[tuple[str, torch.Tensor]]:
    if kind(layer) == "kv":
        return [("k", layer.keys), ("v", layer.values)]
    out = [(f"conv:{s}", t) for s, t in layer.conv_states.items()]
    return out + [(f"rec:{s}", t) for s, t in layer.recurrent_states.items()]


def pack(cache, model_id: str, layers: list[int] | None = None, quant: str = "fp16", extra: dict | None = None) -> bytes:
    """layers: attention layers to include (None = all). Linear-attention layers are always included and
    never quantised (their recurrent state is small and accumulates error)."""
    ls = _layers(cache)
    kinds = [kind(l) for l in ls]
    keep = sorted(i for i in range(len(ls)) if kinds[i] == "linear" or layers is None or i in layers)
    header = {"model": model_id, "n_layers": len(ls), "kinds": kinds, "seq_len": seq_len(cache), "layers": keep,
              "quant": quant, "tensors": [], "extra": extra or {}}
    chunks, off = [], 0
    for i in keep:
        for name, t in _tensors(ls[i]):
            if quant == "int8" and kinds[i] == "kv":
                parts = list(zip(("data", "scale"), _q8(t.float())))
            else:
                parts = [("data", t.to(torch.float16) if kinds[i] == "kv" else t.float())]
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


def unpack(blob: bytes, model_id: str, device, dtype=torch.float32, config=None):
    """Rebuild a full-depth cache. Attention layers missing from the blob are zero-filled; the caller must mask
    those positions (see latent.attn), otherwise zeros act as real keys. Hybrid models need `config` so the
    cache gets the right layer types. Returns (cache, header)."""
    header, base = read_header(blob)
    if header["model"] != model_id:
        raise ValueError(f"KV blob is for {header['model']}, receiver runs {model_id}")
    kinds = header.get("kinds") or ["kv"] * header["n_layers"]
    got: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for t in header["tensors"]:
        raw = bytearray(blob[base + t["offset"]: base + t["offset"] + t["nbytes"]])
        x = torch.frombuffer(raw, dtype=torch.uint8).view(getattr(torch, t["dtype"])).reshape(t["shape"])
        got.setdefault(t["layer"], {}).setdefault(t["name"], {})[t["part"]] = x

    def tensor(i, name, dt):
        p = got[i][name]
        x = p["data"].float() * p["scale"].float() if "scale" in p else p["data"].float()
        return x.to(device=device, dtype=dt)

    if "linear" in kinds:
        if config is None:
            raise ValueError("hybrid model state needs the receiver's model config")
        c = DynamicCache(config=config)
    else:
        c = DynamicCache()
    kv_shape = next((tensor(i, "k", dtype).shape for i in got if kinds[i] == "kv"), None)
    for i, k in enumerate(kinds):
        if k == "kv":
            if i in got:
                kk, vv = tensor(i, "k", dtype), tensor(i, "v", dtype)
            else:
                kk = torch.zeros(kv_shape, device=device, dtype=dtype)
                vv = kk.clone()
            c.update(kk, vv, layer_idx=i)
        else:
            layer = c.layers[i]
            for name in got[i]:
                which, s = name.split(":")
                st = tensor(i, name, torch.float32 if which == "rec" else dtype)
                s = int(s)
                if which == "conv":
                    layer.conv_states[s] = st
                    layer.conv_kernel_size[s] = st.shape[-1]  # the conv state keeps exactly kernel-size steps
                    layer.is_conv_states_initialized[s] = True
                else:
                    layer.recurrent_states[s] = st
                    layer.is_recurrent_states_initialized[s] = True
                layer.has_previous_state[s] = True
    return c, header
