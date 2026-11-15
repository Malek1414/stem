"""Per-layer control over what the receiver attends to, for KVComm-style selective layer sharing.

transformers uses one attention mask for every layer. KVComm needs the sender's positions to be visible
only in the selected layers, so this registers an attention function ("af_masked") that rebuilds the mask
per layer from module.layer_idx. It can also record, per layer, how much attention the receiver's
queries put on the sender's positions; that is the importance score used to pick layers."""
from __future__ import annotations

import contextlib

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

NAME = "af_masked"
_S = {"hide": 0, "selected": None, "record": None}


def _causal(q_len: int, k_len: int, device) -> torch.Tensor:
    i = torch.arange(q_len, device=device)[:, None]
    j = torch.arange(k_len, device=device)[None, :]
    return j <= (k_len - q_len + i)


def masked_sdpa(module, query, key, value, attention_mask, **kw):
    hide, sel, rec = _S["hide"], _S["selected"], _S["record"]
    li = getattr(module, "layer_idx", None)
    if hide and (rec is not None or (sel is not None and li not in sel)):
        q_len, k_len = query.shape[-2], key.shape[-2]
        allowed = _causal(q_len, k_len, query.device)
        if rec is not None:  # attention mass from the receiver's queries onto the sender's positions
            k = repeat_kv(key, query.shape[1] // key.shape[1])
            s = (query.float() @ k.float().transpose(-1, -2)) * kw.get("scaling", query.shape[-1] ** -0.5)
            p = s.masked_fill(~allowed, float("-inf")).softmax(-1)
            rec[li] = rec.get(li, 0.0) + p[..., :hide].sum(-1).mean().item()
        if sel is not None and li not in sel:
            allowed = allowed & (torch.arange(k_len, device=query.device)[None, :] >= hide)
            attention_mask = allowed[None, None]
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


AttentionInterface.register(NAME, masked_sdpa)
try:  # newer transformers build masks per attention implementation
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
    ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, sdpa_mask)
except ImportError:
    pass


@contextlib.contextmanager
def hide_sender(n_positions: int, selected_layers: set[int] | None = None, record: dict | None = None):
    """Inside this block, layers not in selected_layers cannot see the first n_positions keys."""
    old = dict(_S)
    _S.update(hide=n_positions, selected=selected_layers, record=record)
    try:
        yield
    finally:
        _S.update(old)
