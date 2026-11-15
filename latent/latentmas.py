"""LatentMAS latent thoughts (Zou et al. 2025, arXiv 2511.20639), training-free.

Instead of decoding tokens, the sender feeds its own last-layer hidden state back in as the next input
embedding for m steps. The hidden state lives in output space, so it is first mapped into input-embedding
space with a ridge-regression alignment W_a = (W_out^T W_out + lam I)^-1 W_out^T W_in, then rescaled to the
mean input-embedding norm. The KV cache those steps leave behind is the working memory handed on."""
from __future__ import annotations

import torch


@torch.no_grad()
def alignment(model, lam: float = 1e-2) -> tuple[torch.Tensor, float]:
    w_in = model.get_input_embeddings().weight.float().cpu()
    w_out = model.get_output_embeddings().weight.float().cpu()
    d = w_in.shape[1]
    w_a = torch.linalg.solve(w_out.T @ w_out + lam * torch.eye(d), w_out.T @ w_in)
    return w_a, w_in.norm(dim=-1).mean().item()


@torch.no_grad()
def think(model, cache, last_hidden: torch.Tensor, steps: int, w_a: torch.Tensor, target_norm: float):
    """Run `steps` latent steps after the cached prefix. last_hidden: [1, 1, d] from the prefill.
    Returns the grown cache and the final hidden state."""
    w = w_a.to(device=last_hidden.device, dtype=torch.float32)
    h = last_hidden
    for _ in range(steps):
        e = h.float() @ w
        e = e * (target_norm / e.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        out = model(inputs_embeds=e.to(model.dtype), past_key_values=cache, use_cache=True, output_hidden_states=True)
        cache, h = out.past_key_values, out.hidden_states[-1][:, -1:]
    return cache, h
