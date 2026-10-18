import math
import torch
from torch import Tensor


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                                 mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """q [B,H,Q,D], k [B,H,K,D], v [B,H,K,Dv]; True means allowed."""
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is None:
        weights = scores.softmax(-1)
    else:
        allowed = torch.broadcast_to(mask, scores.shape)
        scores = scores.masked_fill(~allowed, -torch.inf)
        # A padding-only query has zero output, including finite zero gradients.
        scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = scores.softmax(-1).masked_fill(~allowed, 0.)
    return weights @ v, weights


def causal_mask(query_length: int, key_length: int, device=None) -> Tensor:
    """Provided: upper-left aligned [Q,K] mask; key j is visible iff j <= i."""
    return torch.arange(key_length, device=device)[None, :] <= torch.arange(query_length, device=device)[:, None]
