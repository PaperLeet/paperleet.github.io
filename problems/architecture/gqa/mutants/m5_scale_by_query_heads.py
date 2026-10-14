# Bug: logits are scaled by query-head count instead of key feature width.
import math
import torch
from torch import Tensor


def mean_pool_kv_heads(weight: Tensor, num_kv_heads: int) -> Tensor:
    """[H,D_model,D_head] -> [G,D_model,D_head], pooling consecutive heads."""
    if weight.ndim != 3 or type(num_kv_heads) is not int or num_kv_heads < 1:
        raise ValueError('Expected a three-dimensional projection and a positive integer group count')
    heads = weight.shape[0]
    if heads < 1 or heads % num_kv_heads:
        raise ValueError('The group count must divide the original head count')
    return weight.reshape(num_kv_heads, heads // num_kv_heads, *weight.shape[1:]).mean(dim=1)


def grouped_query_attention(q: Tensor, k: Tensor, v: Tensor,
                            mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """q [B,H,Q,D], k [B,G,K,D], v [B,G,K,Dv]; True means visible."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('Expected four-dimensional projected tensors')
    heads, groups = q.shape[1], k.shape[1]
    if groups < 1 or heads < 1 or heads % groups or v.shape[1] != groups:
        raise ValueError('Query heads must form equal contiguous KV groups')
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0] or k.shape[2] != v.shape[2] or k.shape[2] < 1 or q.shape[-1] != k.shape[-1] or q.shape[-1] < 1 or v.shape[-1] < 1:
        raise ValueError('Incompatible batch, sequence or feature dimensions')
    repetitions = heads // groups
    expanded_k = k.repeat_interleave(repetitions, dim=1)
    expanded_v = v.repeat_interleave(repetitions, dim=1)
    scores = (q @ expanded_k.transpose(-1, -2)) / math.sqrt(q.shape[1])
    if mask is not None:
        allowed = torch.broadcast_to(mask, scores.shape)
        scores = scores.masked_fill(~allowed, -torch.inf)
        scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = scores.softmax(-1).masked_fill(~allowed, 0.)
    else:
        weights = scores.softmax(-1)
    return weights @ expanded_v, weights
