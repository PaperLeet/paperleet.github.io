import math
import torch
from torch import Tensor


def mean_pool_kv_heads(weight: Tensor, num_kv_heads: int) -> Tensor:
    """[H,D_model,D_head] -> [G,D_model,D_head], pooling consecutive heads.

    Returns:
        Return one floating Tensor `[G,D_model,D_head]`, where G=num_kv_heads, D_model is
        projection input width and D_head per-head output width. Each returned head averages
        one consecutive group of H/G original heads from weight `[H,D_model,D_head]`,
        preserving group order. Preserve weight's dtype/device and gradients to all original
        heads; do not mutate weight.
    """
    raise NotImplementedError('Implement contiguous head mean pooling')


def grouped_query_attention(q: Tensor, k: Tensor, v: Tensor,
                            mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """q [B,H,Q,D], k [B,G,K,D], v [B,G,K,Dv]; True means visible.

    Returns:
        Return `(output, weights)` as floating tensors `[B,H,Q,Dv]` and `[B,H,Q,K]`,
        retaining all H query heads even though K/V have only G heads. B is batch size, Q/K
        query/key lengths and Dv value width. Weights normalize over keys; fully masked rows
        give zero weights/output and finite zero gradients. Preserve dtype/device and q/k/v
        gradients, accumulating contributions to shared KV heads. Empty queries keep a zero-
        length Q axis; inputs remain unchanged.
    """
    raise NotImplementedError('Implement grouped-query attention')
