import math
import torch
from torch import Tensor


def mean_pool_kv_heads(weight: Tensor, num_kv_heads: int) -> Tensor:
    """[H,D_model,D_head] -> [G,D_model,D_head], pooling consecutive heads."""
    raise NotImplementedError('Implement contiguous head mean pooling')


def grouped_query_attention(q: Tensor, k: Tensor, v: Tensor,
                            mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """q [B,H,Q,D], k [B,G,K,D], v [B,G,K,Dv]; True means visible."""
    raise NotImplementedError('Implement grouped-query attention')
