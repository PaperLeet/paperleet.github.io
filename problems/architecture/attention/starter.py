import math
import torch
from torch import Tensor


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                                 mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Required: q [B,H,Q,D], k [B,H,K,D], v [B,H,K,Dv].
    Return output [B,H,Q,Dv], weights [B,H,Q,K]. Boolean mask broadcasts
    to weights; True means allowed. Fully masked queries return exact zeros
    and finite zero gradients. Preserve dtype/device; no dropout.

    Returns:
        Return the Python tuple `(output, weights)`, both floating tensors:
        `output[B,H,Q,Dv]` contains value aggregates and `weights[B,H,Q,K]` contains
        attention probabilities. B is batch size, H head count, Q/K query/key lengths, and
        Dv value width. Each nonempty allowed-key row sums to one; fully masked rows have
        exactly zero weights and output. Preserve the inputs' dtype/device and gradients
        through q/k/v; masked rows have finite zero gradients. Do not modify inputs.
    """
    raise NotImplementedError()


def causal_mask(query_length: int, key_length: int, device=None) -> Tensor:
    """Provided: upper-left aligned [Q,K] mask; key j is visible iff j <= i.

    Returns:
        Return one `torch.bool` Tensor `[Q,K]`, where Q=query_length and K=key_length; rows
        are queries and columns are keys. An entry is True exactly when the key index is at
        most the query index (upper-left alignment). Use the requested device. This
        visibility mask has no gradients.
    """
    return torch.arange(key_length, device=device)[None, :] <= torch.arange(query_length, device=device)[:, None]
