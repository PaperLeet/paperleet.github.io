import math
import torch
from torch import Tensor


def attention_kernel(q: Tensor, k: Tensor, v: Tensor,
                     mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Provided: [B,H,Q,D] attention; True means visible, empty rows are zero.

    Returns:
        Return `(output, weights)` as floating tensors `[B,H,Q,Dv]` and `[B,H,Q,K]`: B batch
        size, H heads, Q/K query/key lengths and Dv value width. Weights sum to one over
        visible keys; a fully masked row has zero output and weights. Preserve the input
        dtype/device and q/k/v autograd with finite gradients for masked rows. This helper
        returns per-head values before concatenation/output projection and does not mutate
        inputs.
    """
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is None:
        weights = scores.softmax(-1)
    else:
        allowed = torch.broadcast_to(mask, scores.shape)
        scores = scores.masked_fill(~allowed, -torch.inf)
        scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = scores.softmax(-1).masked_fill(~allowed, 0.)
    return weights @ v, weights


def multi_head_attention(query: Tensor, key: Tensor, value: Tensor,
                         w_q: Tensor, w_k: Tensor, w_v: Tensor, w_o: Tensor,
                         num_heads: int, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Required: project, split heads, attend, concatenate, and output-project.
    query [B,Q,D], key/value [B,K,D], all projection matrices [D,D].
    Matrix convention is x @ w (paper layout, not nn.Linear.weight layout).
    D is divisible by num_heads. Boolean mask broadcasts to [B,H,Q,K],
    True means visible. Return output [B,Q,D] and weights [B,H,Q,K].
    Preserve autograd, dtype and device; no bias/dropout; fully masked rows zero.
    The attention kernel above is supplied and may be used unchanged.

    Returns:
        Return `(output, weights)` in that order: floating tensors `[B,Q,D]` and
        `[B,H,Q,K]`, where B is batch size, Q/K query/key lengths, D model width and
        H=num_heads. output has already concatenated heads and applied w_o; weights retain
        each head rather than averaging heads. Preserve input dtype/device and gradients to
        query/key/value and all four projection matrices. Fully masked query rows are zero;
        do not mutate inputs.
    """
    raise NotImplementedError()
