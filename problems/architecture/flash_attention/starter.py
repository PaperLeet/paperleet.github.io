import math
import torch
from torch import Tensor


def online_softmax_update(scores: Tensor, values: Tensor, row_max: Tensor,
                          normalizer: Tensor, numerator: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Required: merge finite scores [Q,K] and values [K,Dv] into online state.
    row_max [Q] starts at -inf, normalizer [Q] and numerator [Q,Dv] at zero.
    Rescale old mass AND weighted values whenever the row maximum changes.
    Return new maximum, unnormalized sum, unnormalized weighted sum.
    Preserve autograd and inputs. All blocks have at least one key.

    Returns:
        Return `(new_max, new_normalizer, new_numerator)` in exactly this order, as a Python
        tuple of floating tensors shaped `[Q]`, `[Q]`, `[Q,Dv]`. Q is the number of query
        rows and Dv the value width. These are the updated row maximum, unnormalized
        exponential mass and unnormalized weighted-value sum over all processed key blocks;
        the numerator is not yet the normalized attention output. Preserve dtype/device,
        autograd and all input state tensors; return updated state rather than updating in
        place.
    """
    raise NotImplementedError()


def block_attention(q: Tensor, k: Tensor, v: Tensor, block_size: int = 16) -> tuple[Tensor, Tensor]:
    """Provided: tile traversal, no dropout/masks; returns output and per-row log-sum-exp.

    Returns:
        Return `(output, logsumexp)`, a Python tuple of floating tensors `[Q,Dv]` and `[Q]`.
        Q is total query length and Dv value width. output is normalized attention;
        logsumexp is one natural-log sum-exp of scaled scores over all keys per query, not
        an attention-weight matrix. Preserve q/k/v dtype/device and their gradient paths,
        including gradients from logsumexp. Inputs remain unchanged.
    """
    outputs, logsumexp = [], []
    for query_start in range(0, q.shape[0], block_size):
        queries = q[query_start:query_start + block_size]
        row_max = q.new_full((len(queries),), -torch.inf)
        normalizer = q.new_zeros(len(queries))
        numerator = q.new_zeros((len(queries), v.shape[-1]))
        for key_start in range(0, k.shape[0], block_size):
            scores = (queries @ k[key_start:key_start + block_size].T) / math.sqrt(q.shape[-1])
            values = v[key_start:key_start + block_size]
            row_max, normalizer, numerator = online_softmax_update(scores, values, row_max, normalizer, numerator)
        outputs.append(numerator / normalizer[:, None])
        logsumexp.append(row_max + normalizer.log())
    return torch.cat(outputs), torch.cat(logsumexp)
