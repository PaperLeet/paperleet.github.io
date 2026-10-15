import torch
from torch import Tensor


def headwise_orthogonalize(matrix: Tensor, num_heads: int) -> Tensor:
    """Implement GLM-5's split / per-head orthogonalize / merge core.

    Input [num_heads*head_dim,input_dim], float32/float64 on CPU.
    Heads occupy consecutive ROW blocks. For each block U @ diag(s) @ Vh,
    return U @ diag(s > max(head_dim,input_dim)*eps(dtype)*max(s)) @ Vh.
    Preserve input shape/dtype and autograd on full-rank blocks with distinct singular values. Do not mutate.
    Reject nonfloating/nonmatrix inputs, empty dimensions, and invalid head counts.
    This exact-SVD exercise replaces the production approximate polar iteration.

    Returns:
        Return one floating Tensor `[num_heads*head_dim,input_dim]` matching matrix's
        shape/dtype/device. It concatenates the per-head partial polar factors in original
        consecutive row-head order, not a list or a tensor with an extra head axis. Singular
        directions below the declared relative cutoff contribute zero; an all-zero head
        stays zero. Preserve autograd only on full-rank blocks with distinct singular values
        (rank boundaries/repeated values are excluded from the gradient contract). Do not
        modify matrix.
    """
    raise NotImplementedError()
