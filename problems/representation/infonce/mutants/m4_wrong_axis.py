# BUG: Normalize across contexts instead of candidates.
"""InfoNCE as positive-candidate classification, following CPC Equation (4)."""
import torch
from torch import Tensor


def info_nce_loss(logits: Tensor, positive_indices: Tensor) -> Tensor:
    """Mean loss for finite [B,K] logits and int64 [B] positive columns; preserve gradients."""
    _check_inputs(logits, positive_indices)
    positive = logits.gather(1, positive_indices[:, None]).squeeze(1)
    return (torch.logsumexp(logits, dim=0).mean() - positive).mean()


# Provided validation; implement only the objective above.
def _check_inputs(logits, positive_indices):
    if not isinstance(logits, Tensor) or logits.ndim != 2 or min(logits.shape) < 1:
        raise ValueError('logits must have nonempty shape [B,K]')
    if logits.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(logits).all()):
        raise ValueError('logits must contain finite float32 or float64 values')
    if not isinstance(positive_indices, Tensor) or positive_indices.shape != (logits.shape[0],):
        raise ValueError('positive_indices must have shape [B]')
    if positive_indices.dtype != torch.int64 or positive_indices.device != logits.device:
        raise ValueError('positive_indices must be int64 on the logits device')
    if bool((positive_indices < 0).any()) or bool((positive_indices >= logits.shape[1]).any()):
        raise ValueError('positive index outside the candidate range')
