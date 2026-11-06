# Bug: Detach prediction instead of preserving its learning path.
"""SimSiam: a stopped target in each direction, with a live prediction path."""
import torch
from torch import Tensor
from torch.nn import functional as F


def negative_cosine_similarity(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean negative row cosine; stop only the target, using normalization eps=1e-12."""
    _check_inputs(prediction, target)
    p = F.normalize(prediction.detach(), dim=1, eps=1e-12)
    z = F.normalize(target.detach(), dim=1, eps=1e-12)
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(p1: Tensor, p2: Tensor, z1: Tensor, z2: Tensor) -> Tensor:
    """Average p1-to-z2 and p2-to-z1; all four inputs share nonempty [B,D]."""
    for value in (p2, z1, z2):
        _check_inputs(p1, value)
    return (negative_cosine_similarity(p1, z2) + negative_cosine_similarity(p2, z1)) / 2


# Provided validation; the two objectives above are the exercise.
def _check_inputs(prediction, target):
    if prediction.ndim != 2 or target.shape != prediction.shape or min(prediction.shape) < 1:
        raise ValueError('inputs must share nonempty shape [B,D]')
    if prediction.dtype not in (torch.float32, torch.float64) or target.dtype != prediction.dtype:
        raise ValueError('inputs must share float32 or float64 dtype')
    if prediction.device != target.device:
        raise ValueError('inputs must share a device')
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError('inputs must contain finite values')
