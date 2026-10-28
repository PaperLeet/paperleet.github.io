# Allow a negative KL baseline
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _finite(*xs):
    if any(not x.numel() or not x.is_floating_point() or not torch.isfinite(x).all() for x in xs):
        raise ValueError("nonempty finite floating tensors required")
    if any(x.dtype != xs[0].dtype or x.device != xs[0].device for x in xs):
        raise ValueError("dtype and device must agree")


def _positive(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("positive finite scalar required")

def kto_loss(current: Tensor, reference: Tensor, labels: Tensor, unrelated_log_ratios: Tensor, beta: float) -> Tensor:
    """(B,) transition log probabilities and +/-1 feedback; unrelated baseline has shape (M,)."""
    _finite(current,reference,labels,unrelated_log_ratios);_positive(beta)
    if current.ndim!=1 or current.shape!=reference.shape or current.shape!=labels.shape or unrelated_log_ratios.ndim!=1:raise ValueError("vector shapes required")
    if not ((labels==1)|(labels==-1)).all():raise ValueError("labels must be +/-1")
    baseline=unrelated_log_ratios.detach().mean()
    margin=labels.detach()*beta*(current-reference.detach()-baseline)
    return -(torch.sigmoid(margin)-.5).mean()
