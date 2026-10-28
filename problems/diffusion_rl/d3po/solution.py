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

def d3po_loss(preferred: Tensor, rejected: Tensor, ref_preferred: Tensor, ref_rejected: Tensor, beta: float) -> Tensor:
    """Equal (B,T) log probabilities; mean of independent per-transition preference losses."""
    _finite(preferred,rejected,ref_preferred,ref_rejected);_positive(beta)
    if preferred.ndim!=2 or any(x.shape!=preferred.shape for x in (rejected,ref_preferred,ref_rejected)):raise ValueError("equal (B,T) shapes required")
    margin=beta*((preferred-ref_preferred.detach())-(rejected-ref_rejected.detach()))
    return F.softplus(-margin).mean()
