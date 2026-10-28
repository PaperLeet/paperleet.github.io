import math
import torch
from torch import Tensor


def _float(x, ndim=None):
    if not isinstance(x, Tensor) or not x.is_floating_point() or x.numel() == 0 or (ndim is not None and x.ndim != ndim) or not torch.isfinite(x).all():
        raise ValueError("expected a nonempty finite floating tensor with the declared rank")


def _same(a, b):
    _float(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("tensor shape, dtype and device must match")

def temporal_ensemble(actions: Tensor, valid: Tensor, decay: float) -> Tensor:
    """Oldest-to-newest (K,D) predictions -> (D,) after masking and exponential weighting."""
    _float(actions,2)
    if valid.shape!=actions.shape[:1] or valid.dtype!=torch.bool or valid.device!=actions.device or not valid.any() or not math.isfinite(decay) or decay<0:
        raise ValueError("invalid valid mask or decay")
    selected=actions[valid]
    weights=torch.exp(-decay*torch.arange(selected.shape[0],dtype=actions.dtype,device=actions.device))
    weights=weights/weights.sum()
    return (selected*weights.unsqueeze(-1)).sum(dim=0)


def style_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    """Mean over examples of KL(N(mu,exp(logvar)) || N(0,I))."""
    _float(mu,2);_same(mu,logvar)
    return .5*(mu.square()+logvar.exp()-1-logvar).sum(dim=-1).mean()
