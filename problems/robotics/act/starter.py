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
    'Oldest-to-newest (K,D) predictions -> (D,) after masking and exponential weighting.'
    raise NotImplementedError("Implement temporal_ensemble according to the README contract")


def style_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    'Mean over examples of KL(N(mu,exp(logvar)) || N(0,I)).'
    raise NotImplementedError("Implement style_kl according to the README contract")
