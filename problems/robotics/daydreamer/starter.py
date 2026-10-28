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

def balanced_kl(posterior_logits: Tensor, prior_logits: Tensor, alpha: float = .8) -> Tensor:
    'Sum categorical latent KLs, average batch, and balance stopped gradient branches.'
    raise NotImplementedError("Implement balanced_kl according to the README contract")
