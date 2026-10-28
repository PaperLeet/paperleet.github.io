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

def token_learner_pool(features: Tensor, logits: Tensor) -> Tensor:
    'Pool original (B,L,C) features with spatial softmax of (B,L,K) logits.'
    raise NotImplementedError("Implement token_learner_pool according to the README contract")
