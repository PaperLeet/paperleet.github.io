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

def simnorm(values: Tensor, group_dim: int) -> Tensor:
    'Normalize each consecutive group into a simplex.'
    raise NotImplementedError("Implement simnorm according to the README contract")


def elite_moments(actions: Tensor, values: Tensor, temperature: float, min_std: float, max_std: float) -> tuple:
    'Weighted MPPI moments over supplied elites (H,E,D); temperature multiplies value.'
    raise NotImplementedError("Implement elite_moments according to the README contract")
