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

def symlog(values: Tensor) -> Tensor:
    'Signed logarithmic compression; preserve the source sign/abs autodiff convention.'
    raise NotImplementedError("Implement symlog according to the README contract")


def symexp(values: Tensor) -> Tensor:
    'Inverse signed exponential transform.'
    raise NotImplementedError("Implement symexp according to the README contract")


def two_hot(values: Tensor, bins: Tensor) -> Tensor:
    'Detached linear interpolation labels on an ordered transformed-value support.'
    raise NotImplementedError("Implement two_hot according to the README contract")
