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

def pack_actions(local: Tensor, slots: list, width: int) -> tuple:
    'Map (B,H,D) actions to width semantic slots; return actions and bool validity.'
    raise NotImplementedError("Implement pack_actions according to the README contract")


def mask_actions(actions: Tensor, valid: Tensor) -> Tensor:
    'Suppress absent semantic dimensions in a sampled (B,H,W) action sequence.'
    raise NotImplementedError("Implement mask_actions according to the README contract")
