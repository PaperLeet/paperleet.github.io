# BUG: Execution fails to mask absent action coordinates.
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
    """Map (B,H,D) actions to width semantic slots; return actions and bool validity."""
    _float(local,3)
    if type(width)!=int or width<1 or not isinstance(slots,list) or len(slots)!=local.shape[-1] or any(type(i)!=int or i<0 or i>=width for i in slots) or len(set(slots))!=len(slots):
        raise ValueError("invalid unified slot map")
    basis=local.new_tensor([[float(k==slot) for k in range(width)] for slot in slots])
    return local@basis,basis.sum(dim=0)>0


def mask_actions(actions: Tensor, valid: Tensor) -> Tensor:
    """Suppress absent semantic dimensions in a sampled (B,H,W) action sequence."""
    _float(actions,3)
    if valid.shape!=(actions.shape[-1],) or valid.dtype!=torch.bool or valid.device!=actions.device:
        raise ValueError("invalid action mask")
    return actions
