# BUG: Padding dilutes the loss because the valid fraction is omitted.
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

def masked_action_loss(prediction: Tensor, target: Tensor, timestep_valid: Tensor, action_valid: Tensor) -> Tensor:
    """Octo masked squared-error reduction before action-head-specific scaling."""
    _float(prediction,4);_same(prediction,target)
    if timestep_valid.shape!=prediction.shape[:2] or action_valid.shape!=prediction.shape or timestep_valid.dtype!=torch.bool or action_valid.dtype!=torch.bool or timestep_valid.device!=prediction.device or action_valid.device!=prediction.device: raise ValueError("invalid padding masks")
    valid=(timestep_valid[:,:,None,None]&action_valid).to(dtype=prediction.dtype)
    error=(prediction-target.detach()).square()
    return (error*valid).mean()
