"""I-JEPA's released training objective and momentum-target operations."""
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def _features(x, ndim=None):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when x is a finite, nonempty float32/float64
        Tensor with the requested rank when ndim is supplied. Raise ValueError when a checked
        condition fails; do not mutate inputs or return transformed tensors.
    """
    if not isinstance(x, Tensor) or (ndim is not None and x.ndim != ndim) or not x.numel() or x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all():
        raise ValueError('expected finite, nonempty float32/float64 tensors')


def _pair(a, b):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when both finite nonempty floating tensors have
        identical shape, dtype and device. Raise ValueError when a checked condition fails; do
        not mutate inputs or return transformed tensors.
    """
    _features(a); _features(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError('tensor shape, dtype and device must match')


def _check_targets(features, masks, num_context, eps):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when features [B,N,D], equal-length int64 masks
        [B,M], positive integer context count and positive finite epsilon satisfy the target
        contract. Raise ValueError when a checked condition fails; do not mutate inputs or
        return transformed tensors.
    """
    _features(features, 3)
    if not isinstance(num_context, int) or isinstance(num_context, bool) or num_context < 1 or not math.isfinite(eps) or eps <= 0:
        raise ValueError('positive context count and finite epsilon required')
    if not isinstance(masks, (list, tuple)) or not masks:
        raise ValueError('at least one target block required')
    size = None
    for mask in masks:
        if not isinstance(mask, Tensor) or mask.ndim != 2 or mask.dtype != torch.int64 or mask.device != features.device or mask.shape[0] != features.shape[0] or mask.shape[1] < 1:
            raise ValueError('each mask must be int64 [B,M] on the feature device')
        if size is not None and mask.shape[1] != size:
            raise ValueError('target blocks must have equal lengths')
        if (mask < 0).any() or (mask >= features.shape[1]).any():
            raise ValueError('target indices out of range')
        size = mask.shape[1]


def masked_targets(features: Tensor, masks: list[Tensor], num_context: int = 1, eps: float = 1e-5) -> Tensor:
    """Return detached normalized targets [K*C*B,M,D], ordered by block/context/batch.

    Returns:
        Return a detached floating Tensor [K*C*B,M,D], ordered by target block, context, then
        batch item. K is the number of masks, C is num_context, B is batch size, M is tokens per
        mask and D is feature width. Preserve the feature dtype/device; no gradient reaches
        teacher features and no input is modified.
    """
    raise NotImplementedError("Implement masked_targets")


def masked_prediction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Released I-JEPA code convention: smooth L1 with beta=1, mean over all entries.

    Returns:
        Return one scalar Tensor, shape (), of the common prediction/target dtype and device:
        smooth-L1 with beta=1 averaged over every tensor element. Retain prediction gradients
        but detach the target internally. Do not return per-token losses, a Python float or
        mutate either input.
    """
    raise NotImplementedError("Implement masked_prediction_loss")


def ema_update(target: Tensor, online: Tensor, momentum: float) -> Tensor:
    """Functional, detached equivalent of the author's in-place parameter EMA.

    Returns:
        Return a new detached Tensor with the same shape, floating dtype and device as target
        and online, containing the updated target parameters. Neither input receives gradients
        or is updated in place. This functional helper returns the new value, not None.
    """
    raise NotImplementedError("Implement ema_update")
