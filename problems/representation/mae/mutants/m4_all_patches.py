# BUG: Average every patch instead of only removed patches.
"""MAE masked-pixel reconstruction with the released sample-variance convention."""
import math
import torch
from torch import Tensor


def normalize_patch_targets(target: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize each [B,N,P] patch over P pixels using variance divisor P-1."""
    _features(target)
    _epsilon(eps)
    if target.shape[-1] < 2:
        raise ValueError('normalized patches need at least two pixel coordinates')
    centered = target - target.mean(-1, keepdim=True)
    variance = centered.square().sum(-1, keepdim=True) / (target.shape[-1] - 1)
    return centered / (variance + eps).sqrt()


def masked_reconstruction_loss(prediction: Tensor, target: Tensor, mask: Tensor,
                               norm_pix_loss: bool = False, eps: float = 1e-6) -> Tensor:
    """Mean pixel MSE over removed patches; targets are fixed data, mask=True means removed."""
    _check(prediction, target, mask, norm_pix_loss, eps)
    target = target.detach()
    if norm_pix_loss:
        target = normalize_patch_targets(target, eps)
    patch_loss = (prediction - target).square().mean(-1)
    return patch_loss.mean()


# Provided input checks; implement only the two functions above.
def _features(x):
    if x.ndim != 3 or min(x.shape) < 1 or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('expected nonempty float32/float64 [B,N,P]')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('pixel values must be finite')


def _epsilon(eps):
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be a finite positive Python number')


def _check(prediction, target, mask, norm_pix_loss, eps):
    _features(prediction); _features(target); _epsilon(eps)
    if prediction.shape != target.shape or prediction.dtype != target.dtype or prediction.device != target.device:
        raise ValueError('prediction and target must share shape, dtype and device')
    if mask.shape != prediction.shape[:2] or mask.dtype != torch.bool or mask.device != prediction.device:
        raise ValueError('mask must be boolean [B,N] on the prediction device')
    if not isinstance(norm_pix_loss, bool):
        raise ValueError('norm_pix_loss must be bool')
    if norm_pix_loss and prediction.shape[-1] < 2:
        raise ValueError('normalized patches need P>=2')
