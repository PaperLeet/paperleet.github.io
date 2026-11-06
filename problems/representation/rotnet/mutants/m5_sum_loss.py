# BUG: Sum instead of averaging cross entropy.
"""RotNet's four exact image rotations and self-generated classification targets."""
import torch
from torch import Tensor


def rotation_batch(images: Tensor) -> tuple[Tensor, Tensor]:
    """[B,C,H,H] -> [4B,C,H,H], int64 [4B]; rotation-major CCW order 0,90,180,270."""
    _images(images)
    r90 = images.transpose(-2, -1).flip(-2)
    r180 = images.flip((-2, -1))
    r270 = images.transpose(-2, -1).flip(-1)
    views = torch.cat([images, r90, r180, r270], dim=0)
    labels = torch.arange(4, dtype=torch.int64, device=images.device).repeat_interleave(images.shape[0])
    return views, labels


def rotation_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Stable mean four-way cross entropy for [M,4] logits and supplied int64 [M] labels."""
    _loss_inputs(logits, labels)
    chosen = logits.gather(1, labels[:, None]).squeeze(1)
    return (torch.logsumexp(logits, dim=1) - chosen).sum()


def _images(x):
    if not isinstance(x, Tensor) or x.ndim != 4 or min(x.shape) < 1 or x.shape[-2] != x.shape[-1]:
        raise ValueError('images must have nonempty square shape [B,C,H,H]')
    if x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
        raise ValueError('images must be finite float32/float64')


def _loss_inputs(x, y):
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != 4:
        raise ValueError('logits must have nonempty shape [M,4]')
    if x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
        raise ValueError('logits must be finite float32/float64')
    if not isinstance(y, Tensor) or y.shape != (x.shape[0],) or y.dtype != torch.int64 or y.device != x.device:
        raise ValueError('labels must be int64 [M] on the logits device')
    if bool((y < 0).any()) or bool((y > 3).any()):
        raise ValueError('rotation labels must lie in [0,3]')
