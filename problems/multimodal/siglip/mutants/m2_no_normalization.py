# BUG: 图像特征未做逐行归一化
"""SigLIP pairwise logistic objective with differentiable normalized features."""
import torch
from torch import Tensor
import torch.nn.functional as F


def _check_features(images, texts, log_scale, bias):
    if images.ndim != 2 or images.numel() == 0 or not images.is_floating_point():
        raise ValueError("images must be a nonempty floating (B,D) tensor")
    if texts.shape != images.shape or texts.dtype != images.dtype or texts.device != images.device:
        raise ValueError("texts must match images")
    for value in (images, texts, log_scale, bias):
        if value.dtype != images.dtype or value.device != images.device or not torch.isfinite(value).all():
            raise ValueError("all inputs must be finite and share dtype/device")
    if log_scale.ndim != 0 or bias.ndim != 0 or not torch.isfinite(log_scale.exp()).all():
        raise ValueError("log_scale and bias must be scalars with a finite exponential scale")


def pairwise_logits(images: Tensor, texts: Tensor, log_scale: Tensor, bias: Tensor) -> Tensor:
    """Return (B,B) logits, preserving gradients through both towers and both scalars."""
    _check_features(images, texts, log_scale, bias)
    image_unit = images
    text_unit = F.normalize(texts, p=2, dim=-1, eps=1e-12)
    return log_scale.exp() * (image_unit @ text_unit.T) + bias


def siglip_loss(logits: Tensor) -> Tensor:
    """Sum all B squared pair losses and divide by B, with diagonal positives."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or logits.numel() == 0:
        raise ValueError("logits must be a nonempty square (B,B) tensor")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating point")
    labels = 2 * torch.eye(logits.shape[0], dtype=logits.dtype, device=logits.device) - 1
    return F.softplus(-labels * logits).sum() / logits.shape[0]
