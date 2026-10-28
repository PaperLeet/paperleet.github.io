# BUG: Sum features instead of averaging per-example denoising errors.
"""Diffusion-DPO with the official feature-mean, half-beta convention."""
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def _matching(*values):
    first = values[0]
    if first.ndim < 2 or not first.numel() or not first.is_floating_point():
        raise ValueError("nonempty floating (B,...) tensors required")
    for value in values:
        if value.shape != first.shape or value.dtype != first.dtype or value.device != first.device:
            raise ValueError("tensor shapes, dtypes and devices must match")
        if not torch.isfinite(value).all():
            raise ValueError("inputs must be finite")


def paired_noising(winner: Tensor, loser: Tensor, noise: Tensor,
                   alpha_bar: Tensor) -> tuple[Tensor, Tensor]:
    """Use one noise tensor and one cumulative alpha per preference pair."""
    _matching(winner, loser, noise)
    if alpha_bar.shape != winner.shape[:1] or alpha_bar.dtype != winner.dtype or alpha_bar.device != winner.device:
        raise ValueError("alpha_bar must match the (B,) batch")
    if not torch.isfinite(alpha_bar).all() or ((alpha_bar < 0) | (alpha_bar > 1)).any():
        raise ValueError("alpha_bar must be in [0,1]")
    a = alpha_bar.detach().reshape((-1,) + (1,) * (winner.ndim - 1))
    shared = (1 - a).sqrt() * noise.detach()
    return a.sqrt() * winner.detach() + shared, a.sqrt() * loser.detach() + shared


def diffusion_dpo_loss(model_w: Tensor, model_l: Tensor, ref_w: Tensor,
                       ref_l: Tensor, noise: Tensor, beta: float) -> Tensor:
    """Stable pairwise logistic loss; gradients reach only the current predictions."""
    _matching(model_w, model_l, ref_w, ref_l, noise)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    target = noise.detach()
    current_gap = ((model_w - target).square() - (model_l - target).square()).flatten(1).sum(-1)
    reference_gap = ((ref_w.detach() - target).square() - (ref_l.detach() - target).square()).flatten(1).sum(-1)
    margin = -0.5 * beta * (current_gap - reference_gap)
    return F.softplus(-margin).mean()
