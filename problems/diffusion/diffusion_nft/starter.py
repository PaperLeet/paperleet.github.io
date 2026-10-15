"""DiffusionNFT Eq. (5), with a frozen rollout policy and bounded rewards."""
import math
import torch
from torch import Tensor


def _check_velocity(current, old, beta):
    """Provided validation helper.

    Returns:
        Provided: return Python None when velocities and beta satisfy the contract, not a
        boolean or a validated tensor. Invalid inputs raise ValueError. Validation only:
        leave inputs and their gradient settings unchanged. No returned tensor shape or
        dtype/device applies.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if current.ndim < 2 or current.numel() == 0 or not current.is_floating_point():
        raise ValueError("velocity needs a nonempty batch and feature dimensions")
    if old.shape != current.shape or old.dtype != current.dtype or old.device != current.device:
        raise ValueError("old and current velocity must match")
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError("velocities must be finite")


def implicit_velocities(current: Tensor, old: Tensor, beta: float) -> tuple[Tensor, Tensor]:
    """Return the implicit positive and negative policies; only current receives gradients.

    Returns:
        Return the tuple (positive, negative), in that order. Both are Tensors matching
        current shape [B,...] and dtype/device; B is batch size and trailing axes are
        feature dimensions. Keep both branches differentiable through current, detach old
        inside the function, and leave inputs unchanged.
    """
    _check_velocity(current, old, beta)
    raise NotImplementedError("Implement the paper core")


def nft_loss(current: Tensor, old: Tensor, target: Tensor, rewards: Tensor,
             beta: float) -> Tensor:
    """Mean over examples of reward-weighted squared L2 norms, summed over features.

    Returns:
        Return a scalar Tensor of shape (), with current dtype/device. Sum squared errors
        over all non-batch coordinates within each example, then average over B examples; do
        not average over all elements. Gradients flow only into current; old, target and
        rewards are stopped. Do not mutate inputs.
    """
    _check_velocity(current, old, beta)
    if target.shape != current.shape or target.dtype != current.dtype or target.device != current.device:
        raise ValueError("target velocity must match current")
    if rewards.shape != current.shape[:1] or rewards.dtype != current.dtype or rewards.device != current.device:
        raise ValueError("rewards must be a matching (B,) tensor")
    if not torch.isfinite(target).all() or not torch.isfinite(rewards).all() or ((rewards < 0) | (rewards > 1)).any():
        raise ValueError("finite targets and rewards in [0,1] required")
    raise NotImplementedError("Implement the paper core")
