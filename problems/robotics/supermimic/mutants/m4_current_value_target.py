# BUG: Bootstrap lambda targets from the current state instead of the next state.
"""SuperMimic soft-survival actor objective and detached critic targets."""
import math
import torch
from torch import Tensor


def _rollout(rewards, termination, gamma):
    if rewards.ndim != 2 or rewards.numel() == 0 or not rewards.is_floating_point() or not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite nonempty floating (B, H)")
    if termination.shape != rewards.shape or termination.dtype != rewards.dtype or termination.device != rewards.device:
        raise ValueError("termination must match rewards")
    if not torch.isfinite(termination).all() or ((termination < 0) | (termination > 1)).any():
        raise ValueError("termination probabilities must lie in [0,1]")
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must lie in [0,1)")


def imagined_return(rewards: Tensor, termination: Tensor, bootstrap: Tensor, gamma: float) -> Tensor:
    """Mean of survival-weighted rewards plus discounted terminal critic value."""
    _rollout(rewards, termination, gamma)
    if bootstrap.shape != rewards.shape[:1] or bootstrap.dtype != rewards.dtype or bootstrap.device != rewards.device or not torch.isfinite(bootstrap).all():
        raise ValueError("bootstrap must be a finite matching (B,) tensor")
    weight = torch.ones_like(bootstrap)
    total = torch.zeros_like(bootstrap)
    for step in range(rewards.shape[1]):
        total = total + weight * rewards[:, step]
        weight = weight * gamma * (1 - termination[:, step])
    return (total + weight * bootstrap).mean()


def lambda_return_targets(rewards: Tensor, termination: Tensor, values: Tensor, gamma: float, lam: float) -> Tensor:
    """Frozen Eq. (3) targets, with values[:, H] bootstrapping the final step."""
    _rollout(rewards, termination, gamma)
    if values.shape != (rewards.shape[0], rewards.shape[1] + 1) or values.dtype != rewards.dtype or values.device != rewards.device or not torch.isfinite(values).all():
        raise ValueError("values must be finite matching (B, H+1)")
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("lambda must lie in [0,1]")
    r, d, v = rewards.detach(), termination.detach(), values.detach()
    target = v[:, -1]
    targets = []
    for step in range(rewards.shape[1] - 1, -1, -1):
        target = r[:, step] + gamma * (1 - d[:, step]) * ((1 - lam) * v[:, step] + lam * target)
        targets.append(target)
    return torch.stack(targets[::-1], dim=1)
