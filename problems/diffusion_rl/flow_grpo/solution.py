"""Reverse-time rectified-flow Gaussian transitions and Flow-GRPO."""
import math
import torch
from torch import Tensor


def _matching(*values):
    first = values[0]
    if not first.numel() or not first.is_floating_point():
        raise ValueError("nonempty floating tensors required")
    for value in values:
        if value.shape != first.shape or value.dtype != first.dtype or value.device != first.device:
            raise ValueError("tensor shapes, dtypes and devices must match")
        if not torch.isfinite(value).all():
            raise ValueError("inputs must be finite")


def sde_transition(state: Tensor, velocity: Tensor, time: float,
                   next_time: float, noise_level: float) -> tuple[Tensor, Tensor]:
    """Return a reverse-time Gaussian mean and scalar standard deviation."""
    _matching(state, velocity)
    if state.ndim < 2:
        raise ValueError("state needs a batch and feature dimensions")
    if not all(math.isfinite(v) for v in (time, next_time, noise_level)):
        raise ValueError("schedule values must be finite")
    if not 0 <= next_time < time < 1 or noise_level < 0:
        raise ValueError("require 0 <= next_time < time < 1 and nonnegative noise_level")
    dt = next_time - time
    sigma2 = noise_level ** 2 * time / (1 - time)
    drift = velocity + sigma2 / (2 * time) * (state.detach() + (1 - time) * velocity)
    mean = state.detach() + dt * drift
    std = state.new_tensor(math.sqrt(sigma2 * (-dt)))
    return mean, std


def flow_grpo_loss(current: Tensor, old: Tensor, reference: Tensor,
                   actions: Tensor, std: Tensor, rewards: Tensor,
                   clip_epsilon: float, beta: float, eps: float = 1e-8) -> Tensor:
    """Negative clipped objective plus exact equal-variance Gaussian KL.

    Means/actions: (P,G,T,D); std: (T,); terminal rewards: (P,G).
    Each density sums over D; average the objective over prompts, group and time.
    """
    _matching(current, old, reference, actions)
    if current.ndim != 4 or current.shape[1] < 2:
        raise ValueError("means need shape (P,G,T,D) with at least two group members")
    for value, shape in ((std, current.shape[2:3]), (rewards, current.shape[:2])):
        if value.shape != shape or value.dtype != current.dtype or value.device != current.device or not torch.isfinite(value).all():
            raise ValueError("std/rewards must have matching shape, dtype, device and finite values")
    if (std <= 0).any():
        raise ValueError("policy standard deviations must be positive")
    if not all(math.isfinite(v) for v in (clip_epsilon, beta, eps)) or not 0 < clip_epsilon < 1 or beta < 0 or eps <= 0:
        raise ValueError("require 0 < clip_epsilon < 1, beta >= 0 and eps > 0")
    frozen_rewards = rewards.detach()
    centered = frozen_rewards - frozen_rewards.mean(1, keepdim=True)
    advantage = centered / (centered.square().mean(1, keepdim=True).sqrt() + eps)
    variance = std.detach().square().reshape(1, 1, -1, 1)
    sampled = actions.detach()
    log_ratio = (((sampled - old.detach()).square() - (sampled - current).square()) / (2 * variance)).sum(-1)
    advantage = advantage.unsqueeze(-1)
    # Apply the equivalent sign-aware clipping before exponentiating. This
    # avoids overflow in already-clipped positive samples and in tied groups.
    bounded = torch.where(advantage > 0, log_ratio.clamp(max=math.log1p(clip_epsilon)),
                          log_ratio.clamp(min=math.log1p(-clip_epsilon)))
    bounded = torch.where(advantage == 0, torch.zeros_like(bounded), bounded)
    ratio = bounded.exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    kl = ((current - reference.detach()).square() / (2 * variance)).sum(-1)
    return (-surrogate + beta * kl).mean()
