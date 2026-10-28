# BUG: Use cumulative alpha directly instead of its standard-deviation square root.
"""Action-space DDPM training and observation-aligned receding-horizon execution."""
import torch
from torch import Tensor


def _sequence(value):
    if value.ndim != 3 or value.numel() == 0 or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("expected finite nonempty floating (B, T, D) actions")


def _matching(left, right):
    _sequence(left)
    if right.shape != left.shape or right.dtype != left.dtype or right.device != left.device or not torch.isfinite(right).all():
        raise ValueError("action tensors must match shape, dtype and device")


def add_action_noise(actions: Tensor, noise: Tensor, alpha_bar: Tensor) -> Tensor:
    """Forward noise normalized actions using one cumulative alpha per example."""
    _matching(actions, noise)
    if alpha_bar.shape != actions.shape[:1] or alpha_bar.dtype != actions.dtype or alpha_bar.device != actions.device:
        raise ValueError("alpha_bar must be a matching (B,) tensor")
    if not torch.isfinite(alpha_bar).all() or ((alpha_bar <= 0) | (alpha_bar >= 1)).any():
        raise ValueError("alpha_bar must lie strictly between zero and one")
    scale = alpha_bar.reshape(-1, 1, 1)
    return scale * actions + (1 - scale) * noise


def action_denoising_loss(prediction: Tensor, noise: Tensor) -> Tensor:
    """Mean squared epsilon error across examples, horizon and action coordinates."""
    _matching(prediction, noise)
    return (prediction - noise.detach()).square().mean()


def action_chunk(actions: Tensor, observation_horizon: int, execution_horizon: int) -> Tensor:
    """Extract the actions aligned with the last observation, then execute Ta steps."""
    _sequence(actions)
    if type(observation_horizon) is not int or type(execution_horizon) is not int or observation_horizon < 1 or execution_horizon < 1:
        raise ValueError("horizons must be positive integers")
    start = observation_horizon - 1
    if start + execution_horizon > actions.shape[1]:
        raise ValueError("execution chunk must fit inside the prediction horizon")
    return actions[:, start:start + execution_horizon, :]
