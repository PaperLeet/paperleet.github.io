import torch
from torch import Tensor


# Required: implement the core functions below.
def compute_gae(rewards: Tensor, values: Tensor, next_values: Tensor, terminated: Tensor, truncated: Tensor, gamma: float=0.99, lam: float=0.95) -> tuple[Tensor, Tensor]:
    """Compute advantages and returns, both [T,N], preserving rewards dtype/device.
    rewards/values/next_values: floating [T,N]. next_values uses the true pre-reset final
    state. terminated/truncated: bool [T,N]. Termination removes bootstrap; both endings
    stop propagation. returns = advantages + values.

    Returns:
        Return (advantages, returns), a tuple of two floating Tensors in that order. Both
        have shape [T, N]: T time steps and N environments, preserving rewards dtype/device.
        returns = advantages + values; keep both axes even when T = N = 1. Return the
        arrays, with no batch/time reduction.
    """
    raise NotImplementedError()
