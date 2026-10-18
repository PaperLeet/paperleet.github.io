# BUG: average parameter contributions instead of summing.
"""EWC with an already supplied Fisher diagonal; no Fisher estimation is required."""
import torch
from torch import Tensor


def ewc_penalty(theta: Tensor, anchor: Tensor, fisher_diag: Tensor, strength: float) -> Tensor:
    """Sum the fixed Fisher-weighted displacement, preserving theta-only gradients."""
    if strength == 0:
        return (theta * 0).sum()
    fisher = fisher_diag.detach()
    displacement = theta - anchor.detach()
    displacement = torch.where(fisher > 0, displacement, torch.zeros_like(theta))
    return (strength / 2) * (fisher * displacement.square()).mean()
