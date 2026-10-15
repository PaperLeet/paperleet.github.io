import torch
from torch import Tensor

def sequence_ratio(logp: Tensor, old_logp: Tensor, mask: Tensor) -> Tensor:
    """Length-normalized sequence importance weights, [B,L] -> [B].

    Returns:
        Return one floating Tensor ratio, shape [B], for [B, L] scores/mask (B responses, L
        token positions). Each value is the geometric mean of valid-token importance ratios;
        an empty response returns 1. Preserve input floating dtype/device and gradients to
        logp; old_logp is fixed. Do not reduce over the batch.
    """
    raise NotImplementedError('Implement sequence_ratio')

def gspo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor, clip_eps: float=0.2) -> Tensor:
    """Sequence clipping; empty responses contribute zero to the batch mean.

    Returns:
        Return one scalar floating Tensor loss, shape (), the negative mean clipped sequence
        objective over all B responses, including zero contributions from empty responses.
        Retain logp gradients, freezing old_logp and advantages. An all-empty mask returns a
        differentiable zero, not a Python number.
    """
    raise NotImplementedError('Implement gspo_loss')
