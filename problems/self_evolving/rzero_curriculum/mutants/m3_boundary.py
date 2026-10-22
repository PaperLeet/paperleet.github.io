# BUG: Drop the lower endpoint.
import torch
from torch import Tensor

def consensus(answers: Tensor, num_answers: int) -> tuple[Tensor, Tensor]:
    """answers: long [B,K], B,K>=1, IDs in [0,num_answers). Answer equivalence is supplied.
    Return (labels: long [B], consistency: float64 [B]) on the input device.
    Ties choose the smallest answer ID. Count ALL K answers. No mutation.
    """
    counts = (answers[:, :, None] == torch.arange(num_answers, device=answers.device)).sum(dim=1)
    labels = counts.argmax(dim=1)
    shares = counts.max(dim=1).values.to(torch.float64) / answers.shape[1]
    return labels, shares

def curriculum_mask(consistency: Tensor, low: float = .25, high: float = .75) -> Tensor:
    """Float [B], 0<=low<=high<=1. Return bool [B], retaining BOTH boundary values."""
    return (consistency > low) & (consistency <= high)

def pseudo_rewards(answers: Tensor, labels: Tensor) -> Tensor:
    """New long answers [B,K], frozen pseudo-labels [B]. Return float64 [B,K].
    Match against the supplied OLD labels, never recompute a vote from new answers.
    These are agreement rewards, not independent correctness verification.
    """
    return (answers == labels[:, None]).to(torch.float64)
