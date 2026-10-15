import torch
from torch import Tensor
import torch.nn.functional as F


# Required: implement the core functions below.
def dpo_loss(pi_chosen: Tensor, pi_rejected: Tensor, ref_chosen: Tensor, ref_rejected: Tensor, beta: float=0.1, label_smoothing: float=0.0) -> tuple[Tensor, Tensor, Tensor]:
    """Four [B] sequence log-probs; reference tensors are fixed snapshots.
    Return (loss, chosen_rewards, rejected_rewards), in that order:
      loss: batch-mean scalar Tensor, shape (), with policy gradients.
      chosen_rewards: beta * (pi_chosen - ref_chosen), detached, shape [B].
      rejected_rewards: beta * (pi_rejected - ref_rejected), detached, shape [B].
    Keep rewards shaped [1] for B=1; do not average or squeeze them.
    Keep loss as a Tensor (no item/detach); reference inputs receive no gradients.
    label_smoothing weights reversed labels; beta scales implicit rewards."""
    raise NotImplementedError()


# Provided: keep these helpers and call them as needed.
def sequence_logprob(logits: Tensor, input_ids: Tensor, loss_mask: Tensor) -> Tensor:
    """Provided helper: shift labels/mask and sum response log-probabilities."""
    scores = logits[:, :-1].log_softmax(-1)
    selected = scores.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (selected * loss_mask[:, 1:]).sum(-1)
