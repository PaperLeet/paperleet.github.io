import torch
from torch import Tensor
import torch.nn.functional as F


# Required: implement the core functions below.
def dpo_loss(pi_chosen: Tensor, pi_rejected: Tensor, ref_chosen: Tensor, ref_rejected: Tensor, beta: float=0.1, label_smoothing: float=0.0) -> tuple[Tensor, Tensor, Tensor]:
    """Four [B] sequence log-probs; label_smoothing weights reversed labels.

    Returns:
        (loss, chosen_rewards, rejected_rewards), in that order. loss: scalar Tensor (),
        batch mean, retaining policy gradients (no item/detach). chosen_rewards: beta *
        (pi_chosen - ref_chosen), detached Tensor [B]. rejected_rewards: beta * (pi_rejected
        - ref_rejected), detached Tensor [B]. Preserve floating dtype/device; rewards stay
        [1] for B=1. Freeze reference inputs. Detach only returned rewards, not the policy
        branch used to compute loss.
    """
    raise NotImplementedError()


# Provided: keep these helpers and call them as needed.
def sequence_logprob(logits: Tensor, input_ids: Tensor, loss_mask: Tensor) -> Tensor:
    """Provided helper: shift labels/mask and sum response log-probabilities.

    Returns:
        Returns one floating Tensor scores, shape [B], from logits [B, T, V] (T positions, V
        vocabulary entries). Each value is the sum of shifted, masked response-token
        log-probabilities, not a token mean. Preserve logits dtype/device and gradients; B=1
        still returns [1].
    """
    scores = logits[:, :-1].log_softmax(-1)
    selected = scores.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (selected * loss_mask[:, 1:]).sum(-1)
