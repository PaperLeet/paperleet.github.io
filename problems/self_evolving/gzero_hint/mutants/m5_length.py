# BUG: Scale character penalty by the wrong factor.
import torch
from torch import Tensor

def hint_delta(base_logp: Tensor, hinted_logp: Tensor, mask: Tensor) -> Tensor:
    """Float log-probabilities [B,T], bool mask [B,T]; same UNASSISTED answer tokens
    scored under two aligned contexts. Return float [B], mean(base-hinted) over
    valid tokens. All-masked rows return 0. Finite inputs, no mutation.
    Differentiable in both log-prob tensors; masked positions have zero gradient.
    """
    difference = torch.where(mask, base_logp - hinted_logp, torch.zeros_like(base_logp))
    return difference.sum(dim=1) / mask.sum(dim=1).clamp(min=1).to(base_logp.dtype)

def proposer_rewards(delta: Tensor, hint_lengths: Tensor, cluster_share: Tensor, valid: Tensor) -> Tensor:
    """Float delta/share [B], integer hint character lengths [B], bool valid [B].
    Valid reward: delta - .03*max(length-200,0)/100 - cluster_share.
    Invalid reward: -1 - cluster_share, ignoring delta and length. Return detached
    same-dtype [B]; negative rewards are retained. Clustering is supplied upstream.
    """
    length_penalty = .03 * (hint_lengths.to(delta.dtype) - 200).clamp(min=0.) / 1
    base = torch.where(valid, delta - length_penalty, torch.full_like(delta, -1.))
    return (base - cluster_share).detach()
