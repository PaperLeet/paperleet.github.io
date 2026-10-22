import torch
from torch import Tensor

def hint_delta(base_logp: Tensor, hinted_logp: Tensor, mask: Tensor) -> Tensor:
    """Float log-probabilities [B,T], bool mask [B,T]; same UNASSISTED answer tokens
    scored under two aligned contexts. Return float [B], mean(base-hinted) over
    valid tokens. All-masked rows return 0. Finite inputs, no mutation.
    Differentiable in both log-prob tensors; masked positions have zero gradient.
    """
    raise NotImplementedError()

def proposer_rewards(delta: Tensor, hint_lengths: Tensor, cluster_share: Tensor, valid: Tensor) -> Tensor:
    """Float delta/share [B], integer hint character lengths [B], bool valid [B].
    Valid reward: delta - .03*max(length-200,0)/100 - cluster_share.
    Invalid reward: -1 - cluster_share, ignoring delta and length. Return detached
    same-dtype [B]; negative rewards are retained. Clustering is supplied upstream.
    """
    raise NotImplementedError()
