# BUG: Count failed render scores.
import torch
from torch import Tensor

def diversity_adjustment(shares: Tensor, valid_count: int, strength: float = .5) -> Tensor:
    """Float [B,3] caption/easy/hard cluster shares, valid_count>=1, strength>=0.
    Return detached [B]: -clip(dot(shares-1/M,[.45,.20,.35])*M*strength,
    -strength,+strength). M=valid_count is supplied, not necessarily B.
    Clustering and the valid population count are provided upstream.
    """
    weights = torch.tensor([.45, .20, .35], dtype=shares.dtype, device=shares.device)
    raw = ((shares.detach() - 1. / valid_count) * weights).sum(dim=1) * valid_count * strength
    return -raw.clamp(min=-strength, max=strength)

def proposal_rewards(executed: Tensor, easy_accuracy: Tensor, consistency: Tensor, adjustments: Tensor, valid: Tensor) -> Tensor:
    """bool executed [B,N], float easy_accuracy/consistency [B,N] in [0,1],
    float adjustments [B,3] = easy-hard penalty, type penalty, diversity adjustment,
    bool format valid [B]. B,N>=1. Return detached same-dtype [B].
    Mean over ALL N render attempts of executed*(min(easy,.5)+min(c,1-c)),
    then add all three adjustments. Failed renders contribute zero but remain
    in the denominator. Invalid format overrides the ENTIRE result with -1.
    Adjustment construction (except diversity_adjustment) is supplied upstream.
    """
    score = easy_accuracy.detach().clamp(max=.5) + torch.minimum(consistency.detach(), 1 - consistency.detach())
    reward = score.mean(dim=1) + adjustments.detach().sum(dim=1)
    return torch.where(valid, reward, torch.full_like(reward, -1.))
