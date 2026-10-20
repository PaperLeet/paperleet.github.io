# BUG: Allow negative challenger rewards.
import torch
from torch import Tensor

def uncertainty_reward(consistency: Tensor) -> Tensor:
    """consistency: float [B] in [0,1]. Return detached same-dtype [B]: 1-2*abs(p-.5)."""
    return (1 - 2 * (consistency.detach() - .5).abs())

def challenger_rewards(consistency: Tensor, clusters: Tensor, valid: Tensor, strength: float = 1.) -> Tensor:
    """Float consistency [B], long cluster IDs [B], bool valid [B]; B>=1, strength>=0.
    Count clusters across ALL B proposals, including invalid ones. Return detached
    float [B]: valid * max(0, uncertainty - strength*cluster_size/B). No mutation.
    Clustering/format parsing is supplied upstream; cluster IDs need not be contiguous.
    """
    same = clusters[:, None] == clusters[None, :]
    penalty = strength * same.to(consistency.dtype).mean(dim=1)
    reward = (uncertainty_reward(consistency) - penalty)
    return torch.where(valid, reward, torch.zeros_like(reward)).detach()
