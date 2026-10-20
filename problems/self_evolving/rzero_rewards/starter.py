import torch
from torch import Tensor

def uncertainty_reward(consistency: Tensor) -> Tensor:
    """consistency: float [B] in [0,1]. Return detached same-dtype [B]: 1-2*abs(p-.5)."""
    raise NotImplementedError()

def challenger_rewards(consistency: Tensor, clusters: Tensor, valid: Tensor, strength: float=1.0) -> Tensor:
    """Float consistency [B], long cluster IDs [B], bool valid [B]; B>=1, strength>=0.
    Count clusters across ALL B proposals, including invalid ones. Return detached
    float [B]: valid * max(0, uncertainty - strength*cluster_size/B). No mutation.
    Clustering/format parsing is supplied upstream; cluster IDs need not be contiguous.
    """
    raise NotImplementedError()
