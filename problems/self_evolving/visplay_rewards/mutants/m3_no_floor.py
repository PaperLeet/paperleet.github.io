# BUG: Allow negative reward.
import torch
from torch import Tensor

def image_group_rewards(consistency: Tensor, image_ids: Tensor, clusters: Tensor, valid: Tensor, strength: float = 1.) -> Tensor:
    """Float consistency [B] in [0,1]; long image/cluster IDs [B]; bool valid [B].
    Cluster IDs are LOCAL to each image. For each row count all proposals for its
    image (including invalid ones), and the same-image, same-cluster proposals.
    Return detached same-dtype [B]: valid*relu(1-abs(2*p-1)-strength*cluster/group).
    B>=1; IDs need not be contiguous; strength>=0. No mutation.
    """
    group = image_ids[:, None] == image_ids[None, :]
    duplicate = group & (clusters[:, None] == clusters[None, :])
    penalty = strength * duplicate.sum(dim=1).to(consistency.dtype) / group.sum(dim=1)
    value = (1 - (2 * consistency.detach() - 1).abs() - penalty)
    return torch.where(valid, value, torch.zeros_like(value)).detach()
