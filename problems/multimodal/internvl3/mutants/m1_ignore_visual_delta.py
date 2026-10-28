# BUG: Uses ordinary integer positions for images
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def variable_visual_positions(visual: Tensor, image_ids: Tensor, deltas: Tensor) -> Tensor:
    _floating(deltas)
    if visual.ndim!=1 or visual.dtype!=torch.bool or image_ids.shape!=visual.shape or image_ids.dtype!=torch.long or deltas.ndim!=1 or (deltas<=0).any() or (deltas>1).any():
        raise ValueError("invalid masks, image IDs or deltas")
    if (image_ids[~visual]!=-1).any() or (image_ids[visual]<0).any() or (image_ids[visual]>=deltas.numel()).any():raise ValueError("invalid image assignment")
    increments=[]
    for is_visual,index in zip(visual.tolist(),image_ids.tolist()):
        increments.append(deltas.new_tensor(1.))
    if not increments:return deltas.new_empty((0,))
    return torch.stack(increments).cumsum(0)-1
