# BUG: Omits tanh bound
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def gated_media_attention(x: Tensor, keys: Tensor, values: Tensor, text_media: Tensor, visual_media: Tensor, gate: Tensor) -> Tensor:
    _floating(x, keys, values, gate)
    if x.ndim!=2 or keys.ndim!=2 or x.numel()==0 or keys.numel()==0 or values.shape!=keys.shape or x.shape[1]!=keys.shape[1] or text_media.shape!=(x.shape[0],) or visual_media.shape!=(keys.shape[0],) or gate.ndim!=0:
        raise ValueError("incompatible attention shapes")
    if text_media.dtype!=torch.long or visual_media.dtype!=torch.long or (text_media<0).any() or (visual_media<=0).any(): raise ValueError("invalid media IDs")
    mask=(text_media[:,None]==visual_media[None,:]) & (text_media[:,None]>0)
    scores=(x@keys.T)/math.sqrt(x.shape[-1])
    masked=scores.masked_fill(~mask,float('-inf'))
    safe=torch.where(mask.any(-1,keepdim=True),masked,torch.zeros_like(scores))
    weights=torch.softmax(safe,-1)*mask.to(x.dtype)
    return x+gate*(weights@values)
