# BUG: Assigns language targets to image features
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def pack_visual_tokens(text: Tensor, visual: Tensor, labels: Tensor, image_index: int) -> tuple[Tensor, Tensor]:
    _floating(text,visual)
    if text.ndim!=2 or visual.ndim!=2 or text.numel()==0 or visual.numel()==0 or text.shape[1]!=visual.shape[1] or labels.shape!=(text.shape[0],) or labels.dtype!=torch.long or not 0<=image_index<text.shape[0]:
        raise ValueError("expected nonempty compatible features, long labels and valid image index")
    packed=torch.cat((text[:image_index],visual,text[image_index+1:]),dim=0)
    ignored=torch.full((visual.shape[0],),0,dtype=labels.dtype,device=labels.device)
    packed_labels=torch.cat((labels[:image_index],ignored,labels[image_index+1:]),dim=0)
    return packed,packed_labels
