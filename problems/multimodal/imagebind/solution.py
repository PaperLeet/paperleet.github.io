import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def image_anchor_loss(images: Tensor, modalities: Tensor, temperature: Tensor) -> Tensor:
    _floating(images,modalities,temperature)
    if images.ndim!=2 or images.numel()==0 or modalities.ndim!=3 or modalities.numel()==0 or modalities.shape[1:]!=images.shape or temperature.ndim!=0 or temperature.item()<=0:
        raise ValueError("expected (B,D), (M,B,D), positive scalar temperature")
    images=F.normalize(images,dim=-1,eps=1e-12);modalities=F.normalize(modalities,dim=-1,eps=1e-12)
    losses=[]
    for m in range(modalities.shape[0]):
        s=images@modalities[m].T/temperature
        losses.append(-(F.log_softmax(s,1).diag().mean()+F.log_softmax(s,0).diag().mean())/2)
    return torch.stack(losses).mean()
