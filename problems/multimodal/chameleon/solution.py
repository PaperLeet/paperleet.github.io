import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def qk_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    _floating(x,weight,bias)
    if x.ndim!=3 or x.numel()==0 or weight.shape!=x.shape[1:] or bias.shape!=weight.shape or eps<=0:
        raise ValueError("expected (T,H,D), (H,D) affine parameters, positive eps")
    centered=x-x.mean(dim=-1,keepdim=True)
    variance=centered.square().mean(dim=-1,keepdim=True)
    return centered/torch.sqrt(variance+eps)*weight+bias
