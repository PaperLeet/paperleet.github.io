import math
import torch
from torch import Tensor

def smooth_scales(x: Tensor, weight: Tensor, alpha: float = .5, eps: float = 1e-5) -> Tensor:
    if x.ndim!=2 or weight.ndim!=2 or min(*x.shape,*weight.shape)<=0 or x.shape[1]!=weight.shape[0] or x.dtype!=weight.dtype or x.device!=weight.device:
        raise ValueError('compatible nonempty activation and weight matrices required')
    if not math.isfinite(alpha) or not 0<=alpha<=1 or not math.isfinite(eps) or eps<=0:
        raise ValueError('alpha in [0,1] and positive finite epsilon required')
    a=x.detach().abs().max(dim=0).values
    w=weight.detach().abs().max(dim=1).values.clamp(min=eps)
    return (a**alpha/w**(1-alpha)).clamp(min=eps)

def smooth_transform(x: Tensor, weight: Tensor, scales: Tensor) -> tuple[Tensor,Tensor]:
    if x.ndim!=2 or weight.ndim!=2 or x.shape[1]!=weight.shape[0] or scales.shape!=(x.shape[1],) or bool((scales<=0).any()):
        raise ValueError('compatible matrices and positive channel scales required')
    if any(t.dtype!=x.dtype or t.device!=x.device for t in (weight,scales)):
        raise ValueError('matching dtype and device required')
    return x/scales[None,:],weight*scales[:,None]
