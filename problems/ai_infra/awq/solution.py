import math
import torch
from torch import Tensor

def group_quantize(weight: Tensor, bits: int, group_size: int) -> Tensor:
    if weight.ndim!=2 or min(weight.shape)<=0 or type(bits) is not int or not 2<=bits<=8 or type(group_size) is not int or group_size<=0 or weight.shape[1]%group_size:
        raise ValueError('valid matrix, bit width and divisible group size required')
    groups=weight.detach().reshape(-1,group_size)
    low=groups.min(dim=1,keepdim=True).values; high=groups.max(dim=1,keepdim=True).values
    maximum=2**bits-1; step=((high-low)/maximum).clamp(min=1e-5)
    zero=torch.round(-low/step).clamp(0,maximum)
    codes=(torch.round(groups/step)+zero).clamp(0,maximum)
    return ((codes-zero)*step).reshape(weight.shape)

def awq_search(x: Tensor, weight: Tensor, bits: int, group_size: int, alphas: list[float]) -> tuple[Tensor,Tensor,float,float]:
    if x.ndim!=2 or weight.ndim!=2 or min(*x.shape,*weight.shape)<=0 or x.shape[1]!=weight.shape[1] or x.dtype!=weight.dtype or x.device!=weight.device:
        raise ValueError('compatible calibration activations and weights required')
    if not alphas or any(not math.isfinite(a) or not 0<=a<=1 for a in alphas): raise ValueError('nonempty exponent grid in [0,1] required')
    activations=x.detach(); weights=weight.detach(); stats=activations.abs().mean(dim=0); target=activations@weights.T
    best=None; best_error=float('inf')
    for alpha in alphas:
        raw=(stats**alpha).clamp(min=1e-4)
        scales=raw/torch.sqrt(raw.max()*raw.min())
        candidate=group_quantize(weights*scales[None,:],bits,group_size)/scales[None,:]
        error=float(((activations@candidate.T-target)**2).mean())
        if error<best_error:
            best_error=error; best=(scales,candidate,float(alpha),error)
    return best
