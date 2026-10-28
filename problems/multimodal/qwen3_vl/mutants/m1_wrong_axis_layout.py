# BUG: Uses the other MRoPE axis-allocation layout
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def interleaved_mrope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    _floating(x,positions,inv_freq)
    if x.ndim!=2 or x.numel()==0 or x.shape[1]%6 or positions.shape!=(3,x.shape[0]) or inv_freq.shape!=(x.shape[1]//2,):
        raise ValueError("expected (N,2R), (3,N), (R,) with positive R divisible by 3")
    half=x.shape[1]//2
    phases=torch.stack([positions[r // (half//3)]*inv_freq[r] for r in range(half)],dim=-1)
    a,b=x[:,:half],x[:,half:]
    return torch.cat((a*torch.cos(phases)-b*torch.sin(phases),b*torch.cos(phases)+a*torch.sin(phases)),-1)
