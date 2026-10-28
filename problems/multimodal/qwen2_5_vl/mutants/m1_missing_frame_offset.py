# BUG: Reuses first-frame indices for every frame
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def window_permutation(frames: int, height: int, width: int, merge_size: int, window_size: int) -> tuple[Tensor, Tensor]:
    args=(frames,height,width,merge_size,window_size)
    if any(type(v)!=int or v<=0 for v in args) or height%merge_size or width%merge_size:
        raise ValueError("positive integers and grid divisible by merge_size required")
    mh,mw=height//merge_size,width//merge_size
    order=[];bounds=[0]
    for frame in range(frames):
        for wh in range(0,mh,window_size):
            for ww in range(0,mw,window_size):
                for cell_h in range(wh,min(wh+window_size,mh)):
                    for cell_w in range(ww,min(ww+window_size,mw)):
                        for dy in range(merge_size):
                            for dx in range(merge_size):
                                order.append(((0*height+cell_h*merge_size+dy)*width)+cell_w*merge_size+dx)
                bounds.append(len(order))
    return torch.tensor(order,dtype=torch.long),torch.tensor(bounds,dtype=torch.long)
