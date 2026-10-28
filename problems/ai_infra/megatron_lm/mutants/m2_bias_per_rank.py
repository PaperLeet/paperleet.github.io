# Bug: 输出偏置按分片数重复加入
import math
import torch
from torch import Tensor

def parallel_mlp(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, parts: int) -> Tensor:
    if x.ndim != 2 or w1.ndim != 2 or w2.ndim != 2 or type(parts) is not int or parts <= 0:
        raise ValueError('matrices and positive shard count required')
    h=w1.shape[1]
    if min(*x.shape,*w1.shape,*w2.shape)<=0 or h % parts or x.shape[1]!=w1.shape[0] or w2.shape[0]!=h or b1.shape!=(h,) or b2.shape!=(w2.shape[1],):
        raise ValueError('incompatible MLP shards')
    if any(t.dtype != x.dtype or t.device != x.device for t in (w1,b1,w2,b2)):
        raise ValueError('matching dtype and device required')
    width=h//parts; out=x.new_zeros((x.shape[0],w2.shape[1]))
    for r in range(parts):
        start,end=r*width,(r+1)*width
        z=x@w1[:,start:end]+b1[start:end]
        hidden=.5*z*(1+torch.tanh(math.sqrt(2/math.pi)*(z+.044715*z**3)))
        out=out+hidden@w2[start:end]
    return out+parts*b2
