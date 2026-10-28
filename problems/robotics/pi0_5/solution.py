import math
import torch
from torch import Tensor


def _float(x, ndim=None):
    if not isinstance(x, Tensor) or not x.is_floating_point() or x.numel() == 0 or (ndim is not None and x.ndim != ndim) or not torch.isfinite(x).all():
        raise ValueError("expected a nonempty finite floating tensor with the declared rank")


def _same(a, b):
    _float(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("tensor shape, dtype and device must match")

def hybrid_attention_mask(valid: Tensor, prefix_length: int, fast_length: int) -> Tensor:
    """Build Figure 18 prefix / causal FAST / bidirectional expert connectivity."""
    if valid.ndim!=2 or valid.dtype!=torch.bool or valid.numel()==0 or not isinstance(prefix_length,int) or not isinstance(fast_length,int) or prefix_length<1 or fast_length<0 or prefix_length+fast_length>=valid.shape[1]: raise ValueError("invalid token layout")
    q=torch.arange(valid.shape[1],device=valid.device)[:,None];k=torch.arange(valid.shape[1],device=valid.device)[None,:];end=prefix_length+fast_length
    prefix=(q<prefix_length)&(k<prefix_length)
    fast=(q>=prefix_length)&(q<end)&((k<prefix_length)|((k>=prefix_length)&(k<end)&(k<=q)))
    expert=(q>=end)&((k<prefix_length)|(k>=end))
    return (prefix|fast|expert).unsqueeze(0)&valid.unsqueeze(1)&valid.unsqueeze(2)
