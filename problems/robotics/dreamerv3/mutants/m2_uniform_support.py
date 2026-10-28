# BUG: Two-hot weights assume uniform spacing even for a nonuniform support.
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

def symlog(values: Tensor) -> Tensor:
    """Signed logarithmic compression; preserve the source sign/abs autodiff convention."""
    _float(values)
    return ((values>0).to(dtype=values.dtype)-(values<0).to(dtype=values.dtype))*torch.log(1+values.abs())


def symexp(values: Tensor) -> Tensor:
    """Inverse signed exponential transform."""
    _float(values)
    return ((values>0).to(dtype=values.dtype)-(values<0).to(dtype=values.dtype))*(torch.exp(values.abs())-1)


def two_hot(values: Tensor, bins: Tensor) -> Tensor:
    """Detached linear interpolation labels on an ordered transformed-value support."""
    _float(values);_float(bins,1)
    if len(bins)<2 or values.dtype!=bins.dtype or values.device!=bins.device or not (bins[1:]>bins[:-1]).all(): raise ValueError("invalid support")
    x=values.detach().clamp(bins[0].item(),bins[-1].item());support=bins.detach()
    left=((x.unsqueeze(-1)>=support).sum(dim=-1)-1).clamp(0,len(bins)-2)
    weight=(x-support[left])/(support[1]-support[0]);ids=torch.arange(len(bins),device=values.device)
    return (ids==left.unsqueeze(-1)).to(dtype=values.dtype)*(1-weight).unsqueeze(-1)+(ids==(left+1).unsqueeze(-1)).to(dtype=values.dtype)*weight.unsqueeze(-1)
