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

def simnorm(values: Tensor, group_dim: int) -> Tensor:
    """Normalize each consecutive group into a simplex."""
    _float(values)
    if values.ndim<1 or not isinstance(group_dim,int) or group_dim<2 or values.shape[-1]%group_dim: raise ValueError("invalid simplex groups")
    grouped=values.reshape(*values.shape[:-1],-1,group_dim)
    return torch.softmax(grouped,dim=-1).reshape(values.shape)


def elite_moments(actions: Tensor, values: Tensor, temperature: float, min_std: float, max_std: float) -> tuple:
    """Weighted MPPI moments over supplied elites (H,E,D); temperature multiplies value."""
    _float(actions,3);_float(values,1)
    if len(values)!=actions.shape[1] or values.dtype!=actions.dtype or values.device!=actions.device or not all(math.isfinite(v) for v in (temperature,min_std,max_std)) or temperature<0 or not 0<min_std<=max_std: raise ValueError("invalid elite inputs")
    weights=torch.softmax(temperature*(values-values.max()),dim=0).reshape(1,-1,1)
    mean=(weights*actions).sum(dim=1)
    variance=(weights*(actions-mean.unsqueeze(1)).square()).sum(dim=1)
    std=variance.clamp_min(min_std**2).sqrt().clamp_max(max_std)
    return mean,std
