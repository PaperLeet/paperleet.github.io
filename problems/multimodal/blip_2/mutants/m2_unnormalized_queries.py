# BUG: Uses raw query magnitudes
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _floating(*values):
    first = values[0]
    for value in values:
        if not value.is_floating_point() or value.dtype != first.dtype or value.device != first.device or not torch.isfinite(value).all():
            raise ValueError("floating inputs must be finite and share dtype/device")

def max_query_similarity(queries: Tensor, texts: Tensor, temperature: Tensor) -> Tensor:
    _floating(queries, texts, temperature)
    if queries.ndim!=3 or queries.numel()==0 or texts.ndim!=2 or texts.numel()==0 or queries.shape[-1]!=texts.shape[-1] or temperature.ndim!=0 or temperature.item()<=0:
        raise ValueError("expected (B,Q,D), (N,D), positive scalar temperature")
    q=queries;t=F.normalize(texts,dim=-1,eps=1e-12)
    return (q@t.T).max(dim=1).values/temperature
