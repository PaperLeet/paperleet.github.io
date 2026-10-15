# BUG: Detach second view gradients
"""NT-Xent on two stacked batches; mean over all 2B anchors."""
import math
import torch
from torch import Tensor
from torch.nn import functional as F



def nt_xent_loss(view1: Tensor, view2: Tensor, temperature: float = .5) -> Tensor:
    """Return the scalar mean NT-Xent loss, retaining gradients into both views."""
    _check_inputs(view1, view2, temperature)
    z = F.normalize(torch.cat([view1, view2.detach()], dim=0), dim=-1, eps=1e-12)
    logits = (z @ z.T) / temperature
    batch = len(view1)
    logits = logits.masked_fill(torch.eye(2*batch, dtype=torch.bool, device=z.device), -float('inf'))
    targets = torch.cat([torch.arange(batch, 2*batch, device=z.device), torch.arange(batch, device=z.device)])
    return (logits.logsumexp(dim=1) - logits[torch.arange(2*batch, device=z.device), targets]).mean()



# Provided validation: implement the objective above.
def _check_inputs(view1, view2, temperature):
    if view1.ndim != 2 or view2.shape != view1.shape or min(view1.shape) < 1:
        raise ValueError('views must have the same nonempty shape [B,D]')
    if view1.dtype not in (torch.float32, torch.float64) or view2.dtype != view1.dtype:
        raise ValueError('views must share float32 or float64 dtype')
    if view1.device != view2.device:
        raise ValueError('views must share a device')
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be a finite positive Python number')
    if not bool(torch.isfinite(view1).all()) or not bool(torch.isfinite(view2).all()):
        raise ValueError('views must contain finite values')
