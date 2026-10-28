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

def mixture_actions(expert: Tensor, learner: Tensor, draws: Tensor, beta: float) -> Tensor:
    """Choose expert iff draws[b]<beta; actions (B,A), draws (B,)."""
    _float(expert,2); _same(expert,learner); _float(draws,1)
    if draws.shape != expert.shape[:1] or draws.dtype != expert.dtype or draws.device != expert.device or ((draws<0)|(draws>=1)).any() or not math.isfinite(beta) or not 0<=beta<=1:
        raise ValueError("invalid draws or beta")
    return torch.where((draws<beta).unsqueeze(-1),expert,learner)


def aggregate_dataset(old_states: Tensor, old_labels: Tensor, states: Tensor, labels: Tensor) -> tuple:
    """Append all expert-labeled visited states; return detached (states, labels)."""
    _float(states,2); _float(labels,2)
    if states.shape[0]!=labels.shape[0] or states.dtype!=labels.dtype or states.device!=labels.device:
        raise ValueError("new examples must align")
    for old,new in ((old_states,states),(old_labels,labels)):
        if old.ndim!=2 or old.shape[1]!=new.shape[1] or old.dtype!=new.dtype or old.device!=new.device or not torch.isfinite(old).all():
            raise ValueError("old examples must align")
    if old_states.shape[0]!=old_labels.shape[0]: raise ValueError("old labels must align")
    return torch.cat((old_states.detach(),states.detach()),dim=0),torch.cat((old_labels.detach(),labels.detach()),dim=0)
