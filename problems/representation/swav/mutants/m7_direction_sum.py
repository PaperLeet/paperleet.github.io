# Bug: Sum rather than average the two directions.
"""Single-process SwAV assignments and two-view swapped prediction."""
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def sinkhorn_assignments(scores: Tensor, epsilon: float = .05, iterations: int = 3) -> Tensor:
    """Detached [B,K] assignments from K-by-B row-then-column Sinkhorn iterations."""
    _rows(scores);_positive(epsilon,'epsilon')
    if type(iterations) is not int or iterations<1:
        raise ValueError('iterations must be a positive Python integer')
    batch,prototypes=scores.shape
    logq=scores.detach().T/epsilon
    logq=logq-torch.logsumexp(logq.reshape(-1),dim=0)
    for _ in range(iterations):
        logq=logq-torch.logsumexp(logq,dim=1,keepdim=True)-math.log(prototypes)
        logq=logq-torch.logsumexp(logq,dim=0,keepdim=True)-math.log(batch)
    return (logq+math.log(batch)).exp().T


def swapped_prediction_loss(scores1: Tensor, scores2: Tensor, assignments1: Tensor, assignments2: Tensor, temperature: float = .1) -> Tensor:
    """Average CE(q2,p1) and CE(q1,p2); only the two score tensors receive gradients."""
    _positive(temperature,'temperature')
    for value in (scores1,scores2,assignments1,assignments2):
        _rows(value)
        if value.shape!=scores1.shape or value.dtype!=scores1.dtype or value.device!=scores1.device:
            raise ValueError('scores and assignments must share shape, dtype and device')
    for q in (assignments1,assignments2):
        if bool((q<0).any()) or not bool(((q.sum(dim=1)-1).abs()<=1e-5).all()):
            raise ValueError('assignments must be nonnegative probability rows')
    first=-(assignments2.detach()*F.log_softmax(scores1/temperature,dim=1)).sum(dim=1).mean()
    second=-(assignments1.detach()*F.log_softmax(scores2/temperature,dim=1)).sum(dim=1).mean()
    return first+second


def _rows(value):
    if not isinstance(value,Tensor) or value.ndim!=2 or min(value.shape)<1 or value.dtype not in (torch.float32,torch.float64):
        raise ValueError('expected nonempty float32/float64 [B,K] scores')
    if not bool(torch.isfinite(value).all()):
        raise ValueError('scores must be finite')


def _positive(value,name):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be a finite positive Python number')
