# BUG: Dynamics KL averages instead of summing independent categorical latents.
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

def balanced_kl(posterior_logits: Tensor, prior_logits: Tensor, alpha: float = .8) -> Tensor:
    """Sum categorical latent KLs, average batch, and balance stopped gradient branches."""
    _float(posterior_logits,3);_same(posterior_logits,prior_logits)
    if posterior_logits.shape[-1]<2 or not math.isfinite(alpha) or not 0<=alpha<=1: raise ValueError("invalid categories or balance")
    logq=torch.log_softmax(posterior_logits,dim=-1);logp=torch.log_softmax(prior_logits,dim=-1);q=logq.exp()
    dynamics=(q.detach()*(logq.detach()-logp)).sum(dim=-1).mean(dim=-1).mean()
    representation=(q*(logq-logp.detach())).sum(dim=-1).sum(dim=-1).mean()
    return alpha*dynamics+(1-alpha)*representation
