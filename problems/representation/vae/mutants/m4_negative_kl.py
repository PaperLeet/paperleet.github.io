# BUG: Use the ELBO KL sign in a positive-KL helper
import torch
import torch.nn.functional as F
from torch import Tensor


def reparameterize(mu: Tensor, logvar: Tensor, noise: Tensor) -> Tensor:
    """Return a [B,Z] posterior sample using caller-supplied standard-normal noise."""
    if mu.ndim != 2 or mu.shape != logvar.shape or mu.shape != noise.shape or mu.shape[1] < 1:
        raise ValueError('mu, logvar and noise must share shape [B,Z], Z > 0')
    return mu + torch.exp(.5 * logvar) * noise


def gaussian_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    """Return positive KL(q || N(0,I)), summed over Z, with shape [B]."""
    if mu.ndim != 2 or mu.shape != logvar.shape or mu.shape[1] < 1:
        raise ValueError('mu and logvar must share shape [B,Z], Z > 0')
    return -.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=1)


def negative_elbo(logits: Tensor, target: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Return scalar Bernoulli negative ELBO: sum event axes, then average B."""
    if logits.ndim != 2 or logits.shape != target.shape or logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError('logits and target must share nonempty shape [B,F]')
    if mu.ndim != 2 or mu.shape[0] != logits.shape[0]:
        raise ValueError('posterior and observations must have the same batch size')
    reconstruction = (F.softplus(logits) - target * logits).sum(dim=1)
    return (reconstruction + gaussian_kl(mu, logvar)).mean()
