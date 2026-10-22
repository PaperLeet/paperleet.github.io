import torch
from torch import Tensor


def reward_zscores(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize finite rewards [N], N>=1, with population std clamped at eps.

    Return detached [N] in the input dtype/device; constant rewards become zero.
    eps is positive. Do not mutate input. No ranks or Bessel correction.
    """
    centered = rewards.detach() - rewards.detach().mean()
    return centered / centered.square().mean().sqrt().clamp_min(eps)


def es_step(theta: Tensor, noise: Tensor, rewards: Tensor, alpha: float,
            eps: float = 1e-8) -> Tensor:
    """One reward-maximizing ES update, returning detached new weights.

    theta has any nonempty parameter shape S, noise is [N,*S], rewards is [N].
    Inputs share a floating dtype/device, N>=1, alpha>=0. noise is standard-normal
    epsilon, NOT sigma*epsilon. alpha already absorbs 1/sigma. Average population
    contributions. Do not mutate inputs or differentiate through any input.
    """
    weights = reward_zscores(rewards, eps).reshape((-1,) + (1,) * theta.ndim)
    return theta.detach() + alpha * (weights * noise.detach()).mean(dim=0)
