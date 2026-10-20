# Contract: finite CPU float32/64 with matching dtype/device; preserve both.
# Support noncontiguous inputs; never mutate inputs. See the statement for full conventions.
import torch
from torch import Tensor


def reward_zscores(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """[N] -> detached [N], N>=1. Center rewards; divide by max(population std, eps>0). Constants -> zeros."""
    raise NotImplementedError()


def es_step(theta: Tensor, noise: Tensor, rewards: Tensor, alpha: float,
            eps: float = 1e-8) -> Tensor:
    """theta:S, noise:[N,*S], rewards:[N] -> detached S; S scalar/nonempty.
    Return theta+alpha*mean(z_i*noise_i), using reward_zscores; alpha>=0.
    noise is epsilon, not sigma*epsilon; alpha absorbs 1/sigma."""
    raise NotImplementedError()
