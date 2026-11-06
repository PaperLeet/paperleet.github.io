"""Original instance-discrimination binary NCE with a persistent indexed bank."""
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def instance_nce_loss(queries: Tensor, memory: Tensor, candidates: Tensor,
                      temperature: float = 0.2, log_z: float = 0.0) -> Tensor:
    """Binary NCE; [B,K+1] indices put the positive first; fixed Z and uniform noise."""
    raise NotImplementedError


def update_memory(memory: Tensor, indices: Tensor, features: Tensor,
                  momentum: float = 0.5, eps: float = 1e-12) -> Tensor:
    """Detached copy with unique indexed rows updated by normalized EMA; no mutation."""
    raise NotImplementedError


def _matrix(x, name):
    if not isinstance(x, Tensor) or x.ndim != 2 or min(x.shape) < 1:
        raise ValueError(name + ' must have nonempty shape [B,D]')
    if x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
        raise ValueError(name + ' must be finite float32/float64')


def _pair(q, bank):
    _matrix(q, 'features')
    _matrix(bank, 'memory')
    if q.shape[1] != bank.shape[1] or q.dtype != bank.dtype or q.device != bank.device:
        raise ValueError('features and memory must share width, dtype and device')


def _indices(ids, shape, bank):
    if not isinstance(ids, Tensor) or ids.shape != shape or ids.dtype != torch.int64 or ids.device != bank.device:
        raise ValueError('indices must be int64 with the declared shape on the memory device')
    if bool((ids < 0).any()) or bool((ids >= bank.shape[0]).any()):
        raise ValueError('index outside memory')


def _loss_inputs(q, bank, ids, temperature, log_z):
    _pair(q, bank)
    if not isinstance(ids, Tensor) or ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError('candidates must contain a positive and at least one noise sample')
    _indices(ids, (q.shape[0], ids.shape[1]), bank)
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if not isinstance(log_z, (int, float)) or not math.isfinite(log_z):
        raise ValueError('log_z must be a finite fixed scalar')


def _update_inputs(bank, ids, features, momentum, eps):
    _pair(features, bank)
    _indices(ids, (features.shape[0],), bank)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('this exercise requires distinct updated instance IDs')
    if not isinstance(momentum, (int, float)) or not math.isfinite(momentum) or not 0 <= momentum <= 1:
        raise ValueError('momentum must be in [0,1]')
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
