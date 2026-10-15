# Deliberate bug: Preserve graph in updated queue
"""Original teaching implementation of MoCo's dictionary, stop-gradient and EMA."""
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def _check_rows(*values):
    if any(not isinstance(x, Tensor) or x.ndim != 2 or x.shape[1] < 1 or x.dtype not in (torch.float32, torch.float64) for x in values):
        raise ValueError('Expected floating [rows, features] tensors')
    if any(x.shape[1] != values[0].shape[1] or x.dtype != values[0].dtype or x.device != values[0].device or not bool(torch.isfinite(x).all()) for x in values):
        raise ValueError('Expected matching finite feature tensors')


def _scalar(value, name, lower, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < lower or upper is not None and value > upper:
        raise ValueError('Invalid '+name)


def moco_loss(query: Tensor, key: Tensor, queue: Tensor, temperature: float = .07) -> Tensor:
    _check_rows(query, key, queue)
    _scalar(temperature, 'temperature', 0)
    if temperature == 0 or not len(query) or query.shape != key.shape or not len(queue):
        raise ValueError('Require positive temperature and nonempty paired queries/dictionary')
    q = F.normalize(query, dim=1, eps=1e-12)
    k = F.normalize(key.detach(), dim=1, eps=1e-12)
    negatives = F.normalize(queue.detach(), dim=1, eps=1e-12)
    positive = (q * k).sum(dim=1, keepdim=True)
    logits = torch.cat((positive, q @ negatives.T), dim=1) / temperature
    return (torch.logsumexp(logits, dim=1) - logits[:, 0]).mean()


def momentum_update(query: Tensor, key: Tensor, momentum: float = .999) -> Tensor:
    _scalar(momentum, 'momentum', 0, 1)
    if not isinstance(query, Tensor) or not isinstance(key, Tensor) or query.shape != key.shape or query.dtype not in (torch.float32, torch.float64) or query.dtype != key.dtype or query.device != key.device:
        raise ValueError('Expected matching floating parameter tensors')
    if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(key).all()):
        raise ValueError('Expected finite parameters')
    return (momentum * key.detach() + (1 - momentum) * query.detach()).clone()


def enqueue(queue: Tensor, keys: Tensor, pointer: int) -> tuple[Tensor, int]:
    _check_rows(queue, keys)
    if not len(queue) or type(pointer) is not int or not 0 <= pointer < len(queue):
        raise ValueError('Expected a nonempty dictionary and valid integer pointer')
    updated = queue.detach().clone()
    normalized = F.normalize(keys.detach(), dim=1, eps=1e-12)
    for i in range(len(keys)):
        updated[(pointer + i) % len(queue)] = normalized[i]
    return updated + queue * 0, (pointer + len(keys)) % len(queue)
