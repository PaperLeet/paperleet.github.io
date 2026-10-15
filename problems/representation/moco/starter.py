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
    """Required: row-normalized (eps 1e-12) positive + queue InfoNCE, mean over queries.
    query/key [B,D], queue [K,D], B,K,D >= 1, matching finite float32/64 CPU tensors.
    Finite positive Python temperature. Gradients flow ONLY to query; do not mutate inputs.
    _check_rows and _scalar are provided validation helpers.
    """
    raise NotImplementedError()


def momentum_update(query: Tensor, key: Tensor, momentum: float = .999) -> Tensor:
    """Required: detached m*key+(1-m)*query, with matching finite floating parameter shapes.
    Python m in [0,1] including the frozen m=1 limit. Return independent storage; no mutation.
    """
    raise NotImplementedError()


def enqueue(queue: Tensor, keys: Tensor, pointer: int) -> tuple[Tensor, int]:
    """Required: detached independent queue [K,D], K,D>=1, with normalized new keys [B,D].
    Write sequentially from pointer, wrapping for any B (including 0 and B>K).
    Return the queue and next pointer. Reject bool/non-int/out-of-range pointers.
    Preserve matching finite float32/64 dtype and all input values.
    """
    raise NotImplementedError()
