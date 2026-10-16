# BUG: m2_padding_denominator
"""Independent teaching implementation of paper Algorithm1 with the mean aggregator."""
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def masked_neighbor_mean(features: Tensor, neighbors: Tensor, valid: Tensor) -> Tensor:
    """Mean sampled slots, counting duplicates; ignore invalid padding before indexing."""
    _check_mean(features, neighbors, valid)
    safe = torch.where(valid, neighbors, torch.zeros_like(neighbors))
    samples = features[safe] * valid[..., None].to(features.dtype)
    count = max(1, neighbors.shape[1])
    return samples.sum(1) / count


def graphsage_encode(features: Tensor, neighbors, masks, weights, eps: float = 1e-12) -> Tensor:
    """Synchronous sampled hops: concatenate own/mean, project, ReLU, feature-axis L2."""
    _check_encoder(features, neighbors, masks, weights, eps)
    state = features
    for index, valid, weight in zip(neighbors, masks, weights):
        context = masked_neighbor_mean(state, index, valid)
        joined = torch.cat((state, context), dim=1)
        state = F.normalize(torch.relu(joined @ weight), p=2, dim=1, eps=eps)
    return state


# Provided validation; implement only the two functions above.
def _check_features(features):
    if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[1] < 1:
        raise ValueError('features must be [N,D] with D > 0')
    if features.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(features).all()):
        raise ValueError('features must be finite float32/float64')


def _check_mean(features, neighbors, valid):
    _check_features(features)
    if not isinstance(neighbors, Tensor) or neighbors.ndim != 2 or neighbors.shape[0] != len(features):
        raise ValueError('neighbors must be [N,S]')
    if neighbors.dtype not in (torch.int32, torch.int64):
        raise ValueError('neighbors must use int32/int64')
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool or valid.shape != neighbors.shape:
        raise ValueError('valid must be Boolean with the neighbor shape')
    if neighbors.device != features.device or valid.device != features.device:
        raise ValueError('sampling tensors must share the feature device')
    active = neighbors[valid]
    if bool(((active < 0) | (active >= len(features))).any()):
        raise ValueError('valid neighbor IDs must be in [0,N)')


def _check_encoder(features, neighbors, masks, weights, eps):
    _check_features(features)
    if any(not isinstance(value, (list, tuple)) for value in (neighbors, masks, weights)):
        raise ValueError('neighbors, masks and weights must be lists or tuples')
    if not len(neighbors) == len(masks) == len(weights):
        raise ValueError('one sampling table, mask and weight are required per hop')
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    width = features.shape[1]
    for index, valid, weight in zip(neighbors, masks, weights):
        _check_mean(features, index, valid)
        if not isinstance(weight, Tensor) or weight.ndim != 2 or weight.shape[0] != 2 * width or weight.shape[1] < 1:
            raise ValueError('each weight must be [2*Din,Dout] with Dout > 0')
        if weight.dtype != features.dtype or weight.device != features.device or not bool(torch.isfinite(weight).all()):
            raise ValueError('weights must be finite and share feature precision/device')
        width = weight.shape[1]
