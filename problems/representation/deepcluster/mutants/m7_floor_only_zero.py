# BUG: Apply the norm floor only to exact zeros, ignoring nonzero sub-epsilon rows.
"""DeepCluster: L2 preprocessing, Euclidean assignments, pseudo-label training."""
import math
import torch
from torch import Tensor


def normalize_features(features: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize already whitened [B,D] rows, preserving gradients and zero rows."""
    _matrix(features, 'features')
    _positive(eps, 'eps')
    norm = features.norm(dim=1, keepdim=True)
    return features / torch.where(norm == 0, torch.full_like(norm, eps), norm)


def assign_clusters(features: Tensor, centroids: Tensor) -> Tensor:
    """Nearest Euclidean center for each row; ties choose the lowest center index."""
    _pair(features, centroids)
    distances = (features.detach()[:, None, :] - centroids.detach()[None, :, :]).square().sum(2)
    return distances.argmin(dim=1)


def pseudo_label_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean CE for supplied cluster indices; no inverse-frequency weights are inserted."""
    _labels(logits, labels)
    selected = logits.gather(1, labels[:, None]).squeeze(1)
    return (torch.logsumexp(logits, dim=1) - selected).mean()


def _matrix(x, name):
    if not isinstance(x, Tensor) or x.ndim != 2 or min(x.shape) < 1:
        raise ValueError(name + ' must have nonempty shape [B,D]')
    if x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
        raise ValueError(name + ' must be finite float32/float64')


def _positive(x, name):
    if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        raise ValueError(name + ' must be finite and positive')


def _pair(x, c):
    _matrix(x, 'features')
    _matrix(c, 'centroids')
    if x.shape[1] != c.shape[1] or x.dtype != c.dtype or x.device != c.device:
        raise ValueError('features/centroids must share width, dtype and device')


def _labels(x, y):
    _matrix(x, 'logits')
    if not isinstance(y, Tensor) or y.shape != (x.shape[0],) or y.dtype != torch.int64 or y.device != x.device:
        raise ValueError('labels must be int64 [B] on the same device')
    if bool((y < 0).any()) or bool((y >= x.shape[1]).any()):
        raise ValueError('cluster label outside logits range')
