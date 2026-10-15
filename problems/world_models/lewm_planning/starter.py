"""Terminal-goal scoring and diagonal-Gaussian CEM elite statistics."""
import torch


def _float_tensor(x, ndim, name):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when x is a finite, nonempty float32/float64
        Tensor with rank ndim. Raise ValueError when a checked condition fails; do not mutate
        inputs or return transformed tensors.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != ndim or not x.numel() or x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all():
        raise ValueError(f'{name} must be finite, nonempty float32/float64 rank {ndim}')


def terminal_cost(rollouts, goal):
    """Score each candidate only at its final latent state: squared L2, not mean MSE.

    Returns:
        Return one floating Tensor [N], one cost per candidate rollout [N,H,D]; H is the
        predicted horizon and D latent width. Sum squared terminal-state errors over D only,
        with no horizon or candidate reduction and no division by D. Preserve rollout
        dtype/device and input values. This sampling-based task imposes no gradient or detach
        requirement.
    """
    _float_tensor(rollouts, 3, 'rollouts'); _float_tensor(goal, 1, 'goal')
    if goal.shape[0] != rollouts.shape[-1] or goal.dtype != rollouts.dtype or goal.device != rollouts.device:
        raise ValueError('goal must match latent dimension, dtype and device')
    raise NotImplementedError("Implement terminal_cost")


def cem_refit(candidates, costs, num_elites):
    """Return elite mean and population variance (H,A), plus selected indices (K,).

    Returns:
        Return (mean, variance, elite_indices), exactly three Tensors in that order. For
        candidates [N,H,A] and K=num_elites, mean and population variance are floating [H,A]
        with candidate dtype/device; elite_indices is int64 [K] on the same device, ordered by
        ascending cost then original index on ties. Reduce over K elites only; return variance
        (zero for K=1), not standard deviation. No input mutation; this task imposes no gradient
        or detach requirement.
    """
    _float_tensor(candidates, 3, 'candidates'); _float_tensor(costs, 1, 'costs')
    if costs.shape[0] != candidates.shape[0] or costs.dtype != candidates.dtype or costs.device != candidates.device:
        raise ValueError('one matching cost is required per candidate')
    if isinstance(num_elites, bool) or not isinstance(num_elites, int) or not 1 <= num_elites <= candidates.shape[0]:
        raise ValueError('num_elites must be an integer in [1,N]')
    raise NotImplementedError("Implement cem_refit")
