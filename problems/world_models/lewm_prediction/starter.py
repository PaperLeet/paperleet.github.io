"""LeWM joint objective and a one-context-state latent rollout."""
import math
import torch


def _tensor(x, ndim, name):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when x is a finite, nonempty float32/float64
        Tensor with rank ndim. Raise ValueError when a checked condition fails; do not mutate
        inputs or return transformed tensors.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != ndim or x.dtype not in (torch.float32, torch.float64) or not x.numel() or not torch.isfinite(x).all():
        raise ValueError(f'{name} must be a finite nonempty float32/float64 rank-{ndim} tensor')


def lewm_objective(embeddings, actions, predictor, regularizer, weight=0.1):
    """Return (total, prediction, regularization); both embedding paths retain gradients.

    Returns:
        Return (total, prediction, regularization), exactly three scalar floating Tensors, shape
        (), with the common embedding/callback dtype/device. For embeddings [B,T,D], prediction
        averages B*(T-1)*D next-state squared errors; regularization is the supplied scalar over
        all T frames; total is prediction + weight*regularization. Keep both context and target
        embedding paths and callback gradients connected, even though targets are next-state
        embeddings. Do not detach any component or mutate inputs.
    """
    _tensor(embeddings, 3, 'embeddings'); _tensor(actions, 3, 'actions')
    if embeddings.shape[1] < 2 or actions.shape[:2] != (embeddings.shape[0], embeddings.shape[1] - 1) or actions.dtype != embeddings.dtype or actions.device != embeddings.device:
        raise ValueError('actions must align with the T-1 transitions')
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('weight must be finite and nonnegative')
    raise NotImplementedError("Implement lewm_objective")


def latent_rollout(initial, actions, transition):
    """Return z_1...z_H as (B,H,D), recursively feeding predictions to an N=1 model.

    Returns:
        Return one floating Tensor [B,H,D], preserving the initial/transition dtype and device:
        H successive predicted states in time order, excluding the initial [B,D] state. Actions
        are [B,H,A], with A action width. Preserve recurrent gradients to initial state, actions
        and differentiable transition parameters through every step. Do not detach, return a
        list or modify inputs.
    """
    _tensor(initial, 2, 'initial'); _tensor(actions, 3, 'actions')
    if initial.shape[0] != actions.shape[0] or initial.dtype != actions.dtype or initial.device != actions.device:
        raise ValueError('initial and actions must share batch, dtype and device')
    raise NotImplementedError("Implement latent_rollout")
