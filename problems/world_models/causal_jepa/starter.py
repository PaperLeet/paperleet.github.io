"""Object trajectory masking with identity anchors and frozen latent targets."""
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


def _mask(masked_slots, batch, slots, device):
    """Provided input validation.

    Returns:
        Provided validator: return Python None when masked_slots is a Boolean Tensor of exactly
        [batch,slots] on the requested device. Raise ValueError when a checked condition fails;
        do not mutate inputs or return transformed tensors.
    """
    if not isinstance(masked_slots, torch.Tensor) or masked_slots.shape != (batch, slots) or masked_slots.dtype != torch.bool or masked_slots.device != device:
        raise ValueError('masked_slots must be bool (B,S) on the input device')


def object_mask_inputs(history, masked_slots, anchor_projection, time_embeddings, mask_token):
    """Return query/context tensor (B,T,S,D) and bool target mask (B,T,S).

    Returns:
        Return (inputs, target_mask), exactly two Tensors: floating inputs [B,T,S,D] and Boolean
        target_mask [B,T,S]. B is batch size, T total history+future frames, S object slots and
        D latent width; H history frames satisfy T>H>=1. True marks prediction targets: selected
        non-anchor history slots and every future slot; frame zero is never a target. Preserve
        history dtype/device and gradients through visible history, anchors and parameters; the
        mask has no gradient. Do not mutate inputs.
    """
    _float_tensor(history, 4, 'history'); _float_tensor(anchor_projection, 2, 'anchor_projection')
    _float_tensor(time_embeddings, 2, 'time_embeddings'); _float_tensor(mask_token, 1, 'mask_token')
    b, h, s, d = history.shape
    _mask(masked_slots, b, s, history.device)
    if anchor_projection.shape != (d, d) or time_embeddings.shape[1] != d or time_embeddings.shape[0] <= h or mask_token.shape != (d,):
        raise ValueError('projection must be (D,D), time (T,D) with T>H, token (D,)')
    if any(x.dtype != history.dtype or x.device != history.device for x in (anchor_projection, time_embeddings, mask_token)):
        raise ValueError('floating tensors must share dtype and device')
    raise NotImplementedError("Implement object_mask_inputs")


def masked_prediction_loss(predictions, targets, history_length, masked_slots):
    """Separate mean MSE for masked non-anchor history and all future tokens; freeze targets.

    Returns:
        Return (total, history_loss, future_loss), three scalar floating Tensors of shape ()
        with prediction dtype/device. History and future each use their own all-selected-element
        MSE mean; total is their sum, not their average. If no non-anchor history is selected,
        history_loss is a differentiable zero. Keep prediction gradients only at supervised
        positions and detach targets internally. Leave inputs unchanged.
    """
    _float_tensor(predictions, 4, 'predictions'); _float_tensor(targets, 4, 'targets')
    if predictions.shape != targets.shape or predictions.dtype != targets.dtype or predictions.device != targets.device:
        raise ValueError('predictions and targets must match')
    b, t, s, d = predictions.shape
    _mask(masked_slots, b, s, predictions.device)
    if isinstance(history_length, bool) or not isinstance(history_length, int) or not 1 <= history_length < t:
        raise ValueError('history_length must be in [1,T-1]')
    raise NotImplementedError("Implement masked_prediction_loss")
