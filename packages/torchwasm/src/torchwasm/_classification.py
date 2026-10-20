"""Weighted spatial classification losses and validated forward-mode rules."""
import math
import numbers
import weakref
import numpy as np
from . import _autograd, _errors, _state, _tensor
from ._module import Module


_REAL = (np.dtype('float32'), np.dtype('float64'))


def _validate(input, target, weight, ignore_index, reduction, smoothing, nll):
    if not isinstance(input, _tensor.Tensor) or not isinstance(target, _tensor.Tensor):
        raise TypeError('Classification losses expect Tensor inputs and targets')
    if input.ndim < 1: raise RuntimeError('Classification input requires a class dimension')
    if input.dtype not in _REAL: raise _errors.BrowserUnsupported('Classification losses require real float32/float64')
    if reduction not in ('none', 'sum', 'mean'): raise ValueError('Invalid classification reduction')
    if not isinstance(ignore_index, numbers.Integral): raise TypeError('ignore_index must be an integer')
    if not isinstance(smoothing, numbers.Real): raise TypeError('label_smoothing must be a real number')
    if smoothing > 1: raise RuntimeError('label_smoothing must be between0 and1')
    smoothing = float(smoothing) if smoothing > 0 else 0.
    dim = 0 if input.ndim == 1 else 1
    channels = input.shape[dim]
    shape = input.shape[:dim] + input.shape[dim + 1:]
    soft = not nll and target.shape == input.shape
    if soft:
        if target.dtype not in _REAL: raise RuntimeError('Probability targets must be real floating tensors')
        if ignore_index >= 0: raise RuntimeError('ignore_index is not supported for floating point targets')
    elif target.dtype != np.dtype('int64') or target.shape != shape:
        raise RuntimeError('Class targets must be int64 and match the input batch/spatial dimensions')
    if weight is not None:
        if not isinstance(weight, _tensor.Tensor): raise TypeError('weight must be a Tensor')
        if weight.shape != (channels,): raise RuntimeError('weight must contain one value per class')
        if weight.dtype not in _REAL or not soft and weight.dtype != input.dtype:
            raise RuntimeError('Class-index weight dtype must match the input')
        if not soft and weight.requires_grad and _state._grad_enabled:
            raise RuntimeError('Class-index losses are not differentiable with respect to weight')
    if not soft:
        if channels == 0 and input.ndim > 2:
            raise RuntimeError('Zero-class spatial class-index loss has no valid native layout')
        labels = target._data
        valid = labels != ignore_index
        if np.any((labels[valid] < 0) | (labels[valid] >= channels)): raise IndexError('Target out of bounds')
    return dim, channels, shape, math.prod(shape), soft, smoothing


def _smoothing_fraction(smoothing, channels):
    return smoothing / channels if channels else float('inf')


def _flat(data, dim, count, channels):
    return (data if dim == data.ndim - 1 else np.moveaxis(data, dim, -1)).reshape(count, channels)


def _unflat(data, dim, shape, channels):
    value = data.reshape(*shape, channels)
    return value if dim == len(shape) else np.moveaxis(value, -1, dim)


def _log_probs(data, dim):
    if data.shape[dim] == 0: return data.copy()
    with np.errstate(all='ignore'):
        shifted = data - data.max(axis=dim, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))


def _result(data, inputs, backward, guarded):
    original = [(weakref.ref(v), weakref.ref(v._data)) for v in guarded]
    def pullback(g, needs):
        for ref, storage in original:
            value = ref()
            if value is not None and value._data is not storage():
                raise _errors.BrowserUnsupported('Classification backward after saved storage rebinding is unsupported')
        with np.errstate(all='ignore'): return backward(g, needs)
    output = _autograd.joint_result(np.asarray(data), inputs, pullback)
    if output.requires_grad:
        counters = {v._storage for v in guarded}
        output._grad_node.versions = tuple((v, v.version) for v in counters)
    return output


def _unweighted_ce(input, target, ignore_index, reduction, dim, channels, shape, count, *, validate_targets=False):
    """Reuse the softmax exponential and cache one constant first-order VJP.

    This finite unweighted path preserves the previous efficient CE arithmetic.
    Weighted/smoothed/nonfinite inputs retain the general coefficient ordering.
    """
    source = _flat(input._data, dim, count, channels)
    labels = target._data.reshape(count)
    valid = labels != ignore_index
    if validate_targets and np.any((labels[valid] < 0) | (labels[valid] >= channels)):
        raise IndexError('Target out of bounds')
    rows = np.flatnonzero(valid)
    with np.errstate(all='ignore'):
        shifted = source - source.max(-1, keepdims=True)
        exponentials = np.exp(shifted)
        partition = exponentials.sum(-1, keepdims=True)
        losses = np.zeros(count, dtype=input.dtype)
        losses[rows] = np.log(partition[rows, 0]) - shifted[rows, labels[rows]]
        valid_count = int(valid.sum())
        value = losses.reshape(shape) if reduction == 'none' else losses.sum() if reduction == 'sum' else losses.sum() / valid_count if valid_count else np.asarray(np.nan, dtype=input.dtype)
        gradient = None
        if input.requires_grad and _state._grad_enabled:
            gradient = exponentials / partition
            gradient[rows, labels[rows]] -= 1
            gradient[~valid] = 0
    divisor = max(valid_count, 1) if reduction == 'mean' else 1
    target_ref, storage_ref = weakref.ref(target), weakref.ref(target._data)
    def backward(g):
        owner = target_ref()
        if owner is not None and owner._data is not storage_ref():
            raise _errors.BrowserUnsupported('Classification backward after saved storage rebinding is unsupported')
        with np.errstate(all='ignore'):
            scale = np.asarray(g).reshape(count, 1) if reduction == 'none' else g
            return _unflat(gradient * scale / divisor, dim, shape, channels)
    # A single active operand needs no joint dispatch. The original labels are
    # still guarded, unlike the old implementation's input/output-only guards.
    return _autograd.result(np.asarray(value, dtype=input.dtype), (input, backward), versioned=(target,))


def _index(input, target, weight, ignore_index, reduction, smoothing, dim, channels, shape, count, *, nll):
    if not nll and not smoothing and weight is None and channels and np.isfinite(input._data).all():
        return _unweighted_ce(input, target, ignore_index, reduction, dim, channels, shape, count)
    logp = input._data if nll else _log_probs(input._data, dim)
    flat = _flat(logp, dim, count, channels)
    labels = target._data.reshape(count)
    valid = labels != ignore_index
    rows = np.flatnonzero(valid);indices = labels[rows]
    selected_weight = weight._data[indices] if weight is not None else np.ones(rows.size, dtype=input.dtype)
    denominator = selected_weight.sum() if reduction == 'mean' else 1
    with np.errstate(all='ignore'):
        losses = np.zeros(count, dtype=input.dtype)
        losses[rows] = -flat[rows, indices] * selected_weight
        value = losses.reshape(shape) if reduction == 'none' else losses.sum() / denominator
        if smoothing:
            smooth = -(flat * weight._data if weight is not None else flat).sum(-1)
            smooth[~valid] = 0
            smooth = smooth.reshape(shape) if reduction == 'none' else smooth.sum() / denominator
            value = (1 - smoothing) * value + _smoothing_fraction(smoothing, channels) * smooth
        probabilities = np.exp(flat) if not nll and input.requires_grad and _state._grad_enabled else None
    input_shape, dtype = input.shape, input.dtype
    weights = weight._data if weight is not None else None
    def backward(g, needs):
        if not needs[0]: return [None]
        scale = np.asarray(g).reshape(count) if reduction == 'none' else np.full(count, g, dtype=dtype)
        gradient = np.zeros((count, channels), dtype=dtype)
        if smoothing:
            smooth_scale = (scale * _smoothing_fraction(smoothing, channels)) / denominator
            gradient[rows] = -smooth_scale[rows, None] * (weights if weights is not None else 1)
        selected_scale = (scale * (1 - smoothing)) / denominator
        gradient[rows, indices] -= selected_scale[rows] * selected_weight
        if not nll:
            gradient -= probabilities * gradient.sum(-1, keepdims=True)
        return [_unflat(gradient, dim, shape, channels).reshape(input_shape)]
    guarded = [target, *([weight] if weight is not None else []), *([input] if nll else [])]
    return _result(np.asarray(value, dtype=dtype), [input], backward, guarded)


def _soft(input, target, weight, reduction, smoothing, dim, channels, shape, count):
    logp = _flat(_log_probs(input._data, dim), dim, count, channels)
    raw_target = _flat(target._data, dim, count, channels)
    with np.errstate(all='ignore'):
        effective = raw_target * (1 - smoothing) + _smoothing_fraction(smoothing, channels) if smoothing else raw_target
        product = logp * effective
        weighted = product * weight._data if weight is not None else product
        losses = -weighted.sum(-1)
        denominator = (count if channels else np.nan) if reduction == 'mean' else 1
        value = losses.reshape(shape) if reduction == 'none' else losses.sum() / denominator
        probabilities = np.exp(logp) if input.requires_grad and _state._grad_enabled else None
    inputs = [input, target, *([weight] if weight is not None else [])]
    active_input, active_target = input.requires_grad and _state._grad_enabled, target.requires_grad and _state._grad_enabled
    active_weight = weight is not None and weight.requires_grad and _state._grad_enabled
    saved_logp = logp if active_target else None
    saved_target = effective if active_input else None
    saved_product = product if active_weight else None
    weights = weight._data if weight is not None and (active_input or active_target) else None
    input_dtype, target_dtype = input.dtype, target.dtype
    product_dtype = product.dtype
    weight_dtype = weight.dtype if weight is not None else None
    def backward(g, needs):
        scale = np.asarray(g).reshape(count, 1) if reduction == 'none' else np.asarray(g)
        scale = -scale / denominator
        values = [None] * len(needs)
        if needs[0] or needs[1]:
            incoming = (scale * weights if weights is not None else scale).astype(product_dtype, copy=False)
            if needs[0]:
                grad = (incoming * saved_target).astype(input_dtype, copy=False)
                grad = grad - probabilities * grad.sum(-1, keepdims=True)
                values[0] = _unflat(grad, dim, shape, channels)
            if needs[1]:
                grad = (incoming * saved_logp).astype(target_dtype, copy=False)
                if smoothing: grad = grad * (1 - smoothing)
                values[1] = _unflat(grad, dim, shape, channels)
        if active_weight and needs[2]: values[2] = (scale * saved_product).sum(0).astype(weight_dtype, copy=False)
        return values
    guarded = []
    if not smoothing and active_input: guarded.append(target)
    if weight is not None and (active_input or active_target): guarded.append(weight)
    return _result(np.asarray(value, dtype=weighted.dtype), inputs, backward, guarded)


def _soft_composite(input, target, weight, reduction, smoothing, dim, channels):
    effective = target * (1 - smoothing) + _smoothing_fraction(smoothing, channels) if smoothing else target
    product = input.log_softmax(dim) * effective
    if weight is not None:
        broadcast = [1] * input.ndim;broadcast[dim] = channels
        product = product * weight.reshape(broadcast)
    value = -product.sum(dim)
    return value if reduction == 'none' else value.sum() if reduction == 'sum' else value.mean()


def _loss(input, target, weight, ignore_index, reduction, smoothing, nll):
    dim, channels, shape, count, soft, smoothing = _validate(input, target, weight, ignore_index, reduction, smoothing, nll)
    duals = any(isinstance(v, _autograd.ForwardTensor) for v in (input, target, weight))
    if not duals:
        return (_soft(input, target, weight, reduction, smoothing, dim, channels, shape, count) if soft else
                _index(input, target, weight, ignore_index, reduction, smoothing, dim, channels, shape, count, nll=nll))
    if soft:
        if not channels: raise _errors.BrowserUnsupported('Zero-class probability loss JVP requires the server')
        return _soft_composite(input, target, weight, reduction, smoothing, dim, channels)
    if isinstance(weight, _autograd.ForwardTensor):
        raise _errors.BrowserUnsupported('Class-index losses have no weight JVP')
    x, dx = (input._primal, input._tangent) if isinstance(input, _autograd.ForwardTensor) else (input, None)
    primal = _index(x, target, weight, ignore_index, reduction, smoothing, dim, channels, shape, count, nll=nll)
    if dx is None: return primal
    if not nll and channels:
        dx = dx - (dx * x.softmax(dim)).sum(dim, True)
    tangent = _index(dx, target, weight, ignore_index, reduction, smoothing, dim, channels, shape, count, nll=True)
    return _autograd.ForwardTensor(primal, tangent)


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.):
    if size_average is not None or reduce is not None:
        raise _errors.BrowserUnsupported('Legacy cross entropy options require the server')
    # The common matrix/index objective should not pay for spatial/probability
    # shape inference or duplicate label masks. Other shapes/options share the
    # full validator below; this route retains its own complete bounds checks.
    if (weight is None and type(label_smoothing) in (int, float) and label_smoothing == 0
            and type(ignore_index) is int and not _state._jvp_active
            and isinstance(input, _tensor.Tensor) and isinstance(target, _tensor.Tensor)
            and input._data.ndim == 2 and target._data.ndim == 1
            and input.dtype in _REAL and target.dtype == np.dtype('int64')
            and target._data.shape[0] == input._data.shape[0] and input._data.shape[1] > 0
            and reduction in ('none', 'sum', 'mean') and np.isfinite(input._data).all()):
        rows, channels = input._data.shape
        return _unweighted_ce(input, target, ignore_index, reduction, 1, channels, (rows,), rows, validate_targets=True)
    return _loss(input, target, weight, ignore_index, reduction, label_smoothing, False)


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None:
        raise _errors.BrowserUnsupported('Legacy negative log likelihood options require the server')
    return _loss(input, target, weight, ignore_index, reduction, 0., True)


class NLLLoss(Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__()
        if size_average is not None or reduce is not None:
            raise _errors.BrowserUnsupported('Legacy negative log likelihood options require the server')
        self.register_buffer('weight', weight)
        self.ignore_index, self.reduction = ignore_index, reduction
    def forward(self, input, target):
        return nll_loss(input, target, self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
