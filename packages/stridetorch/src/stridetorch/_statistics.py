"""Shared moments, validation and analytical rules for real statistics."""
import math
import warnings
import weakref
import numpy as np
from . import _layout

MISSING = _layout.UNSPECIFIED_DIM


def options(shape, args, dim, unbiased, keepdim, correction):
    """Resolve the legacy unbiased overload without accepting bool dimensions."""
    if args and type(args[0]) is bool:
        if len(args) != 1 or dim is not MISSING or unbiased is not MISSING or keepdim is not MISSING:
            raise TypeError('Invalid statistics arguments')
        unbiased = args[0]
    else:
        if len(args) > 3: raise TypeError('Too many statistics arguments')
        values = [dim, unbiased, keepdim]
        for index, value in enumerate(args):
            if values[index] is not MISSING: raise TypeError('Repeated statistics argument')
            values[index] = value
        dim, unbiased, keepdim = values
    if unbiased is not MISSING:
        if type(unbiased) is not bool or correction is not MISSING:
            raise TypeError('Specify boolean unbiased or correction, not both')
        correction = int(unbiased)
    elif correction is MISSING or correction is None:
        correction = 1
    if keepdim is MISSING: keepdim = False
    if type(keepdim) is not bool: raise TypeError('keepdim must be bool')
    from . import _tensor
    if isinstance(correction, _tensor.Tensor):
        if correction.numel() != 1: raise TypeError('correction must be a scalar')
        correction = correction.item()
    if isinstance(correction, (complex, np.complexfloating)):
        raise RuntimeError('correction must be real')
    if not isinstance(correction, (int, float, np.integer, np.floating, np.bool_)):
        raise TypeError('correction must be a real number')
    correction = float(correction)
    if dim is MISSING: dim = None
    requested = () if dim is None else tuple(dim) if isinstance(dim, (tuple, list)) else (dim,)
    normalized = tuple(_layout.dimension(d, len(shape)) for d in requested)
    if len(set(normalized)) != len(normalized): raise RuntimeError('Duplicate reduction dimensions')
    axes = tuple(range(len(shape))) if not requested else normalized if shape else ()
    count = math.prod(shape[d] for d in axes)
    # Native reverse rules distinguish a missing dimension from an explicit
    # empty list. Forward moments still reduce all dimensions in both cases.
    backward_count = 1 if dim is not None and not requested else count
    return axes, count, backward_count, dim is None or not shape, keepdim, correction, dim is not None and not requested


def _mean_array(data, axes):
    # Callers are inside the forward kernel or the reverse engine numerical scope.
    accumulator = np.float32 if data.dtype.itemsize < 4 else data.dtype
    count = math.prod(data.shape[d] for d in axes)
    return data.sum(axis=axes, keepdims=True, dtype=accumulator) / np.asarray(count, dtype=accumulator)


def _restore(value, axes, keepdim):
    if not keepdim:
        for axis in sorted(axes): value = np.expand_dims(value, axis)
    return value


def _multiply(left, right):
    if left.dtype != right.dtype:
        # Module conversion can rebind a saved input's storage/dtype. Native
        # zero-dimensional Tensor promotion still applies to the old seed.
        from . import _tensor
        return (_tensor.Tensor(left)*_tensor.Tensor(right))._data
    return left*right


def _variance_backward(input, seed, config, original=None, center_mean=None):
    axes, _, count, global_rule, keepdim, correction, explicit_empty = config
    data = input._data
    if original is None or original() is not data:
        center_mean = _mean_array(data, axes).astype(data.dtype, copy=False)
    grad = seed if global_rule or explicit_empty else _restore(seed, axes, keepdim)
    if global_rule and count - correction <= 0:
        derivative = np.where(data == center_mean, np.nan, np.inf).astype(data.dtype)
        return _multiply(grad, derivative)
    factor = np.divide(np.asarray(2, dtype=grad.dtype), np.asarray(count-correction, dtype=grad.dtype))
    return _multiply(factor*grad, data-center_mean)

def _exceptional_moments(data, axes, mean, variance, denominator):
    """Welford updates for rows whose ordinary moments overflow or are nonfinite.

    Normal finite rows keep the vectorized two-pass kernel. The fallback visits
    reduced coordinates in logical order, retaining only output-sized moments.
    """
    invalid = (~np.isfinite(mean) | ~np.isfinite(variance)).reshape(-1)
    if not invalid.any(): return mean, variance, False
    retained = tuple(d for d in range(data.ndim) if d not in axes)
    order = retained+tuple(sorted(axes))
    count = math.prod(data.shape[d] for d in axes)
    rows = data.transpose(order).reshape(invalid.size, count)[invalid]
    average = np.zeros(rows.shape[0], dtype=mean.dtype)
    moment = np.zeros_like(average)
    for index in range(count):
        delta = rows[:, index]-average
        average = average+delta/(index+1)
        moment = moment+delta*(rows[:, index]-average)
    updated = moment/denominator
    new_mean, new_variance = mean.reshape(-1).copy(), variance.reshape(-1).copy()
    new_mean[invalid], new_variance[invalid] = average, updated
    return new_mean.reshape(mean.shape), new_variance.reshape(variance.shape), True


def _calculate(input, config, kind, paired):
    from . import _autograd, _state
    axes, count, _, _, keepdim, correction, _ = config
    data = input._data
    if count-correction <= 0:
        warnings.warn('Degrees of freedom <= 0 for statistics reduction', UserWarning, stacklevel=4)
    repaired = False
    with np.errstate(all='ignore'):
        # The paired CPU kernel accumulates float32 moments in double precision.
        # Keep its variance distinct from the separately rounded single result.
        samples = data.astype(np.float64) if paired and data.dtype == np.float32 else data
        mean = _mean_array(samples, axes)
        centered = samples-mean
        variance = np.sum(centered*centered, axis=axes, keepdims=True)
        denominator = np.asarray(max(0., count-correction), dtype=variance.dtype)
        variance = variance/denominator
        if not count: variance = np.full_like(variance, np.nan)
        elif paired: mean, variance, repaired = _exceptional_moments(samples, axes, mean, variance, denominator)
        raw = np.sqrt(variance) if kind == 'std' else variance
        raw = np.asarray(raw if keepdim else raw.squeeze(axis=axes), dtype=data.dtype)
        mean = np.asarray(mean if keepdim else mean.squeeze(axis=axes), dtype=data.dtype)
        original = None
        center_mean = None
        if input.requires_grad and _state._grad_enabled:
            original = weakref.ref(data)
            # Cache only one mean per output, never a centered input-sized array.
            # Paired float32 output means use a different accumulator; reverse
            # follows the input mean kernel. Copy to isolate mutable mean outputs.
            source_mean = _mean_array(data, axes) if paired and (data.dtype == np.float32 or repaired) else _restore(mean, axes, keepdim)
            center_mean = source_mean.astype(data.dtype, copy=True)
    def backward(seed):
        if kind == 'std':
            seed = np.divide(seed, raw*np.asarray(2, dtype=raw.dtype))
            seed = np.where(raw == 0, np.asarray(0, dtype=raw.dtype), seed)
        return _variance_backward(input, seed, config, original, center_mean)
    result = _autograd.result(raw, (input, backward), versioned=(input,) if kind == 'var' else None)
    if not paired: return result
    def mean_backward(seed):
        grad = _restore(seed, axes, keepdim)/np.asarray(count, dtype=seed.dtype)
        return np.broadcast_to(grad, input.shape)
    saved = (input, result) if kind == 'std' else (input,)
    average = _autograd.result(mean, (input, mean_backward), versioned=saved)
    return result, average


def _forward(input, config, kind, paired):
    from . import _autograd, _ops
    primal, tangent = input._primal, input._tangent
    outputs = _calculate(primal, config, kind, paired)
    value = outputs[0] if paired else outputs
    axes, _, count, global_rule, keepdim, correction, explicit_empty = config
    centered = primal-primal.mean(axes, True)
    with np.errstate(all='ignore'):
        if global_rule and count-correction <= 0:
            derivative = _ops.where(centered == 0, float('nan'), float('inf'))
        else:
            factor = float(np.divide(2., count-correction))
            derivative = centered*factor
        product = (derivative*tangent).sum(axes, keepdim)
        if kind == 'std': product = _ops.where(value == 0, 0., product/(value*2))
    first = _autograd.ForwardTensor(value, product)
    if not paired: return first
    return first, _autograd.ForwardTensor(outputs[1], tangent.mean(axes, keepdim))


def reduce(input, *args, dim=MISSING, unbiased=MISSING, keepdim=MISSING,
           correction=MISSING, kind='var', paired=False):
    from . import _autograd
    if input.dtype.kind == 'c':
        from . import _errors
        raise _errors.BrowserUnsupported('Complex statistics require the server')
    if input.dtype.kind != 'f': raise RuntimeError('Statistics require a real floating tensor')
    config = options(input.shape, args, dim, unbiased, keepdim, correction)
    if isinstance(input, _autograd.ForwardTensor): return _forward(input, config, kind, paired)
    return _calculate(input, config, kind, paired)


def var_mean(input, *args, dim=MISSING, unbiased=MISSING, keepdim=MISSING, correction=MISSING):
    return reduce(input, *args, dim=dim, unbiased=unbiased, keepdim=keepdim,
                  correction=correction, paired=True)


def std_mean(input, *args, dim=MISSING, unbiased=MISSING, keepdim=MISSING, correction=MISSING):
    return reduce(input, *args, dim=dim, unbiased=unbiased, keepdim=keepdim,
                  correction=correction, kind='std', paired=True)
