"""Vector norms with direct NumPy reductions and real analytical AD."""
import math
import weakref

import numpy as np

from . import _autograd, _errors, _graph, _layout, _ops, _state, _tensor

_MISSING = _ops._OMITTED


def _order(value):
    if isinstance(value, _tensor.Tensor):
        if value.ndim or value.requires_grad:
            raise TypeError('Norm order must be a scalar without gradients')
        value = value.item()
    if not isinstance(value, (int, float, np.integer, np.floating, np.bool_)):
        raise TypeError('Norm order must be a real number')
    return float(value)


def _axes(dim, ndim):
    requested = () if dim is None else tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
    normalized = tuple(_layout.dimension(axis, ndim) for axis in requested)
    if len(set(normalized)) != len(normalized):
        raise RuntimeError('Duplicate norm reduction dimensions')
    return tuple(range(ndim)) if not requested else normalized if ndim else ()


def _restore(value, axes, keepdim):
    if not keepdim:
        for axis in sorted(axes):
            value = value.unsqueeze(axis) if isinstance(value, _tensor.Tensor) else np.expand_dims(value, axis)
    return value


def _sign(source):
    # Real Torch sign returns zero for NaN, whereas NumPy sign returns NaN.
    return (source > 0).astype(source.dtype) - (source < 0).astype(source.dtype)


def _sign_tensor(primal):
    value = _sign(primal._data)
    if _state._hvp_active:
        return _tensor.Tensor(value, _metadata_owned=True)
    return _autograd.result(value, (primal, lambda g: _graph.LOGICAL_ZERO), saves_tensors=False, zero_passthrough=True)


def _values(source, p, axes, keepdim):
    real_dtype = source.real.dtype
    work = source.astype('float32') if real_dtype.itemsize < 4 else source
    absolute = np.asarray(np.abs(work))
    with np.errstate(all='ignore'):
        if p == 0:
            value = np.count_nonzero(source, axis=axes, keepdims=keepdim)
        elif p == math.inf:
            value = np.max(absolute, axis=axes, keepdims=keepdim)
        elif p == -math.inf:
            value = np.min(absolute, axis=axes, keepdims=keepdim)
        elif p == 1:
            value = absolute.sum(axis=axes, keepdims=keepdim)
        elif p == 2:
            np.square(absolute, out=absolute)
            value = np.sqrt(absolute.sum(axis=axes, keepdims=keepdim))
        else:
            # Native NormOps stores p in the accumulation type and computes
            # its reciprocal there. Rounding only the final reciprocal changes
            # large float32 norm values enough to change composed sin phases.
            order = np.asarray(p, dtype=absolute.dtype)
            inverse = np.asarray(1., dtype=absolute.dtype) / order
            np.power(absolute, order, out=absolute)
            value = np.power(absolute.sum(axis=axes, keepdims=keepdim), inverse)
    return np.asarray(value, dtype=real_dtype)


def _backward(seed, source, value, p, axes, keepdim):
    if p == 0:
        # Native norm_backward returns an undefined gradient for the 0-norm.
        return None
    grad = _restore(seed, axes, keepdim)
    length = _restore(value, axes, keepdim)
    if p == 1:
        return _sign(source) * grad
    if p == 2:
        direction = np.divide(source, length, out=np.zeros_like(source), where=length != 0)
        return grad * direction
    absolute = np.abs(source)
    if math.isinf(p):
        selected = (absolute == length) | np.isnan(absolute)
        count = selected.sum(axis=axes, keepdims=True)
        return _sign(source) * ((grad / count) * selected)
    if p < 1:
        weight = np.asarray(np.power(absolute, p - 1))
        np.copyto(weight, 0, where=source == 0)
        return (_sign(source) * weight) * grad * np.power(length, 1 - p)
    if p < 2:
        direction = _sign(source) * np.power(absolute, p - 1)
    else:
        direction = source * np.power(absolute, p - 2)
    scale = grad / np.power(length, p - 1)
    return direction * np.where(length == 0, 0, scale)


def _calculate(input, p, axes, keepdim):
    value = _values(input._data, p, axes, keepdim)
    if not _state._grad_enabled or not input.requires_grad:
        return _tensor.Tensor(value, _metadata_owned=True)
    source = input._data
    owner = weakref.ref(input)

    def backward(seed):
        current = owner()
        active = current._data if current is not None else source
        return _backward(seed, active, value, p, axes, keepdim)

    output = _autograd.result(value, (input, backward))
    context = _state._tensor_reverse_context
    return context.norm(output, input, p, axes, keepdim) if context is not None else output


def _forward(input, p, axes, keepdim):
    primal, direction = input._primal, input._tangent
    value = _calculate(primal, p, axes, keepdim)
    output = value
    reduction = axes if axes else None
    if p == 0:
        tangent = _tensor.Tensor(np.zeros_like(value._data), _metadata_owned=True)
        return _autograd.ForwardTensor(value, tangent)
    if _state._hvp_active and not math.isinf(p) and (p < 1 or p == 2 or p >= 3):
        # Native functional HVP uses three reverse traversals. At zero entries
        # its norm rules mask derivatives that reverse-of-direct-JVP leaves NaN.
        # Route every tangent dependence through one guarded identity so both
        # the power and the saved norm paths receive that same mask.
        length = _restore(value._data, axes, keepdim)
        if p < 1:
            # At a zero positive fractional norm, or a NaN/Inf norm, native
            # higher derivatives retain IEEE NaNs rather than this subgradient.
            mask = (primal._data == 0) & np.isfinite(length) & ((length != 0) if p > 0 else True)
            if direction.requires_grad:
                # A nonlinear upstream Jacobian contributes its own derivative.
                # Preserve its nonfinite arithmetic instead of applying the
                # finite-input subgradient to the whole Hessian chain.
                mask &= np.all(np.isfinite(primal._data), axis=axes, keepdims=True)
        else:
            mask = np.broadcast_to(length == 0, primal.shape)
        if np.any(mask):
            primal = _autograd.result(primal._data, (primal, lambda g: np.where(mask, 0, g)))
            # Keep the saved-primal subgradient while independently guarding
            # the nonlinear forward-direction route. A nonfinite reduction
            # must retain its IEEE derivative on that latter route.
            direction_mask = mask if p >= 1 else mask & np.all(np.isfinite(primal._data), axis=axes, keepdims=True)
            if np.any(direction_mask):
                direction = _autograd.result(direction._data, (direction, lambda g: np.where(direction_mask, 0, g)), saves_tensors=False)
            value = _calculate(primal, p, axes, keepdim)
    functional = _state._functional_jvp_active or _state._hvp_active
    with np.errstate(all='ignore'):
        if p == 1:
            tangent = (_sign_tensor(primal) * direction).sum(reduction, keepdim)
        elif p == 2:
            if functional:
                length = _restore(value, axes, keepdim)
                slope = _ops.where(length == 0, 0., primal / length)
                tangent = (direction * slope).sum(reduction, keepdim)
            else:
                tangent = (primal * direction).sum(reduction, keepdim) / value
                tangent = _ops.where(value == 0, 0., tangent)
        elif math.isinf(p):
            length = _restore(value, axes, keepdim)
            raw, norm = primal._data, length._data
            selected = (np.abs(raw) == norm) | (np.isnan(raw) & np.isnan(norm))
            mask = _tensor.Tensor(selected.astype(value.dtype), _metadata_owned=True)
            count = mask.sum(reduction, True)
            tangent = ((_sign_tensor(primal) * direction) * mask / count).sum(reduction, keepdim)
        elif p < 1:
            weights = _ops.where(primal == 0, 0., primal.abs() ** (p - 1))
            if functional:
                scale = _restore(value, axes, keepdim) ** (1 - p)
                tangent = ((direction * scale) * (weights * _sign_tensor(primal))).sum(reduction, keepdim)
            else:
                tangent = (weights * (_sign_tensor(primal) * direction)).sum(reduction, keepdim) * value ** (1 - p)
        elif p < 2:
            product = (primal.abs() ** (p - 1)) * (_sign_tensor(primal) * direction)
            if functional:
                product = _ops.where(_restore(value, axes, keepdim) == 0, 0., product)
                tangent = product.sum(reduction, keepdim) / value ** (p - 1)
            else:
                tangent = _ops.where(value == 0, 0., product.sum(reduction, keepdim) / value ** (p - 1))
        else:
            product = (primal.abs() ** (p - 2)) * (primal * direction)
            if functional:
                product = _ops.where(_restore(value, axes, keepdim) == 0, 0., product)
                tangent = product.sum(reduction, keepdim) / value ** (p - 1)
            else:
                tangent = _ops.where(value == 0, 0., product.sum(reduction, keepdim) / value ** (p - 1))
    return _autograd.ForwardTensor(output, tangent)


def vector_norm(x=_MISSING, ord=2, dim=None, keepdim=False, *, dtype=None, out=None, input=_MISSING):
    if x is _MISSING:
        x = input
    elif input is not _MISSING:
        raise TypeError('vector_norm received both x and input')
    if not isinstance(x, _tensor.Tensor):
        raise TypeError('vector_norm expects a Tensor')
    if type(keepdim) is not bool:
        raise TypeError('keepdim must be bool')
    if out is not None:
        raise _errors.BrowserUnsupported('Norm output buffers require the server')
    if x.dtype.kind not in 'fc':
        raise RuntimeError('vector_norm expects a floating or complex Tensor')
    if x.dtype == np.dtype('float16') and _state._grad_enabled and x.requires_grad:
        raise _errors.BrowserUnsupported('Float16 norm autograd requires the server')
    p, axes = _order(ord), _axes(dim, x.ndim)
    if p < 0 or math.isinf(p):
        if any(x.shape[axis] == 0 for axis in axes):
            raise RuntimeError('This norm order has no identity for an empty reduction')
    if dtype is not None:
        target = np.dtype(dtype)
        if target.kind != x.dtype.kind or target.itemsize < x.dtype.itemsize:
            raise RuntimeError('Norm dtype must preserve the real/complex kind and not narrow precision')
        if target.char not in 'efdFD':
            raise _errors.BrowserUnsupported('Unsupported norm dtype')
        x = x.to(dtype=target)
    if isinstance(x, _autograd.ForwardTensor):
        return _forward(x, p, axes, keepdim)
    return _calculate(x, p, axes, keepdim)
