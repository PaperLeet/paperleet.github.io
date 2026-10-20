"""Checked scalar/Tensor clamp kernels and analytical pullbacks."""
import math
import numpy as np
from . import _autograd, _construction, _errors, _layout, _ops, _promotion, _state, _tensor

_HALF = np.dtype('float16')


def _bound(info, dtype):
    if info is None:
        return None
    number = info[0]
    if dtype == _HALF and math.isfinite(number) and abs(number) > 65504:
        raise RuntimeError('Scalar cannot be converted to float16 without overflow')
    return _promotion.scalar_array(number, dtype, parsed=True)


def _parameters(source, lower, upper):
    if lower is None and upper is None:
        raise RuntimeError('clamp needs at least one bound')
    if isinstance(lower, _tensor.Tensor) or isinstance(upper, _tensor.Tensor):
        raise _errors.BrowserUnsupported('Tensor clamp bounds require the server')
    source._real_only()
    lo = None if lower is None else _promotion.scalar_info(lower)
    hi = None if upper is None else _promotion.scalar_info(upper)
    dtype = source.dtype
    for info in (lo, hi):
        if info is not None:
            dtype = _promotion._combine(dtype, info[1], wrapped=True)
    if dtype.kind == 'c':
        raise RuntimeError('clamp is not supported for complex types')
    if dtype.kind == 'b' or dtype.kind == 'u' and dtype.itemsize > 1:
        raise _errors.BrowserUnsupported('clamp requires a supported real numeric computation dtype')
    return dtype, _bound(lo, dtype), _bound(hi, dtype)


def _predicate(data, lower, upper):
    if lower is None:
        return np.less_equal(data, upper)
    mask = np.asarray(np.greater_equal(data, lower))
    if upper is not None:
        # Reuse the lower predicate rather than materializing a second mask.
        np.less_equal(data, upper, out=mask, where=mask)
    return mask


def _value(data, dtype, lower, upper):
    # NumPy versions disagree on scalar NaN-bound clipping. Keep propagation
    # consistent, including on the minimum supported NumPy installation.
    nan_bound = dtype.kind == 'f' and ((lower is not None and np.isnan(lower)) or
                                       (upper is not None and np.isnan(upper)))
    if nan_bound:
        value = np.full_like(data, np.nan, dtype=dtype)
    else:
        # Let the ufunc cast into its result; integer -> floating clamp does
        # not need a separate input-sized converted array.
        value = np.clip(data, lower, upper, dtype=dtype)
    return _layout.contiguous_empty(value) if not value.size else value


def _result(source, dtype, lower, upper):
    data = source._data
    value = _value(data, dtype, lower, upper)
    # Save the array used by native-style backward, not its Python Tensor.
    # No predicate is constructed or retained by an inference/forward call.
    def pullback(gradient):
        return np.where(_predicate(data, lower, upper), gradient, 0)
    return _autograd.result(value, (source, pullback), versioned=(source,))


def apply(source, lower=None, upper=None):
    if any(isinstance(value, _autograd.ForwardTensor) for value in (source, lower, upper)):
        return forward(source, lower, upper)
    if isinstance(lower, _tensor.Tensor) or isinstance(upper, _tensor.Tensor):
        return _tensor_result(_tensor_setup(source, lower, upper))
    return _result(source, *_parameters(source, lower, upper))


def forward(dual, lower=None, upper=None):
    if isinstance(lower, _tensor.Tensor) or isinstance(upper, _tensor.Tensor):
        return _tensor_forward(dual, lower, upper)
    source = dual._primal
    dtype, lower, upper = _parameters(source, lower, upper)
    primal = _result(source, dtype, lower, upper)
    condition = _tensor.Tensor(_predicate(source._data, lower, upper), _metadata_owned=True)
    tangent = _ops.where(condition, dual._tangent, 0.)
    return _autograd.ForwardTensor(primal, tangent)


def _tensor_setup(source, lower, upper):
    bounds = tuple(value for value in (lower, upper) if value is not None)
    if not all(isinstance(value, _tensor.Tensor) for value in bounds):
        raise TypeError('clamp Tensor bounds cannot be mixed with Number bounds')
    inputs = (source, *bounds)
    for value in inputs:
        value._real_only()
    groups = [None, None]
    for value in inputs:
        category = int(bool(value.ndim))
        groups[category] = (value.dtype if groups[category] is None else
                            _construction.promote_types(groups[category], value.dtype))
    dtype = (groups[0] if groups[1] is None else groups[1] if groups[0] is None else
             _promotion._combine(groups[1], groups[0]))
    if dtype.kind == 'b' or dtype.kind == 'u' and dtype.itemsize > 1:
        raise _errors.BrowserUnsupported('clamp requires a supported real numeric computation dtype')
    # Backward comparisons have their own two-operand dtype categories.
    comparisons = (None if lower is None else _promotion.operand_dtype(source, lower),
                   None if upper is None else _promotion.operand_dtype(source, upper),
                   None if lower is None or upper is None else _promotion.operand_dtype(lower, upper))
    arrays = (source._data, None if lower is None else lower._data,
              None if upper is None else upper._data)
    labels = ('input',) + (('lower',) if lower is not None else ()) + (('upper',) if upper is not None else ())
    return dtype, inputs, arrays, comparisons, tuple(value.shape for value in inputs), labels


def _cast(array, dtype):
    if array.dtype == dtype:
        return array
    with np.errstate(all='ignore'):
        return array.astype(dtype, copy=False)


def _compare(left, right, dtype, operation):
    return operation(_cast(left, dtype), _cast(right, dtype))


def _tensor_mask(arrays, comparisons, label):
    data, lower, upper = arrays
    below, above, between = comparisons
    if label == 'input':
        if lower is None:
            return _compare(data, upper, above, np.less_equal)
        mask = _compare(data, lower, below, np.greater_equal)
        return mask if upper is None else np.logical_and(mask, _compare(data, upper, above, np.less_equal))
    if label == 'lower':
        mask = _compare(data, lower, below, np.less)
        return mask if upper is None else np.logical_and(mask, _compare(lower, upper, between, np.less))
    mask = _compare(data, upper, above, np.greater)
    return mask if lower is None else np.logical_or(mask, _compare(upper, lower, between, np.less))


def _tensor_pullback(arrays, comparisons, shapes, labels):
    # Only arrays and shape/dtype metadata survive in the ordinary graph.
    def backward(gradient, needs):
        return tuple(_ops.unbroadcast(np.where(_tensor_mask(arrays, comparisons, label), gradient, 0), shape)
                     if needed else None for label, shape, needed in zip(labels, shapes, needs))
    return backward


def _tensor_result(setup):
    dtype, inputs, arrays, comparisons, shapes, labels = setup
    data, lower, upper = arrays
    try:
        if lower is None:
            value = np.asarray(np.minimum(_cast(data, dtype), _cast(upper, dtype)))
        else:
            value = np.asarray(np.maximum(_cast(data, dtype), _cast(lower, dtype)))
            if upper is not None:
                # This is an owned fresh result, never an input or bound buffer.
                shape = np.broadcast_shapes(value.shape, upper.shape)
                if value.shape == shape:
                    np.minimum(value, _cast(upper, dtype), out=value)
                else:
                    value = np.asarray(np.minimum(value, _cast(upper, dtype)))
    except ValueError as error:
        raise RuntimeError('clamp input and bounds cannot be broadcast together') from error
    if not value.size:
        value = _layout.contiguous_empty(value)
    result = _autograd.joint_result(value, inputs, _tensor_pullback(arrays, comparisons, shapes, labels),
                                    versioned=inputs, prune_inputs=True)
    context = _state._tensor_reverse_context
    if context is not None and result._grad_node is not None:
        from . import _higher_order
        needs = result._grad_node.joint[1]
        dtypes = tuple(item.dtype for item in inputs)
        def tensor_backward(gradient):
            return tuple(_higher_order.unbroadcast(context.where(_tensor_mask(arrays, comparisons, label), gradient, 0.), shape).to(dtype=dtype)
                         if needed else None for label, shape, dtype, needed in zip(labels, shapes, dtypes, needs))
        context.record(result, inputs, tensor_backward, active=needs)
    return result


def _tensor_forward(source, lower, upper):
    if any(value is not None and not isinstance(value, _tensor.Tensor) for value in (lower, upper)):
        raise TypeError('clamp Tensor bounds cannot be mixed with Number bounds')
    values = (source, lower, upper)
    parts = tuple(None if value is None else _autograd.forward_parts(value) for value in values)
    primal_source = parts[0][0]
    primal_lower = None if parts[1] is None else parts[1][0]
    primal_upper = None if parts[2] is None else parts[2][0]
    setup = _tensor_setup(primal_source, primal_lower, primal_upper)
    primal = _tensor_result(setup)
    arrays, comparisons = setup[2], setup[3]
    reverse_based = _state._functional_jvp_active or _state._hvp_active
    if reverse_based:
        # Reverse-based functional transforms sum only transformed arguments.
        tangent = None
        for value, part, label in zip(values, parts, ('input', 'lower', 'upper')):
            if isinstance(value, _autograd.ForwardTensor):
                condition = _tensor.Tensor(_tensor_mask(arrays, comparisons, label), _metadata_owned=True)
                term = _ops.where(condition, part[1], 0.)
                tangent = term if tangent is None else tangent + term
        if tangent is None:
            tangent = _tensor.Tensor(np.zeros_like(primal._data))
    else:
        # Native direct forward AD selects nested clamp directions. Its NaN,
        # equal-bound and signed-zero rules differ from reverse-based JVP.
        tangent = parts[0][1]
        if lower is not None:
            condition = _tensor.Tensor(_compare(arrays[0], arrays[1], comparisons[0], np.less), _metadata_owned=True)
            tangent = _ops.where(condition, parts[1][1], tangent)
        if upper is not None:
            condition = _tensor.Tensor(_tensor_mask(arrays, comparisons, 'upper'), _metadata_owned=True)
            tangent = _ops.where(condition, parts[2][1], tangent)
    return _autograd.ForwardTensor(primal, tangent)
