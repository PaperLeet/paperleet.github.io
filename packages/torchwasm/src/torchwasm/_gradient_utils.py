"""Gradient clipping over existing dense NumPy buffers, without an AD graph.

The foreach argument selects native-compatible mutation grouping. Execution uses
per-gradient NumPy kernels; this module does not provide a fused accelerator API.
"""
import warnings
import numpy as np
from . import _autograd, _defaults, _errors, _ops, _state, _tensor, _vector_norm


def _parameters(parameters):
    return [parameters] if isinstance(parameters, _tensor.Tensor) else list(parameters)


def _foreach(foreach):
    return True if foreach is None else bool(foreach)


def _write(gradient, values):
    # Clipping runs under no_grad and preserves any existing gradient history.
    # Generic Tensor assignment rejects non-leaf updates, so write through the
    # owned storage here while retaining its saved-value version checks.
    if _state._jvp_active:
        raise _errors.BrowserUnsupported('Gradient clipping during JVP requires the server')
    if values.shape != gradient.shape or not np.can_cast(values.dtype, gradient.dtype, casting='same_kind'):
        raise RuntimeError('Gradient clipping cannot change the output shape or dtype kind')
    gradient._data[...] = values
    gradient._storage.bump()


def _total_norm(gradients, norm_type, error_if_nonfinite, foreach):
    order = float(norm_type)
    if not gradients:
        return _tensor.Tensor(np.asarray(0., dtype=_defaults._float_dtype), _metadata_owned=True)
    _foreach(foreach)
    norms = [_vector_norm.vector_norm(gradient, ord=order) for gradient in gradients]
    total = _vector_norm.vector_norm(_ops.stack(norms), ord=order)
    if error_if_nonfinite and not np.isfinite(total._data).item():
        raise RuntimeError('The total gradient norm is non-finite; set error_if_nonfinite=False to scale by it.')
    return total


@_autograd.no_grad()
def clip_grad_norm_(parameters, max_norm, norm_type=2., error_if_nonfinite=False, foreach=None):
    parameters = _parameters(parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    total = _total_norm(gradients, norm_type, error_if_nonfinite, foreach)
    maximum = float(max_norm)
    if gradients:
        coefficient = (maximum / (total + 1e-6)).clamp(max=1.)
        for gradient in gradients:
            _write(gradient, (gradient * coefficient)._data)
    return total


def clip_grad_norm(parameters, max_norm, norm_type=2., error_if_nonfinite=False, foreach=None):
    warnings.warn('clip_grad_norm is deprecated; use clip_grad_norm_ instead.', FutureWarning, stacklevel=2)
    return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)


def _bound(gradient, bound):
    if gradient.dtype.kind != 'f':
        raise RuntimeError('Gradient value clipping requires real floating storage.')
    if np.isfinite(bound) and abs(bound) > np.finfo(gradient.dtype).max:
        raise RuntimeError('The clipping bound cannot be represented by the gradient dtype.')
    return np.asarray(bound, dtype=gradient.dtype)


@_autograd.no_grad()
def clip_grad_value_(parameters, clip_value, foreach=None):
    if isinstance(parameters, _tensor.Tensor):
        parameters = [parameters]
    # Native value conversion happens before consuming a parameter generator.
    bound = float(clip_value)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise RuntimeError('Gradient value clipping requires at least one gradient.')
    groups = {}
    for gradient in gradients:
        groups.setdefault(gradient.dtype, []).append(gradient)
    grouped = _foreach(foreach)
    with np.errstate(all='ignore'):
        for group in groups.values():
            low, high = _bound(group[0], -bound), _bound(group[0], bound)
            if grouped:
                for gradient in group:_write(gradient, np.maximum(gradient._data, low))
                for gradient in group:_write(gradient, np.minimum(gradient._data, high))
            else:
                for gradient in group:_write(gradient, np.minimum(np.maximum(gradient._data, low), high))
