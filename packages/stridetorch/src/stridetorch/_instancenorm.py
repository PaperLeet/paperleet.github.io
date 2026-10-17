"""Per-instance normalization with snapshot state and analytical derivatives."""
import math
import numbers
import warnings
import weakref
import numpy as np

from . import _autograd, _errors, _state, _tensor
from ._batchnorm import _NormBase, _inverse, _statistics


def _validate(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, verify_spatial):
    if not isinstance(input, _tensor.Tensor): raise TypeError('instance_norm input must be a Tensor')
    if input.ndim < 2: raise RuntimeError('instance_norm requires at least two input dimensions')
    if input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('Instance normalization requires real float32/float64')
    if type(use_input_stats) is not bool: raise TypeError('use_input_stats must be bool')
    if not isinstance(momentum, numbers.Real): raise TypeError('momentum must be a real number')
    if not isinstance(eps, numbers.Real): raise TypeError('eps must be a real number')
    spatial = math.prod(input.shape[2:])
    if verify_spatial and use_input_stats and spatial == 1:
        raise ValueError(f'Expected more than 1 spatial element when training, got input size {input.shape}')
    if not use_input_stats:
        if running_mean is None: raise RuntimeError('running_mean must be defined in evaluation mode')
        if running_var is None: raise RuntimeError('running_var must be defined in evaluation mode')
    if input.numel() and (running_mean is None) != (running_var is None):
        raise ValueError('running_mean and running_var must both be None or both be Tensors')
    for name, value in (('weight', weight), ('bias', bias), ('running_mean', running_mean), ('running_var', running_var)):
        if value is None: continue
        if not isinstance(value, _tensor.Tensor): raise TypeError(f'{name} must be a Tensor')
        if value.shape != (input.shape[1],) or value.dtype != input.dtype:
            raise RuntimeError(f'instance_norm {name} shape/dtype does not match the input channels')
        if name.startswith('running_') and value.requires_grad and _state._grad_enabled:
            raise RuntimeError('instance_norm running statistics are not differentiable')
    if input.shape[0] * input.shape[1] == 0 and (weight is not None or bias is not None):
        raise IndexError('instance_norm cannot select affine parameters from an empty batch/channel dimension')
    return spatial


def _primal(input, running_mean, running_var, weight, bias, training, momentum, eps, spatial):
    shape, dtype, size = input.shape, input.dtype, input.numel()
    channels = shape[1]
    broadcast = (1, channels, *([1] * (len(shape) - 2)))
    spatial_axes = tuple(range(2, len(shape)))
    affine_axes = (0, *spatial_axes)
    # Native instance_norm repeats affine/state buffers and normalizes a
    # contiguous N*C view. These snapshots allow later buffer/affine mutations.
    scale = weight._data.reshape(broadcast).copy() if weight is not None else 1
    with np.errstate(all='ignore'):
        if size:
            source = np.ascontiguousarray(input._data)
            if training:
                folded = source.reshape(1, shape[0] * channels, *shape[2:])
                mean, centered, variance = _statistics(folded, (0, *spatial_axes))
                stat_shape = (shape[0], channels, *([1] * (len(shape) - 2)))
                mean, centered, variance = mean.reshape(stat_shape), centered.reshape(shape), variance.reshape(stat_shape)
            else:
                mean = running_mean._data.reshape(broadcast).copy()
                variance = running_var._data.reshape(broadcast).copy()
                centered = source - mean
            inverse = _inverse(variance, eps, dtype)
            if training and eps == 0:
                inverse = np.where(variance == 0, 0, inverse).astype(dtype, copy=False)
            normal = centered * inverse
            value = normal * scale
            if bias is not None: value = value + bias._data.reshape(broadcast)
        else:
            value = np.empty(shape, dtype=dtype)
            mean = variance = centered = inverse = normal = None
        if training:
            updates = []
            for buffer, statistic, unbiased in ((running_mean, mean, False), (running_var, variance, True)):
                if buffer is None: continue
                if size:
                    update = statistic.reshape(shape[0], channels).mean(0)
                    if unbiased: update = update * np.divide(float(spatial), spatial - 1)
                    update = (1 - momentum) * buffer._data + momentum * update
                elif shape[0] == 0:
                    update = np.full(channels, np.nan, dtype=dtype)
                else: update = None
                updates.append((buffer, update))
            # Compute both updates before copying either: running buffers may
            # share storage, while the native composite repeats each first.
            for buffer, update in updates:
                if update is not None: buffer._data[...] = update
                # Native repeat/reduce/copy updates original buffer versions,
                # including empty inputs and forward-mode training calls.
                buffer._storage.bump()
    inputs = [input, *([weight] if weight is not None else []), *([bias] if bias is not None else [])]
    guarded = [input] if size and input._data.flags.c_contiguous else []
    original = [(weakref.ref(v), weakref.ref(v._data)) for v in guarded]
    has_weight, has_bias = weight is not None, bias is not None

    def backward(g, needs):
        for owner_ref, data_ref in original:
            owner = owner_ref()
            if owner is not None and owner._data is not data_ref():
                raise _errors.BrowserUnsupported('Instance normalization backward after storage rebinding is unsupported')
        values = [None] * len(needs)
        if needs[0]:
            if not size: values[0] = np.empty(shape, dtype=dtype)
            else:
                scaled = g * scale
                values[0] = inverse * (scaled - scaled.mean(axis=spatial_axes, keepdims=True)
                    - normal * (scaled * normal).mean(axis=spatial_axes, keepdims=True)) if training else scaled * inverse
        index = 1
        if has_weight:
            if needs[index]: values[index] = (g * normal).sum(axis=affine_axes) if size else np.zeros(channels, dtype=dtype)
            index += 1
        if has_bias and needs[index]: values[index] = g.sum(axis=affine_axes)
        return values

    output = _autograd.joint_result(value, inputs, backward, versioned=guarded)
    return output, centered, inverse


def _instance_norm(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, verify_spatial):
    spatial = _validate(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, verify_spatial)
    def parts(value):
        return (value._primal, value._tangent) if isinstance(value, _autograd.ForwardTensor) else (value, None)
    x, dx = parts(input); w, dw = parts(weight); b, db = parts(bias)
    mean, _ = parts(running_mean); variance, _ = parts(running_var)
    primal, centers, inverses = _primal(x, mean, variance, w, b, use_input_stats, momentum, eps, spatial)
    if dx is None and dw is None and db is None: return primal
    if not x.numel():
        dependencies = [v for v in (x, dx, w, dw, db) if v is not None]
        tangent = _autograd.result(np.empty(x.shape, dtype=x.dtype),
            *[(v, lambda g, shape=v.shape, dtype=v.dtype: np.zeros(shape, dtype=dtype)) for v in dependencies], saves_tensors=False)
        return _autograd.ForwardTensor(primal, tangent)
    axes = tuple(range(2, x.ndim))
    broadcast = (1, x.shape[1], *([1] * (x.ndim - 2)))
    # The native batch-normalization composite freezes saved statistics on
    # reverse-over-forward tapes; centering retains an identity derivative.
    center = _autograd.result(centers, (x, lambda g: g), saves_tensors=False)
    inverse = _tensor.Tensor(inverses)
    tangent = None
    if dx is not None:
        if use_input_stats:
            projected = dx - dx.mean(axes, True)
            tangent = projected * inverse - center * inverse.pow(3) * (projected * center).mean(axes, True)
        else: tangent = dx * inverse
        if w is not None:
            snapshot = _autograd.result(w._data.copy(), (w, lambda g: g), saves_tensors=False)
            tangent = tangent * snapshot.reshape(broadcast)
    if dw is not None:
        term = center * inverse * dw.reshape(broadcast)
        tangent = term if tangent is None else tangent + term
    if db is not None:
        term = db.reshape(broadcast).expand_as(primal)
        tangent = term if tangent is None else tangent + term
    return _autograd.ForwardTensor(primal, tangent)


def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None,
                  use_input_stats=True, momentum=.1, eps=1e-5):
    return _instance_norm(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, True)


def torch_instance_norm(input, weight, bias, running_mean, running_var,
                        use_input_stats, momentum, eps, cudnn_enabled):
    if type(cudnn_enabled) is not bool: raise TypeError('cudnn_enabled must be bool')
    return _instance_norm(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, False)


class _InstanceNorm(_NormBase):
    def __init__(self, num_features, eps=1e-5, momentum=.1, affine=False,
                 track_running_stats=False, device=None, dtype=None, *, bias=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype, bias=bias)

    def forward(self, input):
        if input.ndim not in (self._no_batch_dim, self._no_batch_dim + 1):
            raise ValueError(f'expected {self._no_batch_dim}D or {self._no_batch_dim + 1}D input (got {input.ndim}D input)')
        unbatched = input.ndim == self._no_batch_dim
        feature_dim = 0 if unbatched else 1
        if input.shape[feature_dim] != self.num_features:
            if self.affine: raise ValueError('input feature count does not match num_features')
            warnings.warn('input feature count does not match num_features; num_features is not used when affine=False', UserWarning, stacklevel=2)
        value = input.unsqueeze(0) if unbatched else input
        value = instance_norm(value, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, 0. if self.momentum is None else self.momentum, self.eps)
        return value.squeeze(0) if unbatched else value

    def _load_from_state_dict(self, state, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if local_metadata.get('version') is None and not self.track_running_stats:
            names = [prefix + key for key in ('running_mean', 'running_var') if prefix + key in state]
            if names:
                error_msgs.append('Unexpected running stats buffer(s) for InstanceNorm with track_running_stats=False; enable tracking to load this legacy checkpoint')
                for name in names: state.pop(name)
        super()._load_from_state_dict(state, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class InstanceNorm1d(_InstanceNorm):
    _no_batch_dim = 2


class InstanceNorm2d(_InstanceNorm):
    _no_batch_dim = 3


class InstanceNorm3d(_InstanceNorm):
    _no_batch_dim = 4
