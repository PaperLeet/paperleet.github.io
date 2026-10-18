"""Stateful dense batch normalization with native-compatible tape conventions."""
import math
import numbers
import weakref
import numpy as np

from . import _autograd, _errors, _state, _tensor
from ._module import Module, Parameter
from ._normalization import parameter_dtype


def _statistics(source, axes):
    mean = source.mean(axis=axes, keepdims=True)
    centered = source - mean
    variance = np.mean(centered * centered, axis=axes, keepdims=True)
    if source.dtype == np.dtype('float32'):
        sensitive = (np.isfinite(mean) & np.isfinite(variance) & (variance > 0)
                     & (np.abs(mean) * np.finfo(np.float32).eps > np.sqrt(variance) * 1e-3))
        if sensitive.any():
            channels = source.shape[1]
            mask = sensitive.reshape(channels)
            selected = np.moveaxis(source, 1, 0).reshape(channels, -1)[mask].astype(np.float64)
            exact_mean = selected.mean(axis=1, keepdims=True)
            delta = selected - exact_mean
            # Only ill-conditioned channels pay for wider accumulation. Keep
            # subtraction centered rather than expanding it into alpha*x+beta.
            channel_center = np.moveaxis(centered, 1, 0).copy(order='C')
            channel_center.reshape(channels, -1)[mask] = delta
            centered = np.moveaxis(channel_center, 0, 1)
            mean = mean.copy(); mean.reshape(channels)[mask] = exact_mean.reshape(-1)
            variance = variance.copy(); variance.reshape(channels)[mask] = (delta * delta).mean(axis=1)
    return mean, centered, variance


def _inverse(variance, eps, dtype):
    # Epsilon and reciprocal-square-root round once into the storage dtype.
    # This widens the channel statistics only, not the activation matrix.
    return (1 / np.sqrt(variance.astype(np.float64, copy=False) + eps)).astype(dtype, copy=False)


def _validate(input, running_mean, running_var, weight, bias, training, momentum, eps):
    if not isinstance(input, _tensor.Tensor): raise TypeError('batch_norm input must be a Tensor')
    if input.ndim < 2: raise RuntimeError('batch_norm requires at least two input dimensions')
    if input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('Batch normalization requires real float32/float64')
    if type(training) is not bool: raise TypeError('batch_norm training must be bool')
    if not isinstance(momentum, numbers.Real): raise TypeError('batch_norm momentum must be a real number')
    if not isinstance(eps, numbers.Real): raise TypeError('batch_norm eps must be a real number')
    count = math.prod(input.shape[:1] + input.shape[2:])
    if training and count == 1:
        raise ValueError('Expected more than 1 value per channel when training')
    if (training and eps <= 0) or (not training and eps < 0):
        raise ValueError('batch_norm eps must be positive in training and nonnegative in evaluation')
    if not training:
        if running_mean is None: raise RuntimeError('running_mean must be defined in evaluation mode')
        if running_var is None: raise RuntimeError('running_var must be defined in evaluation mode')
    if (running_mean is None) != (running_var is None):
        raise ValueError('running_mean and running_var must both be None or both be Tensors')
    channels = input.shape[1]
    for name, value in [('weight', weight), ('bias', bias), ('running_mean', running_mean), ('running_var', running_var)]:
        if value is None: continue
        if not isinstance(value, _tensor.Tensor): raise TypeError(f'batch_norm {name} must be a Tensor')
        if value.shape != (channels,) or value.dtype != input.dtype:
            raise RuntimeError(f'batch_norm {name} shape/dtype does not match the input channels')
        if name.startswith('running_') and value.requires_grad and _state._grad_enabled:
            raise RuntimeError('batch_norm running statistics are not differentiable')
    if not channels: raise IndexError('batch_norm requires at least one channel')
    return count


def _primal(input, running_mean, running_var, weight, bias, training, momentum, eps, count):
    source = input._data
    axes = (0, *range(2, input.ndim))
    broadcast = (1, input.shape[1], *([1] * (input.ndim - 2)))
    scale = weight._data.reshape(broadcast) if weight is not None else 1
    size = source.size
    with np.errstate(all='ignore'):
        if size:
            if training:
                mean, centered, variance = _statistics(source, axes)
            else:
                mean = running_mean._data.reshape(broadcast)
                centered = source - mean
                variance = running_var._data.reshape(broadcast)
            inverse = _inverse(variance, eps, source.dtype)
            normalized = centered * inverse
            value = normalized * scale
            if bias is not None: value = value + bias._data.reshape(broadcast)
            if training and running_mean is not None:
                # Native kernels update these buffers without version bumps;
                # explicit user mutations still invalidate saved version checks.
                running_mean._data[...] = (1 - momentum) * running_mean._data + momentum * mean.reshape(-1)
                running_var._data[...] = (1 - momentum) * running_var._data + momentum * variance.reshape(-1) * (count / (count - 1))
        else:
            value = np.empty_like(source)
            centered = inverse = normalized = None
    inputs = [input]
    if weight is not None: inputs.append(weight)
    if bias is not None: inputs.append(bias)
    guarded = [input, *([weight] if weight is not None else [])]
    if size and running_mean is not None: guarded.extend([running_mean, running_var])
    original = [(weakref.ref(value), weakref.ref(value._data)) for value in guarded]
    active_input = input.requires_grad
    active_weight = weight is not None and weight.requires_grad
    saved_normalized = normalized if training and (active_input or active_weight) else None
    saved_inverse = inverse if training and active_input else None
    saved_scale = scale if active_input else None
    saved_source = source if not training and active_weight else None
    saved_mean = running_mean._data if not training and active_weight else None
    saved_variance = running_var._data if not training and (active_input or active_weight) else None
    input_shape, input_dtype, channels = input.shape, input.dtype, input.shape[1]
    has_weight, has_bias = weight is not None, bias is not None

    def backward(g, needs):
        for tensor_ref, data_ref in original:
            owner = tensor_ref()
            if owner is not None and owner._data is not data_ref():
                raise _errors.BrowserUnsupported('Batch normalization backward after storage rebinding is unsupported')
        values = [None] * len(needs)
        if size:
            if training:
                current_inverse, current_normalized = saved_inverse, saved_normalized
            else:
                current_inverse = _inverse(saved_variance, eps, input_dtype).reshape(broadcast) if saved_variance is not None else None
                current_normalized = ((saved_source - saved_mean.reshape(broadcast)) * current_inverse
                                      if active_weight else None)
        if needs[0]:
            if not size: values[0] = np.empty(input_shape, dtype=input_dtype)
            else:
                scaled = g * saved_scale
                if training:
                    values[0] = current_inverse * (scaled - scaled.mean(axis=axes, keepdims=True)
                        - current_normalized * np.mean(scaled * current_normalized, axis=axes, keepdims=True))
                else: values[0] = scaled * current_inverse
        index = 1
        if has_weight:
            if needs[index]:
                values[index] = np.sum(g * current_normalized, axis=axes) if size else np.zeros(channels, dtype=input_dtype)
            index += 1
        if has_bias and needs[index]: values[index] = g.sum(axis=axes)
        return values

    output = _autograd.joint_result(value, inputs, backward)
    if output.requires_grad:
        counters = {value._storage for value in guarded}
        output._grad_node.versions = tuple((counter, counter.version) for counter in counters)
    return output, centered, inverse


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=.1, eps=1e-5):
    count = _validate(input, running_mean, running_var, weight, bias, training, momentum, eps)
    forward = any(isinstance(value, _autograd.ForwardTensor) for value in (input, weight, bias))
    def parts(value):
        return (value._primal, value._tangent) if isinstance(value, _autograd.ForwardTensor) else (value, None)
    x, dx = parts(input)
    w, dw = parts(weight)
    b, db = parts(bias)
    # Native batch_norm declares its running buffers nondifferentiable in
    # forward mode as well: their dual tangents are ignored, not multiplied
    # through the inference expression. Only their primal storage is updated.
    mean, _ = parts(running_mean)
    variance, _ = parts(running_var)
    primal, centered_values, inverse_values = _primal(x, mean, variance, w, b, training, momentum, eps, count)
    if not forward: return primal
    if not x.numel():
        dependencies = [value for value in (x, dx, w, dw, db) if value is not None]
        tangent = _autograd.result(np.empty_like(x._data), *[(value, lambda g, shape=value.shape, dtype=value.dtype: np.zeros(shape, dtype=dtype)) for value in dependencies], saves_tensors=False)
        return _autograd.ForwardTensor(primal, tangent)
    axes = (0, *range(2, x.ndim))
    broadcast = (1, x.shape[1], *([1] * (x.ndim - 2)))
    # Native JVP holds saved mean/inverse fixed on the reverse tape. Centering
    # keeps an identity input derivative; differentiating the statistics here
    # would produce a different reverse-over-forward gradient.
    centered = _autograd.result(centered_values, (x, lambda g: g), saves_tensors=False)
    inverse = _tensor.Tensor(inverse_values)
    if _state._hvp_active and training:
        high = x.to(dtype=np.float64) if x.dtype==np.dtype('float32') else x
        centered = high-high.mean(axes,True)
        inverse = (centered.square().mean(axes,True)+eps).rsqrt()
        if dx is not None:dx=dx.to(dtype=high.dtype)
    tangent = None
    if dx is not None:
        if training:
            projected = dx - dx.mean(axes, True)
            tangent = projected * inverse - centered * inverse.pow(3) * (projected * centered).mean(axes, True)
        else: tangent = dx * inverse
        if w is not None: tangent = tangent * w.reshape(broadcast)
    if dw is not None:
        term = centered * inverse * dw.reshape(broadcast)
        tangent = term if tangent is None else tangent + term
    if db is not None:
        term = db.reshape(broadcast).expand_as(primal)
        tangent = term if tangent is None else tangent + term
    return _autograd.ForwardTensor(primal, tangent)


class _NormBase(Module):
    _version = 2

    def __init__(self, num_features, eps=1e-5, momentum=.1, affine=True, track_running_stats=True, device=None, dtype=None, *, bias=True):
        super().__init__()
        if not isinstance(num_features, numbers.Integral) or isinstance(num_features, (bool, np.bool_)):
            raise TypeError('num_features must be an integer')
        if num_features < 0: raise RuntimeError('Negative BatchNorm dimensions')
        dtype = parameter_dtype(dtype, device, allocate=affine or track_running_stats)
        self.num_features, self.eps, self.momentum = int(num_features), eps, momentum
        self.affine, self.track_running_stats = affine, track_running_stats
        self.register_parameter('weight', Parameter(_tensor.Tensor(np.empty(num_features, dtype=dtype))) if affine else None)
        self.register_parameter('bias', Parameter(_tensor.Tensor(np.empty(num_features, dtype=dtype))) if affine and bias else None)
        self.register_buffer('running_mean', _tensor.Tensor(np.empty(num_features, dtype=dtype)) if track_running_stats else None)
        self.register_buffer('running_var', _tensor.Tensor(np.empty(num_features, dtype=dtype)) if track_running_stats else None)
        self.register_buffer('num_batches_tracked', _tensor.Tensor(np.array(0, dtype=np.int64)) if track_running_stats else None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            with _autograd.no_grad():
                self.running_mean.zero_()
                self.running_var.fill_(1)
                self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            with _autograd.no_grad():
                self.weight.fill_(1)
                if self.bias is not None: self.bias.zero_()

    def extra_repr(self):
        return f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, bias={self.bias is not None}, track_running_stats={self.track_running_stats}'

    def _load_from_state_dict(self, state, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version')
        if self.track_running_stats and (version is None or version < 2):
            name = prefix + 'num_batches_tracked'
            if name not in state:
                state[name] = self.num_batches_tracked if self.num_batches_tracked is not None else _tensor.Tensor(np.array(0, dtype=np.int64))
        super()._load_from_state_dict(state, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class _BatchNorm(_NormBase):
    def forward(self, input):
        self._check_input_dim(input)
        factor = 0. if self.momentum is None else self.momentum
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            if _state._jvp_active:
                if not _state._hvp_active:
                    raise _errors.BrowserUnsupported('BatchNorm JVP cannot mutate a captured batch counter; use untracked or evaluation mode')
                _state._jvp_active=False
                try:self.num_batches_tracked.add_(1)
                finally:_state._jvp_active=True
            else:self.num_batches_tracked.add_(1)
            if self.momentum is None: factor = 1. / float(self.num_batches_tracked)
        training = True if self.training else self.running_mean is None and self.running_var is None
        mean = self.running_mean if not self.training or self.track_running_stats else None
        variance = self.running_var if not self.training or self.track_running_stats else None
        return batch_norm(input, mean, variance, self.weight, self.bias, training, factor, self.eps)



class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.ndim not in (2, 3): raise ValueError(f'expected 2D or 3D input (got {input.ndim}D input)')


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.ndim != 4: raise ValueError(f'expected 4D input (got {input.ndim}D input)')


class BatchNorm3d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.ndim != 5: raise ValueError(f'expected 5D input (got {input.ndim}D input)')
