"""Dense group normalization with a shared pullback and differentiable JVP."""
import math
import numbers
import weakref
import numpy as np

from . import _autograd, _errors, _tensor
from ._batchnorm import _inverse
from ._module import Module, Parameter
from ._normalization import _layer_statistics, parameter_dtype


def _integer(value, name):
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer')
    return int(value)


def _validate(input, num_groups, weight, bias, eps):
    if not isinstance(input, _tensor.Tensor): raise TypeError('group_norm input must be a Tensor')
    if input.ndim < 2: raise RuntimeError('group_norm requires at least two input dimensions')
    groups = _integer(num_groups, 'num_groups')
    if groups == 0: raise ZeroDivisionError('integer division or modulo by zero')
    channels = input.shape[1]
    if groups < 0 or channels % groups: raise RuntimeError('Channels must be divisible by positive num_groups')
    if not channels: raise RuntimeError('group_norm requires at least one channel')
    spatial = math.prod(input.shape[2:])
    if not spatial: raise RuntimeError('group_norm requires nonempty spatial dimensions')
    count = channels // groups * spatial
    if input.shape[0] * count == 1:
        raise ValueError(f'Expected more than 1 value per channel when training, got input size {input.shape}')
    if input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('Group normalization requires real float32/float64')
    for name, value in (('weight', weight), ('bias', bias)):
        if value is None: continue
        if not isinstance(value, _tensor.Tensor): raise TypeError(f'{name} must be a Tensor')
        if value.shape != (channels,) or value.dtype != input.dtype:
            raise RuntimeError(f'group_norm {name} shape/dtype does not match the input channels')
    if not isinstance(eps, numbers.Real): raise TypeError('eps must be a real number')
    return groups, count


def _primal(input, groups, count, weight, bias, eps):
    shape, dtype = input.shape, input.dtype
    grouped_shape = (shape[0], groups, count)
    broadcast = (1, shape[1], *([1] * (len(shape) - 2)))
    with np.errstate(all='ignore'):
        if input.numel():
            centered, variance = _layer_statistics(input._data.reshape(grouped_shape), (2,), (count,))
            inverse = _inverse(variance, eps, dtype)
            normal = (centered * inverse).reshape(shape)
            value = normal * weight._data.reshape(broadcast) if weight is not None else normal.copy()
            if bias is not None: value = value + bias._data.reshape(broadcast)
        else:
            value = np.empty(shape, dtype=dtype)
            centered = inverse = normal = None
    inputs = [input, *([weight] if weight is not None else []), *([bias] if bias is not None else [])]
    guarded = [input, *([weight] if weight is not None else [])]
    original = [(weakref.ref(v), weakref.ref(v._data)) for v in guarded]
    active_input = input.requires_grad
    active_weight = weight is not None and weight.requires_grad
    saved_normal = normal if active_input or active_weight else None
    saved_inverse = inverse if active_input else None
    saved_scale = weight._data.reshape(broadcast) if active_input and weight is not None else 1
    has_weight, has_bias, size = weight is not None, bias is not None, input.numel()
    axes = (0, *range(2, len(shape)))

    def backward(g, needs):
        for tensor_ref, data_ref in original:
            owner = tensor_ref()
            if owner is not None and owner._data is not data_ref():
                raise _errors.BrowserUnsupported('Group normalization backward after storage rebinding is unsupported')
        values = [None] * len(needs)
        if needs[0]:
            if not size: values[0] = np.empty(shape, dtype=dtype)
            else:
                scaled = (g * saved_scale).reshape(grouped_shape)
                z = saved_normal.reshape(grouped_shape)
                values[0] = (saved_inverse * (scaled - scaled.mean(2, keepdims=True)
                    - z * (scaled * z).mean(2, keepdims=True))).reshape(shape)
        index = 1
        if has_weight:
            if needs[index]:
                values[index] = (g * saved_normal).sum(axis=axes) if size else np.zeros(shape[1], dtype=dtype)
            index += 1
        if has_bias and needs[index]: values[index] = g.sum(axis=axes)
        return values

    output = _autograd.joint_result(value, inputs, backward)
    if output.requires_grad:
        counters = {v._storage for v in guarded}
        output._grad_node.versions = tuple((counter, counter.version) for counter in counters)
    return output, centered


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    groups, count = _validate(input, num_groups, weight, bias, eps)
    if not any(isinstance(v, _autograd.ForwardTensor) for v in (input, weight, bias)):
        return _primal(input, groups, count, weight, bias, eps)[0]
    def parts(value):
        return (value._primal, value._tangent) if isinstance(value, _autograd.ForwardTensor) else (value, None)
    x, dx = parts(input); w, dw = parts(weight); b, db = parts(bias)
    primal, centers = _primal(x, groups, count, w, b, eps)
    if not x.numel():
        dependencies = [v for v in (x, dx, w, dw, db) if v is not None]
        tangent = _autograd.result(np.empty(x.shape, dtype=x.dtype),
            *[(v, lambda g, shape=v.shape, dtype=v.dtype: np.zeros(shape, dtype=dtype)) for v in dependencies], saves_tensors=False)
        return _autograd.ForwardTensor(primal, tangent)
    # Unlike native LayerNorm/BatchNorm, GroupNorm differentiates its statistics
    # on the JVP reverse tape. Reusing their detached-statistics rule is wrong.
    z = x.reshape(x.shape[0], groups, count)
    centered = _autograd.result(centers, (z, lambda g: g - g.mean(2, keepdims=True)), saves_tensors=False)
    inverse = (centered.square().mean(2, True) + eps).rsqrt()
    broadcast = (1, x.shape[1], *([1] * (x.ndim - 2)))
    tangent = None
    if dx is not None:
        d = dx.reshape(z.shape)
        projected = d - d.mean(2, True)
        tangent = (projected * inverse - centered * inverse.pow(3)
                   * (projected * centered).mean(2, True)).reshape(x.shape)
        if w is not None: tangent = tangent * w.reshape(broadcast)
    if dw is not None:
        term = (centered * inverse).reshape(x.shape) * dw.reshape(broadcast)
        tangent = term if tangent is None else tangent + term
    if db is not None:
        term = db.reshape(broadcast).expand_as(primal)
        tangent = term if tangent is None else tangent + term
    return _autograd.ForwardTensor(primal, tangent)


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None):
        super().__init__()
        groups, channels = _integer(num_groups, 'num_groups'), _integer(num_channels, 'num_channels')
        if groups == 0: raise ZeroDivisionError('integer division or modulo by zero')
        if groups < 0 or channels < 0 or channels % groups:
            raise ValueError('num_channels must be divisible by positive num_groups')
        dtype = parameter_dtype(dtype, device, allocate=affine)
        self.num_groups, self.num_channels, self.eps, self.affine = groups, channels, eps, affine
        self.register_parameter('weight', Parameter(_tensor.Tensor(np.ones(channels, dtype=dtype))) if affine else None)
        self.register_parameter('bias', Parameter(_tensor.Tensor(np.zeros(channels, dtype=dtype))) if affine else None)

    def reset_parameters(self):
        if self.affine:
            with _autograd.no_grad():
                self.weight.fill_(1)
                self.bias.zero_()

    def forward(self, input):
        return group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return f'{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}'
