"""Grouped dense 1D/2D/3D cross-correlation, shared VJPs and bilinear JVPs."""
import itertools
from functools import lru_cache
import math
import numbers
import weakref
import numpy as np
from . import _autograd, _convolution_padding as _padding_kernels, _errors, _layout, _random, _state, _tensor
from ._module import Module, Parameter
from ._normalization import parameter_dtype

_REAL = (np.dtype('float32'), np.dtype('float64'))
_ALGORITHM = 'auto'  # Private benchmark/test override, never a public API kwarg.
_MATRIX_WORKSPACE_BYTES = 32 * 1024 * 1024
_PADDING_ALGORITHM = 'auto'  # Private validation/control override.
_PAD_WORKSPACE_BYTES = 32 * 1024 * 1024
_GATHER_WORKSPACE_BYTES = 32 * 1024 * 1024


def _integer(value, name):
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer')
    return int(value)


def _tuple(value, dim, name, *, minimum=None):
    if isinstance(value, numbers.Integral): value = (value,) * dim
    elif isinstance(value, (tuple, list)): value = tuple(value)
    else: raise TypeError(f'{name} must be an integer or sequence')
    if len(value) != dim: raise RuntimeError(f'{name} must have {dim} entries')
    result = tuple(_integer(v, name) for v in value)
    if minimum is not None and any(v < minimum for v in result):
        raise RuntimeError(f'{name} entries must be at least{minimum}')
    return result


def _padding(value, dim, kernel, stride, dilation):
    if isinstance(value, str):
        if value == 'valid': return ((0, 0),) * dim
        if value != 'same': raise RuntimeError("padding must be 'valid', 'same' or nonnegative integers")
        if any(v != 1 for v in stride): raise RuntimeError("padding='same' does not support strided convolution")
        total = [d * (k - 1) for k, d in zip(kernel, dilation)]
        return tuple((v // 2, v - v // 2) for v in total)
    return tuple((v, v) for v in _tuple(value, dim, 'padding', minimum=0))


def _validate(input, weight, bias, stride, padding, dilation, groups, dim, mode):
    if not isinstance(input, _tensor.Tensor) or not isinstance(weight, _tensor.Tensor):
        raise TypeError('Convolution input and weight must be Tensors')
    if input.ndim not in (dim + 1, dim + 2) or weight.ndim != dim + 2:
        raise RuntimeError('Convolution input/weight rank does not match its spatial dimension')
    if input.dtype not in _REAL: raise _errors.BrowserUnsupported('Convolution requires real float32/float64')
    if weight.dtype != input.dtype: raise RuntimeError('Convolution input and weight dtypes must match')
    groups = _integer(groups, 'groups')
    if groups < 1: raise RuntimeError('groups must be positive')
    unbatched = input.ndim == dim + 1
    shape = (1, *input.shape) if unbatched else input.shape
    channels, outputs = shape[1], weight.shape[0]
    if outputs < 1 or outputs % groups or channels != weight.shape[1] * groups:
        raise RuntimeError('Convolution channels and grouped weight shape do not match')
    if bias is not None:
        if not isinstance(bias, _tensor.Tensor): raise TypeError('bias must be a Tensor')
        if bias.shape != (outputs,) or bias.dtype != input.dtype:
            raise RuntimeError('Convolution bias shape/dtype must match the output channels')
    kernel = tuple(weight.shape[2:])
    if any(k < 1 for k in kernel): raise RuntimeError('Convolution kernel dimensions must be positive')
    stride = _tuple(stride, dim, 'stride', minimum=1)
    dilation = _tuple(dilation, dim, 'dilation', minimum=1)
    pairs = _padding(padding, dim, kernel, stride, dilation)
    if mode not in ('zeros', 'reflect', 'replicate', 'circular'): raise ValueError('Invalid convolution padding_mode')
    for size, pair in zip(shape[2:], pairs):
        if mode != 'zeros' and not size: raise RuntimeError('Nonzero padding requires nonempty spatial axes')
        if mode == 'reflect' and max(pair) >= size: raise RuntimeError('Reflection padding must be smaller than its spatial dimension')
        if mode == 'circular' and max(pair) > size: raise RuntimeError('Circular padding cannot wrap more than once')
    padded = tuple(s + a + b for s, (a, b) in zip(shape[2:], pairs))
    effective = tuple(d * (k - 1) + 1 for k, d in zip(kernel, dilation))
    out = tuple((s - k) // st + 1 for s, k, st in zip(padded, effective, stride))
    if any(v < 1 for v in out): raise RuntimeError('Kernel size cannot exceed the padded input size')
    if shape[0] and channels and not math.prod(shape[2:]):
        raise RuntimeError('Only empty batch or channel dimensions are supported')
    return shape, unbatched, groups, kernel, stride, dilation, pairs, padded, effective, out


def _pad(data, pairs, mode):
    if not any(a or b for a, b in pairs): return data, None
    if mode == 'zeros': return np.pad(data, ((0, 0), (0, 0), *pairs)), None
    numpy_mode = {'reflect': 'reflect', 'replicate': 'edge', 'circular': 'wrap'}[mode]
    maps = []
    for axis, pair in enumerate(pairs, 2):
        mapping = np.pad(np.arange(data.shape[axis], dtype=np.int64), pair, mode=numpy_mode)
        maps.append(mapping);data = np.take(data, mapping, axis=axis)
    return data, maps


def _unpad(gradient, shape, pairs, maps):
    if maps is None:
        return gradient[(slice(None), slice(None), *[slice(p[0], p[0] + s) for s, p in zip(shape[2:], pairs)])]
    for offset in reversed(range(len(maps))):
        axis = offset + 2;target_shape = list(gradient.shape);target_shape[axis] = shape[axis]
        result = np.zeros(target_shape, dtype=gradient.dtype)
        key = [slice(None)] * gradient.ndim;key[axis] = maps[offset]
        np.add.at(result, tuple(key), gradient);gradient = result
    return gradient


def _windows(data, shape, groups, kernel, stride, dilation, effective, out):
    dim = len(kernel)
    raw = np.lib.stride_tricks.sliding_window_view(data, effective, axis=tuple(range(2, dim + 2)))
    select = (slice(None), slice(None), *[slice(None, None, v) for v in stride], *[slice(None, None, v) for v in dilation])
    return raw[select].reshape(shape[0], groups, shape[1] // groups, *out, *kernel)


def _matrix_workspace(shape, weight_shape, kernel, padded, out, itemsize):
    # Conservative logical NumPy arrays, including batch weight gradients and
    # padded input gradients. This does not measure BLAS-internal peak memory.
    packed = shape[0] * shape[1] * math.prod(kernel) * math.prod(out)
    weights = math.prod(weight_shape)
    outputs = shape[0] * weight_shape[0] * math.prod(out)
    dx = shape[0] * shape[1] * math.prod(padded)
    return (packed + shape[0] * weights + 2 * weights + 2 * outputs + dx) * itemsize


def _padding_strategy(geometry,itemsize,mode):
    if _PADDING_ALGORITHM not in ('auto','padded','gather','tap'):
        raise ValueError('Unknown internal convolution padding algorithm')
    shape,_,_,kernel,_,_,pairs,padded,_,out = geometry
    if mode!='zeros' or not any(a or b for a,b in pairs):return 'padded'
    if _PADDING_ALGORITHM=='padded':return 'padded'
    return _cached_padding_strategy(shape,kernel,padded,out,itemsize,_PADDING_ALGORITHM,_PAD_WORKSPACE_BYTES,_GATHER_WORKSPACE_BYTES)


@lru_cache(maxsize=64)
def _cached_padding_strategy(shape,kernel,padded,out,itemsize,algorithm,pad_limit,gather_limit):
    padded_bytes = math.prod((shape[0],shape[1],*padded))*itemsize
    gathered_bytes = shape[0]*shape[1]*math.prod(out)*math.prod(kernel)*itemsize
    if algorithm=='auto' and padded_bytes<=pad_limit and padded_bytes<=4*gathered_bytes:
        return 'padded'
    if algorithm=='tap' or gathered_bytes>gather_limit:return 'tap'
    return 'gather'


def _use_matrix(input, weight, geometry, clipped=False):
    shape, _, groups, kernel, _, _, _, padded, _, out = geometry
    if _ALGORITHM not in ('auto', 'direct', 'matrix'): raise ValueError('Unknown internal convolution algorithm')
    if _ALGORITHM == 'direct' or not input.numel(): return False
    work = shape[0] * math.prod(out) * weight.numel()
    if _ALGORITHM == 'auto' and (math.prod(kernel) == 1 or work < 32768): return False
    workspace_shape = shape[2:] if clipped else padded
    if _matrix_workspace(shape, weight.shape, kernel, workspace_shape, out, input._data.itemsize) > _MATRIX_WORKSPACE_BYTES:
        return False
    return bool(np.isfinite(input._data).all() and np.isfinite(weight._data).all())


def _pack(patches):
    dim = (patches.ndim - 3) // 2
    axes = (0, 1, *range(3, 3 + dim), 2, *range(3 + dim, 3 + 2 * dim))
    points = math.prod(patches.shape[3:3 + dim])
    return patches.transpose(axes).reshape(patches.shape[0], patches.shape[1], points, -1)


def _matrix_forward(patches, weights, out):
    packed = _pack(patches)
    wf = weights.reshape(weights.shape[0], weights.shape[1], -1)
    values = packed @ np.swapaxes(wf, -1, -2)
    return np.swapaxes(values, -1, -2).reshape(patches.shape[0], weights.shape[0], weights.shape[1], *out)


def _matrix_input_gradient(seed, weights, shape, kernel, stride, dilation, padded, out):
    n, groups, outputs = seed.shape[:3];channels = weights.shape[2];dim = len(kernel)
    gf = seed.reshape(n, groups, outputs, -1)
    columns = np.swapaxes(gf, -1, -2) @ weights.reshape(groups, outputs, -1)
    axes = (0, 1, *range(3, 3 + dim), 2, *range(3 + dim, 3 + 2 * dim))
    patch_gradient = columns.reshape(n, groups, *out, channels, *kernel).transpose(np.argsort(axes))
    dx = np.zeros((n, groups, channels, *padded), dtype=seed.dtype)
    for position in itertools.product(*[range(k) for k in kernel]):
        delta = patch_gradient[(slice(None), slice(None), slice(None), *([slice(None)] * dim), *position)]
        key = (slice(None), slice(None), slice(None), *[slice(p * d, p * d + size * st, st) for p, d, size, st in zip(position, dilation, out, stride)])
        dx[key] += delta
    return dx


def _matrix_weight_gradient(seed, patches, weight_shape):
    # Repack for backward rather than retaining expanded im2col activations on
    # the tape. Only the original strided window/input view remains saved.
    packed = _pack(patches)
    gf = seed.reshape(seed.shape[0], seed.shape[1], seed.shape[2], -1)
    return (gf @ packed).sum(0).reshape(weight_shape)


def _primal(input, weight, bias, geometry, mode):
    shape, unbatched, groups, kernel, stride, dilation, pairs, padded, effective, out = geometry
    dim = len(kernel);outputs = weight.shape[0];input_dtype = input.dtype
    spatial, taps = 'pqr'[:dim], 'uvw'[:dim]
    empty = not input.numel()
    active_x = input.requires_grad and _state._grad_enabled
    active_w = weight.requires_grad and _state._grad_enabled
    strategy = _padding_strategy(geometry,input._data.itemsize,mode)
    bounded = strategy!='padded'
    before = tuple(p[0] for p in pairs)
    matrix = strategy!='tap' and _use_matrix(input, weight, geometry, bounded)
    grouped_weight = weight._data.reshape(groups, outputs // groups, shape[1] // groups, *kernel)
    patches = maps = None
    with np.errstate(all='ignore'):
        if empty:
            value = _layout.contiguous_empty(np.empty((shape[0], 0 if not shape[1] else outputs, *out), dtype=input_dtype))
        else:
            data = input._data.reshape(shape)
            if strategy=='tap':
                value = _padding_kernels.forward_taps(data,grouped_weight,shape,kernel,stride,dilation,before,out)
            else:
                if bounded:patches = _padding_kernels.gather(data,shape,groups,kernel,stride,dilation,before,out)
                else:
                    data, maps = _pad(data,pairs,mode)
                    patches = _windows(data,shape,groups,kernel,stride,dilation,effective,out)
                value = (_matrix_forward(patches,grouped_weight,out) if matrix else
                    np.einsum(f'bgi{spatial}{taps},goi{taps}->bgo{spatial}',patches,grouped_weight,optimize=False)).reshape(shape[0],outputs,*out)
            if bias is not None: value += bias._data.reshape(1, outputs, *([1] * dim))
            value = np.ascontiguousarray(value)
    saved_patches = patches if active_w and not bounded else None
    saved_source = input._data.reshape(shape) if active_w and bounded else None
    saved_weight = grouped_weight if active_x else None
    # Native saves both versions even on a bias-only tape, but the analytical
    # pullback needs no input/weight arrays when only bias is differentiable.
    guarded = [input, weight]
    original = [(weakref.ref(v), weakref.ref(v._data)) for v in guarded]
    weight_shape, input_shape = weight.shape, input.shape
    operands = [input, weight, *([bias] if bias is not None else [])]
    def backward(seed, needs):
        for ref, storage in original:
            owner = ref()
            if owner is not None and owner._data is not storage():
                raise _errors.BrowserUnsupported('Convolution backward after saved storage rebinding is unsupported')
        result = [None] * len(needs)
        if empty:
            if needs[0]: result[0] = np.zeros(input_shape, dtype=input_dtype)
            if needs[1]: result[1] = np.zeros(weight_shape, dtype=input_dtype)
            if len(needs) > 2 and needs[2]: result[2] = np.zeros(outputs, dtype=input_dtype)
            return result
        g = np.asarray(seed).reshape(shape[0], groups, outputs // groups, *out)
        matrix_backward = matrix and bool(np.isfinite(g).all())
        with np.errstate(all='ignore'):
            if needs[0]:
                if bounded:
                    dx = _padding_kernels.scatter(g,saved_weight,shape,kernel,stride,dilation,before,out,matrix_backward)
                elif matrix_backward:
                    dx = _matrix_input_gradient(g, saved_weight, shape, kernel, stride, dilation, padded, out)
                else:
                    dx = np.zeros((shape[0], groups, shape[1] // groups, *padded), dtype=input_dtype)
                    for position in itertools.product(*[range(k) for k in kernel]):
                        wt = saved_weight[(slice(None), slice(None), slice(None), *position)]
                        delta = np.einsum('goi,bgo...->bgi...', wt, g, optimize=False)
                        key = (slice(None), slice(None), slice(None), *[slice(p * d, p * d + n * st, st) for p, d, n, st in zip(position, dilation, out, stride)])
                        dx[key] += delta
                if not bounded:dx = _unpad(dx.reshape(shape[0],shape[1],*padded),shape,pairs,maps)
                result[0] = dx.reshape(input_shape)
            if needs[1]:
                if strategy=='tap':
                    result[1] = _padding_kernels.weight_gradient_taps(g,saved_source,shape,weight_shape,groups,kernel,stride,dilation,before,out)
                else:
                    current_patches = (_padding_kernels.gather(saved_source,shape,groups,kernel,stride,dilation,before,out) if bounded else saved_patches)
                    result[1] = (_matrix_weight_gradient(g,current_patches,weight_shape) if matrix_backward else
                        np.einsum(f'bgo{spatial},bgi{spatial}{taps}->goi{taps}',g,current_patches,optimize=False).reshape(weight_shape))
            if len(needs) > 2 and needs[2]: result[2] = g.sum(axis=(0, *range(3, dim + 3))).reshape(outputs)
        return result
    if unbatched: value = value[0]
    output = _autograd.joint_result(value, operands, backward)
    if output.requires_grad:
        counters = {v._storage for v in guarded}
        output._grad_node.versions = tuple((v, v.version) for v in counters)
    return output


def _conv(input, weight, bias, stride, padding, dilation, groups, dim, mode='zeros'):
    geometry = _validate(input, weight, bias, stride, padding, dilation, groups, dim, mode)
    if not any(isinstance(v, _autograd.ForwardTensor) for v in (input, weight, bias)):
        return _primal(input, weight, bias, geometry, mode)
    def parts(value):
        return (value._primal, value._tangent) if isinstance(value, _autograd.ForwardTensor) else (value, None)
    x, dx = parts(input);w, dw = parts(weight);b, db = parts(bias)
    primal = _primal(x, w, b, geometry, mode)
    # Native keeps both bilinear terms, including explicit zero tangents. They
    # retain zero reverse dependencies and the 0*Inf/NaN behavior of convolution.
    if dx is None: dx = _tensor.Tensor(np.zeros(x.shape, dtype=x.dtype))
    if dw is None: dw = _tensor.Tensor(np.zeros(w.shape, dtype=w.dtype))
    tangent = _primal(dx, w, None, geometry, mode) + _primal(x, dw, None, geometry, mode)
    if db is not None:
        broadcast = (w.shape[0], *([1] * dim)) if geometry[1] else (1, w.shape[0], *([1] * dim))
        # Empty-channel native outputs have no bias elements.
        if primal.shape[0 if geometry[1] else 1] == 0:
            bias_shape, bias_dtype = db.shape, db.dtype
            term = _autograd.result(np.empty(primal.shape, dtype=primal.dtype), (db, lambda g: np.zeros(bias_shape, dtype=bias_dtype)))
        else: term = db.reshape(broadcast).expand_as(primal)
        tangent = tangent + term
    if not primal.numel():
        tangent = _autograd.view_result(_layout.contiguous_empty(tangent._data),tangent,lambda g:g)
    return _autograd.ForwardTensor(primal, tangent)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(input, weight, bias, stride, padding, dilation, groups, 1)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(input, weight, bias, stride, padding, dilation, groups, 2)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(input, weight, bias, stride, padding, dilation, groups, 3)


class _ConvNd(Module):
    _dimension = None
    _transposed = False
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__();dim = self._dimension
        self.in_channels = _integer(in_channels, 'in_channels');self.out_channels = _integer(out_channels, 'out_channels');self.groups = _integer(groups, 'groups')
        if min(self.in_channels, self.out_channels) < 0: raise RuntimeError('Negative convolution channels')
        if self.groups < 1 or self.in_channels % self.groups or self.out_channels % self.groups:
            raise ValueError('Convolution channels must be divisible by positive groups')
        self.kernel_size = _tuple(kernel_size, dim, 'kernel_size', minimum=0)
        self.stride = _tuple(stride, dim, 'stride')
        self.dilation = _tuple(dilation, dim, 'dilation')
        if isinstance(padding, str):
            if padding not in ('same', 'valid'): raise ValueError('Invalid convolution padding string')
            if padding == 'same' and any(s != 1 for s in self.stride): raise ValueError("padding='same' does not support strided convolutions")
            self.padding = padding
        else: self.padding = _tuple(padding, dim, 'padding')
        if padding_mode not in ('zeros', 'reflect', 'replicate', 'circular'): raise ValueError('Invalid convolution padding_mode')
        self.padding_mode = padding_mode;self.transposed = self._transposed;self.output_padding = (0,) * dim
        pairs = _padding(self.padding, dim, self.kernel_size, (1,) * dim, self.dilation) if isinstance(self.padding, str) else tuple((p, p) for p in self.padding)
        reversed_padding = [v for pair in reversed(pairs) for v in pair]
        self._reversed_padding_repeated_twice = reversed_padding if isinstance(self.padding, str) else tuple(reversed_padding)
        dtype = parameter_dtype(dtype, device)
        weight_channels = (self.in_channels, self.out_channels // self.groups) if self.transposed else (self.out_channels, self.in_channels // self.groups)
        self.weight = Parameter(_tensor.Tensor(np.empty((*weight_channels, *self.kernel_size), dtype=dtype)))
        self.register_parameter('bias', Parameter(_tensor.Tensor(np.empty(self.out_channels, dtype=dtype))) if bias else None)
        self.reset_parameters()
    def reset_parameters(self):
        fan = self.weight.shape[1] * math.prod(self.kernel_size)
        bound = 1 / math.sqrt(fan) if fan else 0.
        with _autograd.no_grad():
            for value in (self.weight, self.bias):
                if value is not None: value.copy_(_tensor.Tensor((_random.uniform_samples(np.random, value.shape, value.dtype) * (2 * bound) - bound).astype(value.dtype)))
    def forward(self, input):
        return _conv(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self._dimension, self.padding_mode)
    def extra_repr(self):
        result = f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}'
        if isinstance(self.padding, str) or any(self.padding): result += f', padding={self.padding}'
        if any(v != 1 for v in self.dilation): result += f', dilation={self.dilation}'
        if any(self.output_padding): result += f', output_padding={self.output_padding}'
        if self.groups != 1: result += f', groups={self.groups}'
        if self.bias is None: result += ', bias=False'
        if self.padding_mode != 'zeros': result += f', padding_mode={self.padding_mode}'
        return result


class Conv1d(_ConvNd):
    _dimension = 1


class Conv2d(_ConvNd):
    _dimension = 2


class Conv3d(_ConvNd):
    _dimension = 3
