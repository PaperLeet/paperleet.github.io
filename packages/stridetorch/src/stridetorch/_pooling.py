"""1D/2D max pooling with ordered selection and bounded window packing."""
import math
import numbers
import numpy as np

from . import _autograd, _errors, _layout, _state, _tensor
from ._module import Module

_ALGORITHM = 'auto'  # Private correctness/performance comparison, not an API option.
_WINDOW_WORKSPACE_BYTES = 32 * 1024 * 1024
_DTYPES = tuple(np.dtype(name) for name in
                ('float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8'))


def _pair(value, name):
    if isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)):
        return (int(value),) * 2
    if not isinstance(value, (tuple, list)):
        raise TypeError(f'max_pool2d {name} must be an integer or sequence of integers')
    if any(not isinstance(v, numbers.Integral) or isinstance(v, (bool, np.bool_)) for v in value):
        raise TypeError(f'max_pool2d {name} entries must be integers')
    if len(value) not in (1, 2):
        raise RuntimeError(f'max_pool2d {name} must have one or two entries')
    return tuple(int(v) for v in value) * (2 if len(value) == 1 else 1)


def _single_int(value, name):
    if isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)):
        value = int(value)
        if not -(2**63) <= value < 2**63: raise ValueError(f'{name} exceeds int64')
        return value
    if not isinstance(value, (tuple, list)):
        raise TypeError(f'{name} must be an integer or single-element sequence')
    if any(not isinstance(v, numbers.Integral) or isinstance(v, (bool, np.bool_)) for v in value):
        raise TypeError(f'{name} entries must be integers')
    if len(value) != 1: raise RuntimeError(f'{name} requires one entry')
    return _single_int(value[0], name)


def _geometry(input, kernel_size, stride, padding, dilation, ceil_mode, *, spatial_dims=2, return_indices=False):
    if not isinstance(input, _tensor.Tensor): raise TypeError('max pooling input must be a Tensor')
    if input.ndim not in (spatial_dims + 1, spatial_dims + 2) or any(n == 0 for n in input.shape[-spatial_dims-1:]):
        raise RuntimeError('max pooling expects nonempty channels/spatial axes and optional empty batch')
    if input.dtype not in _DTYPES or spatial_dims == 1 and not return_indices and input.dtype.kind != 'f':
        raise _errors.BrowserUnsupported('This max-pooling input dtype requires the server')
    if type(ceil_mode) is not bool: raise TypeError('max pooling ceil_mode must be bool')
    if spatial_dims == 1:
        k = _single_int(kernel_size, 'kernel_size')
        step = k if stride is None or isinstance(stride, (tuple, list)) and not stride else _single_int(stride, 'stride')
        kernel, stride = (1, k), (1, step)
        padding, dilation = (0, _single_int(padding, 'padding')), (1, _single_int(dilation, 'dilation'))
        shape = (*input.shape[:-1], 1, input.shape[-1])
    else:
        kernel = _pair(kernel_size, 'kernel_size')
        stride = kernel if stride is None or isinstance(stride, (tuple, list)) and not stride else _pair(stride, 'stride')
        padding, dilation = _pair(padding, 'padding'), _pair(dilation, 'dilation')
        shape = input.shape
    if min(*kernel, *stride, *dilation) <= 0: raise RuntimeError('max pooling kernel, stride and dilation must be positive')
    if any(p < 0 or p > k // 2 for p, k in zip(padding, kernel)):
        raise RuntimeError('max pooling padding must be between zero and half the kernel size')
    out = []
    for size, k, s, p, d in zip(shape[-2:], kernel, stride, padding, dilation):
        length = (size + 2 * p - d * (k - 1) - 1 + (s - 1 if ceil_mode else 0)) // s + 1
        if ceil_mode and (length - 1) * s >= size + p: length -= 1
        if length <= 0: raise RuntimeError('max pooling calculated output size is too small')
        out.append(length)
    if spatial_dims == 1 and math.prod(kernel) > 8_000_000:
        raise _errors.BrowserUnsupported('This max-pooling kernel requires the server')
    return kernel, stride, padding, dilation, tuple(out)


def _coordinates(shape, geometry):
    _, stride, padding, dilation, out = geometry
    starts = [np.arange(n, dtype=np.int64) * s - p for n, s, p in zip(out, stride, padding)]
    first = [start + np.maximum((-start + d - 1) // d, 0) * d for start, d in zip(starts, dilation)]
    return starts, first


def _minimum(dtype):
    return -np.inf if dtype.kind == 'f' else np.iinfo(dtype).min


def _scan(data, geometry, need_indices):
    kernel, stride, padding, dilation, out = geometry
    shape = (*data.shape[:-2], *out)
    value = np.full(shape, _minimum(data.dtype), dtype=data.dtype)
    _, first = _coordinates(data.shape, geometry)
    indices = np.broadcast_to(first[0][:, None] * data.shape[-1] + first[1], shape).copy() if need_indices else None
    seen = np.zeros(out, dtype=bool)
    # A tap outside these intervals cannot reach any input sample, even at
    # the first/last output. Skip it without changing valid-tap tie/NaN order.
    taps = [range(max(0, (p - (n - 1) * s + d - 1) // d),
                  min(k, (size - 1 + p) // d + 1))
            for size, k, s, p, d, n in zip(data.shape[-2:], kernel, stride, padding, dilation, out)]
    for ky in taps[0]:
        offset_y = ky * dilation[0] - padding[0]
        lo_y = max((-offset_y + stride[0] - 1) // stride[0], 0)
        hi_y = min((data.shape[-2] - offset_y + stride[0] - 1) // stride[0], out[0])
        if hi_y <= lo_y: continue
        for kx in taps[1]:
            offset_x = kx * dilation[1] - padding[1]
            lo_x = max((-offset_x + stride[1] - 1) // stride[1], 0)
            hi_x = min((data.shape[-1] - offset_x + stride[1] - 1) // stride[1], out[1])
            if hi_x <= lo_x: continue
            block = value[..., lo_y:hi_y, lo_x:hi_x]
            source = data[..., offset_y + lo_y * stride[0]:offset_y + hi_y * stride[0]:stride[0],
                          offset_x + lo_x * stride[1]:offset_x + hi_x * stride[1]:stride[1]]
            mask = (source > block) | ~seen[lo_y:hi_y, lo_x:hi_x]
            if data.dtype.kind == 'f': mask |= np.isnan(source)
            np.copyto(block, source, where=mask)
            if need_indices:
                y = offset_y + np.arange(lo_y, hi_y, dtype=np.int64) * stride[0]
                x = offset_x + np.arange(lo_x, hi_x, dtype=np.int64) * stride[1]
                np.copyto(indices[..., lo_y:hi_y, lo_x:hi_x], y[:, None] * data.shape[-1] + x, where=mask)
            seen[lo_y:hi_y, lo_x:hi_x] = True
    return value, indices


def _window_workspace(data, geometry):
    kernel, stride, padding, dilation, out = geometry
    effective = tuple(d * (k - 1) + 1 for d, k in zip(dilation, kernel))
    padded = tuple(max(size + 2 * p, (n - 1) * s + e) for size, p, n, s, e in
                   zip(data.shape[-2:], padding, out, stride, effective))
    planes = math.prod(data.shape[:-2]);points = planes * math.prod(out)
    return planes * math.prod(padded) * data.itemsize + points * (math.prod(kernel) * (data.itemsize + 2) + 48)


def _window(data, geometry, need_indices):
    kernel, stride, padding, dilation, out = geometry
    effective = tuple(d * (k - 1) + 1 for d, k in zip(dilation, kernel))
    pads = [(p, max(p, (n - 1) * s + e - size - p)) for size, p, n, s, e in
            zip(data.shape[-2:], padding, out, stride, effective)]
    padded = np.pad(data, [(0, 0)] * (data.ndim - 2) + pads,
                    constant_values=_minimum(data.dtype)) if any(a or b for a, b in pads) else data
    windows = np.lib.stride_tricks.sliding_window_view(padded, effective, axis=(-2, -1))
    windows = windows[..., :out[0] * stride[0]:stride[0], :out[1] * stride[1]:stride[1], ::dilation[0], ::dilation[1]]
    packed = windows.reshape(*data.shape[:-2], *out, math.prod(kernel))
    arg = packed.argmax(axis=-1)
    if data.dtype.kind == 'f' and np.isnan(data).any():
        nan = np.isnan(packed)
        np.copyto(arg, math.prod(kernel) - 1 - nan[..., ::-1].argmax(axis=-1), where=nan.any(axis=-1))
    if not any(a or b for a, b in pads):
        value = np.take_along_axis(packed, arg[..., None], axis=-1)[..., 0].copy()
        if not need_indices: return value, None
        indices = (arg // kernel[1]).astype(np.int64)
        indices *= dilation[0] * data.shape[-1]
        indices += (arg % kernel[1]) * dilation[1]
        indices += (np.arange(out[0], dtype=np.int64) * stride[0] * data.shape[-1])[:, None]
        indices += np.arange(out[1], dtype=np.int64) * stride[1]
        return value, indices
    starts, first = _coordinates(data.shape, geometry)
    y = starts[0][:, None] + (arg // kernel[1]) * dilation[0]
    x = starts[1] + (arg % kernel[1]) * dilation[1]
    invalid = (y < 0) | (y >= data.shape[-2]) | (x < 0) | (x >= data.shape[-1])
    if invalid.any():
        initial = ((first[0] - starts[0]) // dilation[0])[:, None] * kernel[1] + (first[1] - starts[1]) // dilation[1]
        np.copyto(arg, initial, where=invalid)
        y = starts[0][:, None] + (arg // kernel[1]) * dilation[0]
        x = starts[1] + (arg % kernel[1]) * dilation[1]
    value = np.take_along_axis(packed, arg[..., None], axis=-1)[..., 0].copy()
    return value, (y * data.shape[-1] + x).astype(np.int64, copy=False) if need_indices else None


def _block_eligible(data, geometry):
    kernel, stride, padding, dilation, out = geometry
    return (kernel == stride and kernel in ((2, 2), (1, 2)) and padding == (0, 0) and dilation == (1, 1)
            and out == (data.shape[-2] // kernel[0], data.shape[-1] // kernel[1]))


def _block(data, geometry, need_indices):
    out = geometry[-1];kh = geometry[0][0];height, width = out[0] * kh, out[1] * 2
    value = data[..., :height:kh, :width:2].copy()
    base = (np.arange(out[0], dtype=np.int64) * kh * data.shape[-1])[:, None] + np.arange(out[1], dtype=np.int64) * 2 if need_indices else None
    indices = np.broadcast_to(base, value.shape).copy() if need_indices else None
    for y, x in (((0, 1),) if kh == 1 else ((0, 1), (1, 0), (1, 1))):
        source = data[..., y:height:kh, x:width:2]
        mask = source > value
        if data.dtype.kind == 'f': mask |= np.isnan(source)
        np.copyto(value, source, where=mask)
        if need_indices: np.copyto(indices, base + y * data.shape[-1] + x, where=mask)
    return value, indices


def _scatter(seed, indices, shape, nonoverlap, spatial_dims=2):
    planes = math.prod(shape[:-spatial_dims]);points = math.prod(indices.shape[-spatial_dims:])
    result = np.zeros((planes, math.prod(shape[-spatial_dims:])), dtype=seed.dtype)
    key = np.arange(planes)[:, None], indices.reshape(planes, points)
    if nonoverlap: result[key] += seed.reshape(planes, points)
    else: np.add.at(result, key, seed.reshape(planes, points))
    return result.reshape(shape)


def _apply(input, geometry, return_indices, *, spatial_dims=2, engine=None):
    forward = isinstance(input, _autograd.ForwardTensor)
    primal = input._primal if forward else input
    kernel, stride, _, dilation, out = geometry
    original_shape = primal.shape
    one_dimensional = spatial_dims == 1
    values = primal._data[..., None, :] if one_dimensional else primal._data
    shape = values.shape
    dimensions = len(kernel)
    need_indices = bool(return_indices) or forward or primal.requires_grad and _state._grad_enabled
    if (forward or primal.requires_grad and _state._grad_enabled) and math.prod(shape[:-dimensions]):
        _, first = _coordinates(shape, geometry)
        if any(np.any(start >= size) for start, size in zip(first, shape[-dimensions:])):
            raise _errors.BrowserUnsupported('max pooling derivatives for empty dilated sampling windows require the server')
    if _ALGORITHM not in ('auto', 'scan', 'window', 'block'): raise ValueError('Unknown internal pooling algorithm')
    block_eligible = _block_eligible if engine is None else engine._block_eligible
    workspace = _window_workspace if engine is None else engine._window_workspace
    block = _ALGORITHM in ('auto', 'block') and block_eligible(values, geometry)
    window = (_ALGORITHM == 'window' or _ALGORITHM == 'auto' and (need_indices or math.prod(kernel) >= 16))
    window = window and workspace(values, geometry) <= _WINDOW_WORKSPACE_BYTES
    if engine is None: compute = _block if block else _window if window else _scan
    else: compute = engine._block if block else engine._window if window else engine._scan
    data, indices = compute(values, geometry, need_indices)
    if engine is not None:
        data = engine._format_output(values, data)
        if indices is not None: indices = engine._format_output(values, indices)
    if one_dimensional: data = data[..., 0, :]
    if not data.size: data = _layout.contiguous_empty(data)
    if not need_indices: return _tensor.Tensor(data, _metadata_owned=True)
    public_indices = indices[..., 0, :] if one_dimensional else indices
    if not public_indices.size: public_indices = _layout.contiguous_empty(public_indices)
    index = _tensor.Tensor(public_indices, _metadata_owned=True)
    nonoverlap = all(s >= d * (k - 1) + 1 for s, d, k in zip(stride, dilation, kernel))
    def backward(seed):
        if one_dimensional: seed = seed[..., None, :]
        result = _scatter(seed, indices, shape, nonoverlap, dimensions)
        return result.reshape(original_shape) if one_dimensional else result
    value = _autograd.result(data, (primal, backward), versioned=(primal, index))
    if forward:
        tangent = input._tangent
        planes = math.prod(shape[:-dimensions]);points = math.prod(out)
        gathered = np.take_along_axis(tangent._data.reshape(planes, math.prod(shape[-dimensions:])),
                                     indices.reshape(planes, points), axis=1).reshape(data.shape)
        if engine is not None: gathered = engine._format_tangent(values, data, gathered)
        if not gathered.size: gathered = _layout.contiguous_empty(gathered)
        dual = _autograd.result(gathered, (tangent, backward), versioned=(tangent, index))
        value = _autograd.ForwardTensor(value, dual)
    return (value, index) if return_indices else value


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    geometry = _geometry(input, kernel_size, stride, padding, dilation, ceil_mode)
    return _apply(input, geometry, bool(return_indices))


def max_pool2d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode, True)


def torch_max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


class _MaxPoolNd(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding, self.dilation = padding, dilation
        self.return_indices, self.ceil_mode = return_indices, ceil_mode

    def extra_repr(self):
        return (f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, '
                f'dilation={self.dilation}, ceil_mode={self.ceil_mode}')


class MaxPool2d(_MaxPoolNd):
    def forward(self, input):
        return max_pool2d(input, self.kernel_size, self.stride, self.padding,
                          self.dilation, self.ceil_mode, self.return_indices)


def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    geometry = _geometry(input, kernel_size, stride, padding, dilation, ceil_mode,
                         spatial_dims=1, return_indices=bool(return_indices))
    return _apply(input, geometry, bool(return_indices), spatial_dims=1)


def max_pool1d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode, True)


def torch_max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)


def torch_max_pool1d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode, True)


class MaxPool1d(_MaxPoolNd):
    def forward(self, input):
        return max_pool1d(input, self.kernel_size, self.stride, self.padding,
                          self.dilation, self.ceil_mode, self.return_indices)
