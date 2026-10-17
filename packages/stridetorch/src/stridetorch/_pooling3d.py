"""Volumetric max pooling with bounded packing and ordered tap selection."""
import itertools
import math
import numbers
import sys
import numpy as np

from . import _errors, _layout, _pooling, _tensor


def _triple(value, name):
    if isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)):
        entries = (value,) * 3
    elif isinstance(value, (tuple, list)):
        if any(not isinstance(v, numbers.Integral) or isinstance(v, (bool, np.bool_)) for v in value):
            raise TypeError(f'max_pool3d {name} entries must be integers')
        if len(value) not in (1, 3): raise RuntimeError(f'max_pool3d {name} requires one or three entries')
        entries = tuple(value) * (3 if len(value) == 1 else 1)
    else: raise TypeError(f'max_pool3d {name} must be an integer or sequence')
    result = tuple(int(v) for v in entries)
    if any(not -(2**31) <= v < 2**31 for v in result):
        raise RuntimeError(f'max_pool3d {name} exceeds int32')
    return result


def _geometry(input, kernel_size, stride, padding, dilation, ceil_mode):
    if not isinstance(input, _tensor.Tensor): raise TypeError('max pooling input must be a Tensor')
    if input.ndim not in (4, 5) or any(n == 0 for n in input.shape[-4:]):
        raise RuntimeError('max pooling expects nonempty channels/spatial axes and optional empty batch')
    if input.dtype not in _pooling._DTYPES:
        raise _errors.BrowserUnsupported('This max-pooling input dtype requires the server')
    if type(ceil_mode) is not bool: raise TypeError('max pooling ceil_mode must be bool')
    kernel = _triple(kernel_size, 'kernel_size')
    stride = kernel if stride is None or isinstance(stride, (tuple, list)) and not stride else _triple(stride, 'stride')
    padding, dilation = _triple(padding, 'padding'), _triple(dilation, 'dilation')
    if min(*kernel, *stride, *dilation) <= 0: raise RuntimeError('max pooling kernel, stride and dilation must be positive')
    if any(p < 0 or p > k // 2 for p, k in zip(padding, kernel)):
        raise RuntimeError('max pooling padding must be between zero and half the kernel size')
    out = []
    for size, k, s, p, d in zip(input.shape[-3:], kernel, stride, padding, dilation):
        length = (size + 2 * p - d * (k - 1) - 1 + (s - 1 if ceil_mode else 0)) // s + 1
        if ceil_mode and (length - 1) * s >= size + p: length -= 1
        if length <= 0: raise RuntimeError('max pooling calculated output size is too small')
        out.append(length)
    points = math.prod(max(1, n) for n in input.shape[:-3]) * math.prod(out)
    if (math.prod(kernel) > 8_000_000 or points > 8_000_000
            or math.prod(input.shape[-3:]) > np.iinfo(np.intp).max):
        raise _errors.BrowserUnsupported('This max-pooling geometry requires the server')
    return kernel, stride, padding, dilation, tuple(out)


def _channels_last(data):
    if data.ndim != 5 or data.strides[1] == 0: return False
    minimum = 0
    for axis in (1, 4, 3, 2, 0):
        step = data.strides[axis] // data.itemsize
        if data.shape[axis] == 0 or step < minimum: return False
        if axis == 0 and minimum == data.strides[1] // data.itemsize: return False
        minimum = step * data.shape[axis]
    return True


def _format_output(source, result):
    if not result.size: return _layout.contiguous_empty(result)
    if _channels_last(source):
        # copy(order='C') also gives singleton axes their conventional strides.
        return result.transpose(0, 2, 3, 4, 1).copy(order='C').transpose(0, 4, 1, 2, 3)
    return result.copy(order='C') if not result.flags.c_contiguous else result


def _format_tangent(source, output, tangent):
    # Native gather canonicalizes ambiguous contiguous output layouts (C=1
    # or a single spatial point), even when the primal has NDHWC strides.
    return tangent.copy(order='C') if output.flags.c_contiguous else _format_output(source, tangent)


def _initial(data, geometry, need_indices):
    out = geometry[-1];shape = (*data.shape[:-3], *out)
    value = np.full(shape, _pooling._minimum(data.dtype), dtype=data.dtype)
    indices = None
    if need_indices:
        if not value.size: return value, np.empty(shape, dtype=np.int64)
        _, first = _pooling._coordinates(data.shape, geometry)
        base = ((first[0][:, None, None] * data.shape[-2] + first[1][None, :, None])
                * data.shape[-1] + first[2])
        indices = np.broadcast_to(base, shape).copy()
    return value, indices


def _scan(data, geometry, need_indices):
    kernel, stride, padding, dilation, out = geometry
    value, indices = _initial(data, geometry, need_indices)
    if not value.size: return value, indices
    seen = np.zeros(out, dtype=bool)
    taps = [range(max(0, (p - (n - 1) * s + d - 1) // d),
                  min(k, (size - 1 + p) // d + 1))
            for size, k, s, p, d, n in zip(data.shape[-3:], kernel, stride, padding, dilation, out)]
    for tap in itertools.product(*taps):
        offsets = tuple(t * d - p for t, d, p in zip(tap, dilation, padding))
        low = tuple(max(0, (-o + s - 1) // s) for o, s in zip(offsets, stride))
        high = tuple(min(n, (size - o + s - 1) // s) for size, o, s, n in zip(data.shape[-3:], offsets, stride, out))
        if any(h <= l for l, h in zip(low, high)): continue
        region = tuple(slice(l, h) for l, h in zip(low, high))
        source = data[(Ellipsis,) + tuple(slice(o + l * s, o + h * s, s)
                      for o, l, h, s in zip(offsets, low, high, stride))]
        block = value[(Ellipsis,) + region]
        mask = (source > block) | ~seen[region]
        if data.dtype.kind == 'f': mask |= np.isnan(source)
        np.copyto(block, source, where=mask)
        if need_indices:
            z, y, x = [o + np.arange(l, h, dtype=np.int64) * s for o, l, h, s in zip(offsets, low, high, stride)]
            linear = (z[:, None, None] * data.shape[-2] + y[None, :, None]) * data.shape[-1] + x
            np.copyto(indices[(Ellipsis,) + region], linear, where=mask)
        seen[region] = True
    return value, indices


def _window_workspace(data, geometry):
    kernel, stride, padding, dilation, out = geometry
    effective = tuple(d * (k - 1) + 1 for d, k in zip(dilation, kernel))
    padded = tuple(max(size + 2 * p, (n - 1) * s + e) for size, p, n, s, e in
                   zip(data.shape[-3:], padding, out, stride, effective))
    planes = math.prod(data.shape[:-3]);points = planes * math.prod(out)
    return planes * math.prod(padded) * data.itemsize + points * (math.prod(kernel) * (data.itemsize + 2) + 64)


def _window(data, geometry, need_indices):
    if not data.size: return _initial(data, geometry, need_indices)
    kernel, stride, padding, dilation, out = geometry
    effective = tuple(d * (k - 1) + 1 for d, k in zip(dilation, kernel))
    pads = [(p, max(p, (n - 1) * s + e - size - p)) for size, p, n, s, e in
            zip(data.shape[-3:], padding, out, stride, effective)]
    padded = np.pad(data, [(0, 0)] * (data.ndim - 3) + pads,
                    constant_values=_pooling._minimum(data.dtype)) if any(a or b for a, b in pads) else data
    windows = np.lib.stride_tricks.sliding_window_view(padded, effective, axis=(-3, -2, -1))
    windows = windows[(Ellipsis,) + tuple(slice(0, n * s, s) for n, s in zip(out, stride))
                      + tuple(slice(None, None, d) for d in dilation)]
    packed = windows.reshape(*data.shape[:-3], *out, math.prod(kernel))
    arg = packed.argmax(axis=-1)
    if data.dtype.kind == 'f' and np.isnan(data).any():
        nan = np.isnan(packed)
        np.copyto(arg, math.prod(kernel) - 1 - nan[..., ::-1].argmax(axis=-1), where=nan.any(axis=-1))
    starts, first = _pooling._coordinates(data.shape, geometry)
    offsets = ((arg // (kernel[1] * kernel[2])), (arg // kernel[2]) % kernel[1], arg % kernel[2])
    coords = [start.reshape(tuple(n if i == axis else 1 for i, n in enumerate(out))) + offset * d
              for axis, (start, offset, d) in enumerate(zip(starts, offsets, dilation))]
    invalid = np.zeros(arg.shape, dtype=bool)
    for coord, size in zip(coords, data.shape[-3:]): invalid |= (coord < 0) | (coord >= size)
    if invalid.any():
        initial = np.zeros(out, dtype=np.int64)
        for axis, (begin, valid, d, weight) in enumerate(zip(starts, first, dilation, (kernel[1]*kernel[2], kernel[2], 1))):
            offset = (valid - begin) // d
            initial += offset.reshape(tuple(n if i == axis else 1 for i, n in enumerate(out))) * weight
        np.copyto(arg, initial, where=invalid)
        offsets = (arg // (kernel[1] * kernel[2]), (arg // kernel[2]) % kernel[1], arg % kernel[2])
        coords = [start.reshape(tuple(n if i == axis else 1 for i, n in enumerate(out))) + offset * d
                  for axis, (start, offset, d) in enumerate(zip(starts, offsets, dilation))]
    value = np.take_along_axis(packed, arg[..., None], axis=-1)[..., 0].copy()
    indices = ((coords[0] * data.shape[-2] + coords[1]) * data.shape[-1] + coords[2]).astype(np.int64, copy=False) if need_indices else None
    return value, indices


def _block_eligible(data, geometry):
    kernel, stride, padding, dilation, out = geometry
    return (kernel == stride and all(k in (1, 2) for k in kernel) and padding == (0, 0, 0)
            and dilation == (1, 1, 1) and out == tuple(n // k for n, k in zip(data.shape[-3:], kernel)))


def _block(data, geometry, need_indices):
    if not data.size: return _initial(data, geometry, need_indices)
    kernel = geometry[0];out = geometry[-1]
    stops = tuple(n * k for n, k in zip(out, kernel))
    value = data[(Ellipsis,) + tuple(slice(0, stop, k) for stop, k in zip(stops, kernel))].copy()
    indices = None
    if need_indices:
        z, y, x = [np.arange(n, dtype=np.int64) * k for n, k in zip(out, kernel)]
        base = (z[:, None, None] * data.shape[-2] + y[None, :, None]) * data.shape[-1] + x
        indices = np.broadcast_to(base, value.shape).copy()
    for tap in itertools.islice(itertools.product(*(range(k) for k in kernel)), 1, None):
        source = data[(Ellipsis,) + tuple(slice(t, stop, k) for t, stop, k in zip(tap, stops, kernel))]
        mask = source > value
        if data.dtype.kind == 'f': mask |= np.isnan(source)
        np.copyto(value, source, where=mask)
        if need_indices:
            np.copyto(indices, base + (tap[0] * data.shape[-2] + tap[1]) * data.shape[-1] + tap[2], where=mask)
    return value, indices


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    geometry = _geometry(input, kernel_size, stride, padding, dilation, ceil_mode)
    return _pooling._apply(input, geometry, bool(return_indices), spatial_dims=3, engine=sys.modules[__name__])


def max_pool3d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode, True)


def torch_max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)


class MaxPool3d(_pooling._MaxPoolNd):
    def forward(self, input):
        return max_pool3d(input, self.kernel_size, self.stride, self.padding,
                          self.dilation, self.ceil_mode, self.return_indices)
