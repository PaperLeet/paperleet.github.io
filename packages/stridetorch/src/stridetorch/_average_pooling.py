"""Local average-pooling reductions, ordered adjoints and linear forward AD."""
import math
import numbers
from functools import lru_cache
from collections import namedtuple
import numpy as np
from . import _autograd, _errors, _pooling, _state, _tensor
from ._module import Module

_ALGORITHM = 'auto'  # Private comparison of ordered scans and bounded packing.
_BACKWARD_ALGORITHM = 'auto'
_WORKSPACE_BYTES = 32 * 1024 * 1024
_MAX_OUTPUT_ELEMENTS = 8_000_000
_FLOATS = tuple(np.dtype(n) for n in ('float16', 'float32', 'float64'))
_LIMITS = {d:(float(np.finfo(d).max), 2.0 ** np.finfo(d).nmant) for d in _FLOATS[1:]}


def _input(input, *, adaptive):
    if not isinstance(input, _tensor.Tensor): raise TypeError('Average pooling input must be a Tensor')
    if input.ndim not in (3, 4) or min(input.shape[-2:]) <= 0:
        raise RuntimeError('Average pooling expects 3D/4D input with nonempty spatial dimensions')
    if not adaptive and not input.shape[-3]: raise RuntimeError('avg_pool2d requires nonempty channels')
    if input.dtype not in _FLOATS and (adaptive or input.dtype != np.dtype('int64')):
        raise _errors.BrowserUnsupported('This average-pooling input dtype requires the server')


def _allocation(shape, out):
    if math.prod(out) * max(1, math.prod(shape[:-2])) > _MAX_OUTPUT_ELEMENTS or max(out) > 4_000_000:
        raise _errors.BrowserUnsupported('This average-pooling output allocation requires the server')


def _output_size(value, spatial):
    if type(value) is int or isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)): value = (value, value)
    elif not isinstance(value, (tuple, list)): raise TypeError('output_size must be an integer or pair')
    if len(value) != 2: raise RuntimeError('output_size must contain two dimensions')
    result = []
    for size, current in zip(value, spatial):
        if size is None: size = current
        if type(size) is not int and (not isinstance(size, numbers.Integral) or isinstance(size, (bool, np.bool_))):
            raise TypeError('output_size entries must be integers or None')
        if size < 0: raise RuntimeError('output_size entries must be nonnegative')
        result.append(int(size))
    return tuple(result)


def _make_adaptive_geometry(spatial, out):
    ranges = []
    for length, count in zip(spatial, out):
        index = np.arange(count, dtype=np.int64)
        ranges.extend((index * length // max(count, 1), ((index + 1) * length + count - 1) // max(count, 1)))
    for array in ranges: array.setflags(write=False)
    return (*ranges, None, True)


@lru_cache(maxsize=64)
def _cached_geometry(h, w, oh, ow):return _make_adaptive_geometry((h, w), (oh, ow))


def _adaptive_geometry(shape, out):
    if sum(out) <= 2048:return _cached_geometry(*shape[-2:], *out)
    return _make_adaptive_geometry(shape[-2:], out)


@lru_cache(maxsize=32)
def _binary_matrices(h, w, oh, ow, dtype_name):
    dtype = np.dtype(dtype_name)
    if (h * oh + w * ow + oh + ow) * dtype.itemsize > 65536:return None
    geometry = _adaptive_geometry((h, w), (oh, ow));sy, ey, sx, ex = geometry[:4]
    rows = np.zeros((oh, h), dtype=dtype);columns = np.zeros((ow, w), dtype=dtype)
    for i, (a, b) in enumerate(zip(sy, ey)):rows[i, a:b] = 1
    for i, (a, b) in enumerate(zip(sx, ex)):columns[i, a:b] = 1
    rows.setflags(write=False);columns.setflags(write=False)
    height = (ey - sy).astype(dtype)[:, None];width = (ex - sx).astype(dtype)
    height.setflags(write=False);width.setflags(write=False)
    count = int((ey - sy).max()) * int((ex - sx).max())
    overlaps = (math.ceil(oh / h) + 1) * (math.ceil(ow / w) + 1)
    return rows, columns, height, width, count, overlaps


def _matrix_plan(shape, geometry, dtype):
    sy, ey, sx, ex, _, adaptive = geometry
    if not adaptive or dtype.kind != 'f':return None
    if math.prod(shape[:-2]) * min(shape[-2] * len(sx), len(sy) * shape[-1]) * dtype.itemsize > _WORKSPACE_BYTES:return None
    return _binary_matrices(*shape[-2:], len(sy), len(sx), dtype.str)


def _matrix_safe(data, count):
    if not data.size:return True
    # Reassociation of mixed signs can erase a small residual even when every
    # sample passes the exponent-range guard. Keep those planes ordered.
    if np.any((data.min(axis=(-2, -1)) < 0) & (data.max(axis=(-2, -1)) > 0)):
        return False
    limit, precision = _LIMITS[_accumulator(data.dtype)]
    if data.nbytes > min(_WORKSPACE_BYTES // 4, 4 * 1024 * 1024):
        elements = max(1, min(_WORKSPACE_BYTES // 4, 4 * 1024 * 1024) // data.itemsize)
        magnitude = 0.;minimum = float('inf');flat = data.flat
        for start in range(0, data.size, elements):
            absolute = np.abs(flat[start:start+elements]);high = float(absolute.max(initial=0))
            if not high <= limit / (max(count, 1) + 1):return False
            magnitude = max(magnitude, high)
            low = float(absolute.min(initial=np.inf))
            if low == 0:low = float(absolute.min(initial=np.inf, where=absolute != 0))
            minimum = min(minimum, low)
        return minimum >= magnitude * max(count, 1) / precision
    absolute = np.abs(data)
    magnitude = float(absolute.max(initial=0))
    if not magnitude <= limit / (max(count, 1) + 1):return False
    minimum = float(absolute.min(initial=np.inf))
    if minimum == 0 and magnitude:minimum = float(absolute.min(initial=np.inf, where=absolute != 0))
    return minimum >= magnitude * max(count, 1) / precision


def _accumulator(dtype): return _FLOATS[1] if dtype == _FLOATS[0] else dtype



_PackingPlan = namedtuple('_PackingPlan', 'rows columns flat invalid height width taps output plane_bytes index_bytes')
_PLAN_CACHE_BYTES = 64 * 1024
_CACHE_PLANS = True


@lru_cache(maxsize=64)
def _cached_pack_plan(h, w, oh, ow, dtype_name, inverse):
    if min(h, w, oh, ow) <= 0 or (h + w if inverse else oh + ow) > 2048:
        return None
    geometry = (_cached_adjoint_geometry(h, w, oh, ow) if inverse
                else _cached_geometry(h, w, oh, ow))
    sy, ey, sx, ex = geometry[:4]
    height, width = ey - sy, ex - sx
    kh, kw = int(height.max()), int(width.max()); taps = kh * kw
    masked = not (bool(np.all(height == kh)) and bool(np.all(width == kw)))
    dtype = np.dtype(dtype_name); accumulator = _accumulator(dtype)
    coordinates = (len(sy) * kh + len(sx) * kw) * 8
    mask_bytes = len(sy) * len(sx) * taps if masked else 0
    divisor_bytes = 0 if inverse else (len(sy) + len(sx)) * accumulator.itemsize
    if coordinates + mask_bytes + divisor_bytes > _PLAN_CACHE_BYTES:
        return None
    source_h, source_w = (oh, ow) if inverse else (h, w)
    iy = sy[:, None] + np.arange(kh); ix = sx[:, None] + np.arange(kw)
    invalid = ((iy >= ey[:, None])[:, None, :, None] |
               (ix >= ex[:, None])[None, :, None, :]) if masked else None
    rows = np.minimum(iy, source_h - 1); columns = np.minimum(ix, source_w - 1)
    rows.setflags(write=False); columns.setflags(write=False)
    rows = rows[:, None, :, None]; columns = columns[None, :, None, :]
    if invalid is not None:
        invalid.setflags(write=False); invalid = invalid.reshape(len(sy), len(sx), taps)
    flat = None
    flat_bytes = len(sy) * len(sx) * taps * np.dtype(np.intp).itemsize
    if (coordinates + mask_bytes + divisor_bytes + flat_bytes <= _PLAN_CACHE_BYTES
            and source_h * source_w - 1 <= np.iinfo(np.intp).max):
        indices = (rows * source_w + columns).astype(np.intp, copy=False)
        indices.setflags(write=False); flat = indices.reshape(-1)
    if inverse: height = width = None
    else:
        height = height.astype(accumulator); width = width.astype(accumulator)
        height.setflags(write=False); width.setflags(write=False); height = height[:, None]
    plane_bytes = len(sy) * len(sx) * taps * (dtype.itemsize + accumulator.itemsize + 2)
    index_bytes = (len(sy) + len(sx)) * taps * 16
    return _PackingPlan(rows, columns, flat, invalid, height, width, taps,
                        (len(sy), len(sx)), plane_bytes, index_bytes)


def _pack_plan(shape, geometry, dtype, *, inverse=False):
    if not _CACHE_PLANS or not geometry[-1]: return None
    return _cached_pack_plan(*shape[-2:], len(geometry[0]), len(geometry[2]), dtype.str, inverse)


def _packed_from_plan(data, plan):
    # Both gathers own new storage. Request C order only when necessary;
    # accumulation must never write into the user's input.
    if plan.flat is not None and data.flags.c_contiguous:
        gathered = np.take(data.reshape(*data.shape[:-2], -1), plan.flat, axis=-1, mode='clip')
    else: gathered = data[..., plan.rows, plan.columns]
    packed = np.asarray(gathered, dtype=_accumulator(data.dtype), order='C')
    packed = packed.reshape(*data.shape[:-2], *plan.output, plan.taps)
    if plan.invalid is not None: np.copyto(packed, 0, where=plan.invalid)
    np.add.accumulate(packed, axis=-1, out=packed)
    # Copy the final values: do not retain all prefixes, and keep native +0.
    value = np.add(packed[..., -1], 0)
    if plan.height is not None:
        np.divide(value, plan.height, out=value); np.divide(value, plan.width, out=value)
    return value.astype(data.dtype, copy=False)


def _partition_forward(data, geometry, partition):
    kh, kw = partition;oh, ow = len(geometry[0]), len(geometry[2])
    if kh * kw <= 4:
        value = np.zeros((*data.shape[:-2], oh, ow), dtype=_accumulator(data.dtype))
        for y in range(kh):
            for x in range(kw):np.add(value, data[..., y:oh*kh:kh, x:ow*kw:kw], out=value)
    else:
        view = data.reshape(*data.shape[:-2], oh, kh, ow, kw).swapaxes(-3, -2)
        packed = np.array(view, dtype=_accumulator(data.dtype), order='C', copy=True).reshape(*data.shape[:-2], oh, ow, kh * kw)
        np.add.accumulate(packed, axis=-1, out=packed)
        value = np.add(packed[..., -1], 0)
    if geometry[5]:np.divide(value, value.dtype.type(kh), out=value);np.divide(value, value.dtype.type(kw), out=value)
    else:_divide(value, geometry)
    return value.astype(data.dtype, copy=False)


def _divide(value, geometry):
    sy, ey, sx, ex, divisor, adaptive = geometry
    if value.dtype.kind == 'i':
        denominator = np.asarray(divisor, dtype=np.int64)
        if np.any((value == np.iinfo(np.int64).min) & (denominator == -1)):
            raise _errors.BrowserUnsupported('Overflowing integer average division requires the server')
        quotient = np.floor_divide(value, denominator)
        quotient += ((np.remainder(value, denominator) != 0) & ((value < 0) != (denominator < 0)))
        value[...] = quotient
    elif adaptive:
        np.divide(value, (ey - sy).astype(value.dtype)[:, None], out=value)
        np.divide(value, (ex - sx).astype(value.dtype), out=value)
    else:
        np.divide(value, np.asarray(divisor, dtype=value.dtype), out=value)
    return value


def _scan(data, geometry):
    sy, ey, sx, ex, _, _ = geometry
    value = np.zeros((*data.shape[:-2], len(sy), len(sx)), dtype=_accumulator(data.dtype))
    kh, kw = int((ey - sy).max(initial=0)), int((ex - sx).max(initial=0))
    for y in range(kh):
        iy = sy + y;valid_y = iy < ey
        for x in range(kw):
            ix = sx + x;valid = valid_y[:, None] & (ix < ex)
            source = data[..., np.minimum(iy, data.shape[-2] - 1)[:, None], np.minimum(ix, data.shape[-1] - 1)]
            np.add(value, source, out=value, where=valid)
    return _divide(value, geometry).astype(data.dtype, copy=False)


def _packing_bytes(data, geometry):
    sy, ey, sx, ex, _, _ = geometry
    taps = int((ey - sy).max(initial=0)) * int((ex - sx).max(initial=0))
    points = math.prod(data.shape[:-2]) * len(sy) * len(sx)
    return points * taps * (data.itemsize + _accumulator(data.dtype).itemsize + 2) + (len(sy) + len(sx)) * taps * 16


def _packed(data, geometry):
    sy, ey, sx, ex, _, _ = geometry
    kh, kw = int((ey - sy).max(initial=0)), int((ex - sx).max(initial=0))
    iy = sy[:, None] + np.arange(kh);ix = sx[:, None] + np.arange(kw)
    valid = (iy < ey[:, None])[:, None, :, None] & (ix < ex[:, None])[None, :, None, :]
    values = data[..., np.minimum(iy, data.shape[-2] - 1)[:, None, :, None],
                  np.minimum(ix, data.shape[-1] - 1)[None, :, None, :]]
    packed = np.array(values, dtype=_accumulator(data.dtype), order='C', copy=True)
    np.copyto(packed, 0, where=~valid)
    packed = packed.reshape(*data.shape[:-2], len(sy), len(sx), kh * kw)
    # Accumulate follows each region's native row-major order. The final +0
    # preserves native all-negative-zero behavior without retaining prefixes.
    np.add.accumulate(packed, axis=-1, out=packed)
    value = np.add(packed[..., -1], 0)
    return _divide(value, geometry).astype(data.dtype, copy=False)


def _subgeometry(geometry, y, x):
    sy, ey, sx, ex, divisor, adaptive = geometry
    if isinstance(divisor, np.ndarray):divisor = divisor[y, x]
    return sy[y], ey[y], sx[x], ex[x], divisor, adaptive


def _stream_region(data, geometry):
    sy, ey, sx, ex = geometry[:4]
    flat = data[int(sy[0]):int(ey[0]), int(sx[0]):int(ex[0])].flat
    dtype = _accumulator(data.dtype);total = dtype.type(0)
    elements = max(1, (min(_WORKSPACE_BYTES, 4 * 1024 * 1024) - dtype.itemsize) // (data.itemsize + dtype.itemsize))
    for start in range(0, len(flat), elements):
        stop = min(start + elements, len(flat));buffer = np.empty(stop - start + 1, dtype=dtype)
        buffer[0] = total;buffer[1:] = flat[start:stop]
        np.add.accumulate(buffer, out=buffer);total = buffer[-1]
    value = np.asarray(total).reshape(1,1)
    return _divide(value, geometry).astype(data.dtype, copy=False)


def _tiled_packed(data, geometry):
    sy, _, sx = geometry[:3]
    if _packing_bytes(data, geometry) <= _WORKSPACE_BYTES:return _packed(data, geometry)
    value = np.empty((*data.shape[:-2], len(sy), len(sx)), dtype=data.dtype)
    if data.ndim > 2:
        for i in range(data.shape[0]):value[i] = _tiled_packed(data[i], geometry)
    elif len(sy) > 1:
        mid = len(sy) // 2
        value[:mid] = _tiled_packed(data, _subgeometry(geometry, slice(None,mid), slice(None)))
        value[mid:] = _tiled_packed(data, _subgeometry(geometry, slice(mid,None), slice(None)))
    elif len(sx) > 1:
        mid = len(sx) // 2
        value[:,:mid] = _tiled_packed(data, _subgeometry(geometry, slice(None), slice(None,mid)))
        value[:,mid:] = _tiled_packed(data, _subgeometry(geometry, slice(None), slice(mid,None)))
    else:return _stream_region(data, geometry)
    return value


def _forward(data, geometry, partition=None):
    sy, ey, sx, ex, _, _ = geometry
    if not len(sy) or not len(sx) or not data.size: return np.empty((*data.shape[:-2], len(sy), len(sx)), dtype=data.dtype)
    if _ALGORITHM not in ('auto', 'scan', 'packed', 'matrix', 'block'): raise ValueError('Unknown internal average-pooling algorithm')
    with np.errstate(all='ignore'):
        if geometry[-1] and (len(sy), len(sx)) == data.shape[-2:]:
            value = np.empty(data.shape, dtype=data.dtype);np.add(data, 0, out=value);return value
        if (partition is not None and _ALGORITHM in ('auto', 'block')
                and data.size * (data.itemsize + _accumulator(data.dtype).itemsize) <= _WORKSPACE_BYTES):
            return _partition_forward(data, geometry, partition)
        if _ALGORITHM in ('auto', 'matrix'):
            accumulator = _accumulator(data.dtype)
            matrices = _matrix_plan(data.shape, geometry, accumulator)
            if matrices is not None and _matrix_safe(data, matrices[4]):
                rows, columns, height, width = matrices[:4]
                value = (rows @ data) @ columns.T if len(sy) * data.shape[-1] <= data.shape[-2] * len(sx) else rows @ (data @ columns.T)
                np.divide(value, height, out=value);np.divide(value, width, out=value)
                return value.astype(data.dtype, copy=False)
        if _ALGORITHM != 'scan':
            plan = _pack_plan(data.shape, geometry, data.dtype)
            estimate = (math.prod(data.shape[:-2]) * plan.plane_bytes + plan.index_bytes
                        if plan is not None else _packing_bytes(data, geometry))
            if estimate <= _WORKSPACE_BYTES:
                return _packed_from_plan(data, plan) if plan is not None else _packed(data, geometry)
        if _ALGORITHM != 'scan' and _WORKSPACE_BYTES >= 256:return _tiled_packed(data, geometry)
        return _scan(data, geometry)


def _inverse_geometry(shape, geometry):
    ranges = []
    for length, start, end in zip(shape[-2:], (geometry[0], geometry[2]), (geometry[1], geometry[3])):
        index = np.arange(length, dtype=np.int64)
        first = np.searchsorted(end, index, side='right')
        last = np.searchsorted(start, index, side='right')
        first.setflags(write=False);last.setflags(write=False)
        ranges.extend((first, last))
    return (*ranges, 1, False)


@lru_cache(maxsize=64)
def _cached_adjoint_geometry(h, w, oh, ow):
    return _inverse_geometry((h, w), _adaptive_geometry((h, w), (oh, ow)))


def _backward(seed, shape, geometry, nonoverlap, partition=None):
    sy, ey, sx, ex, _, adaptive = geometry
    if not len(sy) or not len(sx) or adaptive and not shape[-3]:
        raise RuntimeError('Average-pooling backward requires nonempty output spatial/channel axes')
    if _BACKWARD_ALGORITHM not in ('auto', 'loop', 'scatter', 'packed', 'block'):
        raise ValueError('Unknown internal average-pooling backward algorithm')
    if partition is not None and _BACKWARD_ALGORITHM in ('auto', 'block'):
        kh, kw = partition
        if adaptive:
            delta = np.divide(seed, seed.dtype.type(kh), dtype=seed.dtype, order='C')
            np.divide(delta, seed.dtype.type(kw), out=delta)
        else:
            delta = np.divide(seed, np.asarray(geometry[4], dtype=seed.dtype), dtype=seed.dtype, order='C')
        result = np.empty(shape, dtype=seed.dtype)
        view = result.reshape(*shape[:-2], len(sy), kh, len(sx), kw)
        np.add(delta[..., :, None, :, None], 0, out=view)
        return result
    if adaptive and (len(sy), len(sx)) == shape[-2:]:return np.add(seed, 0)
    plan = _pack_plan(shape, geometry, seed.dtype)
    delta = np.array(seed, copy=True, order='C')
    if plan is None:_divide(delta, geometry)
    else:
        np.divide(delta, plan.height, out=delta);np.divide(delta, plan.width, out=delta)
    points = len(sy) * len(sx);planes = math.prod(shape[:-2])
    if not planes: return np.zeros(shape, dtype=seed.dtype)

    # Even one contribution per input must fit before inverse-index allocation.
    # Account separately for the arange/searchsorted construction peak.
    if (adaptive and _BACKWARD_ALGORITHM in ('auto', 'packed')
            and max(planes * shape[-2] * shape[-1] * (seed.itemsize * 2 + 2) + sum(shape[-2:]) * 16,
                    sum(shape[-2:]) * 24) <= _WORKSPACE_BYTES):
        inverse_plan = _pack_plan(shape, geometry, seed.dtype, inverse=True)
        if inverse_plan is not None and planes * inverse_plan.plane_bytes + inverse_plan.index_bytes <= _WORKSPACE_BYTES:
            return _packed_from_plan(delta, inverse_plan)
        inverse = (_cached_adjoint_geometry(*shape[-2:], len(sy), len(sx))
                   if sum(shape[-2:]) <= 2048 else _inverse_geometry(shape, geometry))
        if _packing_bytes(delta, inverse) <= _WORKSPACE_BYTES:
            return _packed(delta, inverse)
    kh, kw = int((ey - sy).max()), int((ex - sx).max());taps = kh * kw
    result = np.zeros(shape, dtype=seed.dtype)
    if (_BACKWARD_ALGORITHM == 'loop' or _BACKWARD_ALGORITHM == 'auto' and points <= 16
            or taps * 64 > _WORKSPACE_BYTES):
        for y, (start_y, end_y) in enumerate(zip(sy, ey)):
            for x, (start_x, end_x) in enumerate(zip(sx, ex)):
                result[..., start_y:end_y, start_x:end_x] += delta[..., y, x, None, None]
        return result
    target = result.reshape(planes, shape[-2] * shape[-1]);flat = delta.reshape(planes, points)
    max_planes = max(1, min(planes, _WORKSPACE_BYTES // max(1, taps * 64)))
    offsets = np.arange(taps, dtype=np.int64)
    for first in range(0, planes, max_planes):
        last = min(first + max_planes, planes);count = last - first
        block = max(1, min(points, _WORKSPACE_BYTES // max(1, count * taps * 64)))
        for start in range(0, points, block):
            stop = min(start + block, points);positions = np.arange(start, stop)
            y, x = positions // len(sx), positions % len(sx)
            iy = sy[y, None] + offsets // kw;ix = sx[x, None] + offsets % kw
            valid = (iy < ey[y, None]) & (ix < ex[x, None])
            indices = (iy * shape[-1] + ix)[valid]
            values = np.broadcast_to(flat[first:last, start:stop, None], (count, stop - start, taps))[:, valid]
            key = np.arange(first, last)[:, None], indices
            if nonoverlap: target[key] += values
            else: np.add.at(target, key, values)
    return result


def _apply(input, geometry, nonoverlap, partition=None):
    if isinstance(input, _autograd.ForwardTensor):
        primal = _apply(input._primal, geometry, nonoverlap, partition)
        tangent = _apply(input._tangent, geometry, nonoverlap, partition)
        return _autograd.ForwardTensor(primal, tangent)
    shape = input.shape
    data = _forward(input._data, geometry, partition)
    return _autograd.result(data, (input, lambda seed:_backward(seed, shape, geometry, nonoverlap, partition)), versioned=(input,))


def adaptive_avg_pool2d(input, output_size):
    _input(input, adaptive=True)
    out = _output_size(output_size, input.shape[-2:]);_allocation(input.shape, out)
    if out == (1, 1): return _global_mean(input)
    geometry = _adaptive_geometry(input.shape, out)
    nonoverlap = bool(out[0] and out[1] and input.shape[-2] % out[0] == 0 and input.shape[-1] % out[1] == 0)
    partition = (input.shape[-2] // out[0], input.shape[-1] // out[1]) if nonoverlap else None
    return _apply(input, geometry, nonoverlap, partition)


def _global_mean(input):
    if isinstance(input, _autograd.ForwardTensor):
        return _autograd.ForwardTensor(_global_mean(input._primal), _global_mean(input._tangent))
    shape = input.shape;count = shape[-2] * shape[-1];dtype = input.dtype
    with np.errstate(all='ignore'):
        accumulator = _accumulator(dtype)
        value = input._data.sum(axis=(-2,-1), keepdims=True, dtype=accumulator)
        np.divide(value, accumulator.type(count), out=value)
        value = value.astype(dtype, copy=False)
    def backward(seed):return np.broadcast_to(np.divide(seed, count, dtype=seed.dtype), shape)
    return _autograd.result(value, (input, backward), saves_tensors=False)


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    _input(input, adaptive=False)
    if type(count_include_pad) is not bool: raise TypeError('count_include_pad must be bool')
    if divisor_override is not None:
        if not isinstance(divisor_override, numbers.Integral) or isinstance(divisor_override, (bool, np.bool_)):
            raise TypeError('divisor_override must be an integer')
        divisor_override = int(divisor_override)
        if not -(2**63) <= divisor_override < 2**63: raise OverflowError('divisor_override exceeds int64')
        if divisor_override == 0: raise RuntimeError('divisor_override must not be zero')
    kernel, stride, padding, _, out = _pooling._geometry(input, kernel_size, stride, padding, 1, ceil_mode)
    _allocation(input.shape, out)
    ranges, counts = [], []
    for size, count, k, s, p in zip(input.shape[-2:], out, kernel, stride, padding):
        start = np.arange(count, dtype=np.int64) * s - p
        end = np.minimum(start + k, size + p)
        real_start, real_end = np.maximum(start, 0), np.minimum(end, size)
        ranges.extend((real_start, real_end))
        counts.append(end - start if count_include_pad else real_end - real_start)
    partition = kernel if stride == kernel and padding == (0,0) and tuple(n*k for n,k in zip(out,kernel)) == input.shape[-2:] else None
    divisor = divisor_override if divisor_override is not None else math.prod(kernel) if partition else counts[0][:, None] * counts[1]
    return _apply(input, (*ranges, divisor, False), all(s >= k for s, k in zip(stride, kernel)), partition)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):super().__init__();self.output_size = output_size
    def forward(self, input):return adaptive_avg_pool2d(input, self.output_size)
    def extra_repr(self):return f'output_size={self.output_size}'


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__();self.kernel_size = kernel_size;self.stride = kernel_size if stride is None else stride
        self.padding, self.ceil_mode = padding, ceil_mode
        self.count_include_pad, self.divisor_override = count_include_pad, divisor_override
    def forward(self, input):
        return avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
    def extra_repr(self):return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'
