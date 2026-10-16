"""NumPy-backed random implementation; extracted from PaperLeet."""
import contextlib
import math
import numpy as np


from . import _device

@contextlib.contextmanager
def fork_rng(devices=None, enabled=True, device_type='cuda'):
    if devices and enabled: raise _errors.BrowserUnsupported('Non-CPU RNG states require the server')
    if device_type not in ('cpu', 'cuda'): raise _errors.BrowserUnsupported('This RNG device type requires the server')
    if not enabled:
        yield
        return
    state, normal = np.random.get_state(), _default_generator.normal
    try: yield
    finally:
        np.random.set_state(state)
        _default_generator.normal = normal


class Generator:
    def __init__(self, device=None):
        _device.check_device(device)
        self.rng, self.normal = np.random.RandomState(42), None
    def manual_seed(self, seed):
        if self.rng is np.random: np.random.seed(seed)
        else: self.rng = np.random.RandomState(seed)
        self.normal = None
        return self


_default_generator = Generator()


_default_generator.rng = np.random


def normal_samples(generator, shape, dtype):
    """CPU Box-Muller filling and cached scalar draws, matching PyTorch's stream."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('Normal sampling requires float32/float64')
    size = math.prod(shape)
    rng = generator.rng
    if size < 16:
        values = np.empty(size, dtype=dtype)
        for i in range(size):
            if generator.normal is None:
                u1, u2 = uniform_samples(rng, (2,), np.float64)
                radius, theta = math.sqrt(-2 * math.log1p(-u2)), 2 * math.pi * u1
                value, generator.normal = radius * math.cos(theta), radius * math.sin(theta)
            else:
                value, generator.normal = generator.normal, None
            values[i] = value
        return values.reshape(shape)
    values = uniform_samples(rng, (size,), dtype)
    def transform(blocks):
        radius = np.sqrt(-2 * np.log(1 - blocks[:, :8]))
        theta = (2 * math.pi * blocks[:, 8:].astype(np.float64)).astype(dtype)
        blocks[:, :8], blocks[:, 8:] = radius * np.cos(theta), radius * np.sin(theta)
    transform(values[:size // 16 * 16].reshape(-1, 16))
    if size % 16:
        values[-16:] = uniform_samples(rng, (16,), dtype)
        transform(values[-16:].reshape(1, 16))
    return values.reshape(shape)


def multinomial(probabilities, num_samples, replacement=False, *, generator=None):
    weights = _ops.wrap(probabilities)._data
    if not replacement and num_samples != 1:
        raise _errors.BrowserUnsupported('Multiple draws without replacement require the server')
    if weights.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('Multinomial supports float32/float64 probabilities')
    if weights.ndim not in (1, 2) or not weights.shape[-1] or type(num_samples) is not int or num_samples < 1:
        raise RuntimeError('Invalid multinomial shape or sample count')
    if not np.isfinite(weights).all() or np.any(weights < 0): raise RuntimeError('Invalid multinomial probabilities')
    rng = generator.rng if generator else np.random
    if num_samples == 1:
        if np.any(weights.sum(axis=-1) <= 0): raise RuntimeError('Multinomial probabilities must sum to a positive value')
        # Independent exponential races implement the single-draw categorical law.
        # This is distribution-faithful; its seeded stream is not native PyTorch's.
        uniforms = uniform_samples(rng, weights.shape, weights.dtype)
        with np.errstate(divide='ignore', invalid='ignore'):
            races = -np.log1p(-uniforms) / weights
        races = np.where(weights > 0, races, np.inf)
        return _tensor.Tensor(np.argmin(races, axis=-1)[..., None], dtype=np.int64)
    draws = []
    for row in weights.reshape(-1, weights.shape[-1]):
        cumulative = np.cumsum(row, dtype=weights.dtype)
        if not cumulative[-1] > 0: raise RuntimeError('Multinomial probabilities must sum to a positive value')
        cumulative /= cumulative[-1]; cumulative[-1] = 1
        # Native CPU replacement sampling consumes float64 uniform draws even
        # when the probabilities themselves are float32.
        draws.append(np.searchsorted(cumulative, uniform_samples(rng, (num_samples,), np.float64), side='left'))
    return _tensor.Tensor(np.asarray(draws, dtype=np.int64).reshape(weights.shape[:-1] + (num_samples,)))


def randperm(n, *, generator=None, dtype=np.int64, device=None):
    _device.check_device(device)
    if type(n) is not int or n < 0: raise RuntimeError('randperm needs a non-negative integer')
    if n > 1_000_000 or np.dtype(dtype) != np.dtype('int64'):
        raise _errors.BrowserUnsupported('Large/non-int64 permutations require the server')
    rng = generator.rng if generator else np.random
    values = np.arange(n, dtype=np.int64)
    bits = rng.randint(0, 2**32, size=max(0, n-1), dtype=np.uint32)
    for i, bit in enumerate(bits):
        other = i + int(bit) % (n-i)
        values[i], values[other] = values[other], values[i]
    if not values.size: values = _layout.contiguous_empty(values)
    return _tensor.Tensor(values)


def create(kind, *shape, dtype=np.float32, device=None, requires_grad=False, generator=None):
    _device.check_device(device)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    rng = generator.rng if generator else np.random
    value = normal_samples(generator or _default_generator, shape, dtype) if kind == 'randn' else uniform_samples(rng, shape, dtype) if kind == 'rand' else getattr(np, kind)(shape, dtype=dtype)
    if not value.size: value = _layout.contiguous_empty(np.asarray(value, dtype=dtype))
    return _tensor.Tensor(value, dtype=dtype, requires_grad=requires_grad)


def randint(low=0, high=None, size=None, *, generator=None, dtype=np.int64, device=None):
    _device.check_device(device)
    if size is None and isinstance(high, (tuple, list)):
        size, high, low = high, low, 0
    elif high is None:
        high, low = low, 0
    if not isinstance(low, (int, np.integer)) or not isinstance(high, (int, np.integer)):
        raise TypeError('randint bounds must be integers')
    low, high = int(low), int(high)
    if not isinstance(size, (tuple, list)) or any(not isinstance(s, (int, np.integer)) for s in size):
        raise TypeError('randint requires an integer shape')
    size = tuple(int(s) for s in size)
    if any(s < 0 for s in size): raise RuntimeError('randint dimensions must be nonnegative')
    if math.prod(size) > 16_000_000: raise _errors.BrowserUnsupported('randint allocation exceeds the browser limit')
    if dtype is None: dtype = np.int64
    dtype = np.dtype(dtype)
    if dtype.kind not in 'biuf': raise _errors.BrowserUnsupported('This randint dtype requires the server')
    if dtype.kind in 'biu':
        lower, upper = (0, 1) if dtype.kind == 'b' else (np.iinfo(dtype).min, np.iinfo(dtype).max)
        if low < lower or high - 1 > upper: raise RuntimeError('randint bounds exceed the output dtype')
    rng = generator.rng if generator else np.random
    if not 0 < high-low <= 2**32: raise _errors.BrowserUnsupported('This randint range requires the server')
    bits = rng.randint(0, 2**32, size=size, dtype=np.uint32).astype(np.uint64)
    value = (bits % (high-low)).astype(dtype) + low
    if not value.size: value = _layout.contiguous_empty(value)
    return _tensor.Tensor(value)


def uniform_samples(rng, shape, dtype):
    # CPU MT19937 uniforms use the low mantissa bits, unlike NumPy's
    # random_sample conversion. Keep preference-pair sampling reproducible.
    if np.dtype(dtype) == np.dtype('float64'):
        bits = rng.randint(0,2**32,size=tuple(shape)+(2,),dtype=np.uint32).astype(np.uint64)
        raw = ((bits[...,0] << 32) | bits[...,1]) & np.uint64((1 << 53)-1)
        return raw.astype(np.float64) / (1 << 53)
    bits = rng.randint(0,2**32,size=shape,dtype=np.uint32)
    return (bits & ((1 << 24)-1)).astype(np.float32) / (1 << 24)


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _errors, _layout, _ops, _tensor
