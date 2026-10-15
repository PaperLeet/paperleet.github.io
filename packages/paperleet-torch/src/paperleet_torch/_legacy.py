"""Bounded torch-like surface for the Schedules browser pilot, backed by NumPy.

This is not a general PyTorch replacement. Unsupported tensor/autograd operations
raise BrowserUnsupported so the caller can use the native server judge instead.
"""
import types
import numpy as np


class BrowserUnsupported(NotImplementedError):
    pass


class Tensor(np.ndarray):
    __array_priority__ = 1000

    def __new__(cls, data, dtype=None):
        return np.asarray(data, dtype=dtype).view(cls)

    def __getitem__(self, index):
        return Tensor(super().__getitem__(index))

    @property
    def device(self):
        return "cpu"

    def to(self, other=None, *, dtype=None, device=None):
        if device not in (None, "cpu"):
            raise BrowserUnsupported("Only CPU/WASM tensors are supported")
        if isinstance(other, np.ndarray):
            dtype = other.dtype
        elif other is not None:
            dtype = other
        return Tensor(self, dtype=dtype)

    def double(self):
        return self.to(dtype=np.float64)

    def float(self):
        return self.to(dtype=np.float32)

    def clone(self):
        return Tensor(np.array(self, copy=True))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def sqrt(self):
        return Tensor(np.sqrt(np.asarray(self)))

    def square(self):
        return Tensor(np.square(np.asarray(self)))

    def log(self):
        return Tensor(np.log(np.asarray(self)))

    def abs(self):
        return Tensor(np.abs(np.asarray(self)))

    def cumprod(self, dim=0, **kwargs):
        return Tensor(np.cumprod(np.asarray(self), axis=kwargs.pop("axis", dim), **kwargs))

    def cumsum(self, dim=0, **kwargs):
        return Tensor(np.cumsum(np.asarray(self), axis=kwargs.pop("axis", dim), **kwargs))

    def var(self, dim=None, unbiased=True, *, correction=None, keepdim=False):
        ddof = int(unbiased) if correction is None else correction
        return Tensor(np.var(np.asarray(self), axis=dim, ddof=ddof, keepdims=keepdim))

    def clamp(self, min=None, max=None):
        return Tensor(np.clip(np.asarray(self), min, max))

    def view(self, *shape, **kwargs):
        if not shape or isinstance(shape[0], (type, np.dtype)) or kwargs:
            return super().view(*shape, **kwargs)
        return self.reshape(*shape)

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(np.asarray(self), dim))

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        leading = (1,) * (len(shape) - self.ndim) + self.shape
        return Tensor(np.broadcast_to(np.asarray(self), tuple(old if new == -1 else new for old, new in zip(leading, shape))))

    def expand_as(self, other):
        return self.expand(other.shape)

    def __getattr__(self, name):
        # NumPy probes these private attributes during normal ufunc dispatch.
        if name.startswith("_"):
            raise AttributeError(name)
        raise BrowserUnsupported(f"Tensor.{name} is outside the Schedules browser backend")


def _shape(args):
    return tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)


def _check(kwargs):
    if kwargs.pop("requires_grad", False):
        raise BrowserUnsupported("Autograd requires the server judge")
    device = kwargs.pop("device", "cpu")
    if device != "cpu":
        raise BrowserUnsupported("Only CPU/WASM tensors are supported")
    return kwargs


def tensor(data, dtype=None, **kwargs):
    _check(kwargs)
    if kwargs:
        raise BrowserUnsupported(f"Unsupported tensor arguments: {list(kwargs)}")
    # Match torch's default float32 and inferred int64 conventions.
    array = np.asarray(data)
    if dtype is None and not isinstance(data, (np.ndarray, np.generic)):
        if array.dtype.kind == "f":
            dtype = np.float32
        elif array.dtype.kind in "iu":
            dtype = np.int64
    return Tensor(np.array(array, dtype=dtype, copy=True))


def _create(fn):
    def create(*shape, dtype=np.float32, **kwargs):
        _check(kwargs)
        if kwargs:
            raise BrowserUnsupported(f"Unsupported creation arguments: {list(kwargs)}")
        return Tensor(fn(_shape(shape)), dtype=dtype)
    return create


def _like(fn):
    return lambda x, dtype=None: Tensor(fn(np.asarray(x)), dtype=dtype or x.dtype)


def _unsupported(name):
    if name.startswith("_"):
        raise AttributeError(name)
    raise BrowserUnsupported(f"torch.{name} is outside the Schedules browser backend")


def install():
    torch = types.ModuleType("torch")
    torch._browser_unsupported = BrowserUnsupported
    torch.Tensor = Tensor
    torch.float64, torch.float32, torch.long, torch.int64, torch.bool = np.dtype("float64"), np.dtype("float32"), np.dtype("int64"), np.dtype("int64"), np.dtype("bool")
    torch.tensor = tensor
    torch.as_tensor = lambda data: Tensor(data) if isinstance(data, np.ndarray) else tensor(data)
    torch.linspace = lambda start, end, steps, dtype=np.float32: Tensor(np.linspace(start, end, steps, dtype=dtype))
    torch.arange = lambda *args, dtype=None: Tensor(np.arange(*args, dtype=dtype or (np.float32 if any(isinstance(x, float) for x in args) else np.int64)))
    torch.full = lambda shape, fill_value, dtype=None: Tensor(np.full(shape, fill_value, dtype=dtype or (np.bool_ if isinstance(fill_value, bool) else np.float32 if isinstance(fill_value, float) else np.int64)))
    torch.zeros, torch.ones, torch.empty = _create(np.zeros), _create(np.ones), _create(np.empty)
    torch.randn = _create(lambda shape: np.random.standard_normal(shape))
    torch.randn_like = lambda x: Tensor(np.random.standard_normal(x.shape), dtype=x.dtype)
    torch.ones_like, torch.zeros_like, torch.empty_like = _like(np.ones_like), _like(np.zeros_like), _like(np.empty_like)
    torch.full_like = lambda x, fill_value: Tensor(np.full_like(np.asarray(x), fill_value))
    torch.cat = lambda tensors, dim=0: Tensor(np.concatenate([np.asarray(x) for x in tensors], axis=dim))
    torch.stack = lambda tensors, dim=0: Tensor(np.stack([np.asarray(x) for x in tensors], axis=dim))
    for name in ("cos", "sin", "sqrt", "log", "log1p", "exp", "isfinite", "isnan", "nan_to_num"):
        fn = getattr(np, name)
        setattr(torch, name, lambda x, _fn=fn, **kwargs: Tensor(_fn(np.asarray(x), **kwargs)))
    torch.where = lambda condition, x, y: Tensor(np.where(condition, x, y))
    torch.isclose = lambda x, y, rtol=1e-5, atol=1e-8: Tensor(np.isclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol))
    torch.unravel_index = lambda index, shape: np.unravel_index(int(index), shape)
    torch.manual_seed = np.random.seed
    torch.__getattr__ = _unsupported
    return torch
