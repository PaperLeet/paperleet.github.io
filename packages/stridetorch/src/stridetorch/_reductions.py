"""Checked reduction axes and NaN-aware real reductions."""
import numpy as np
from . import _layout


MISSING = _layout.UNSPECIFIED_DIM


def arguments(dim, keepdim, *, name, explicit_dim=True):
    if dim is MISSING:
        if explicit_dim and keepdim is not MISSING:
            raise TypeError(f'{name} keepdim requires an explicit dimension')
        dim = None
    if keepdim is MISSING:keepdim = False
    if type(keepdim) is not bool:raise TypeError(f'{name} keepdim must be bool')
    if dim is None:return (), keepdim
    if type(dim) is int:return (dim,), keepdim
    requested = tuple(dim) if isinstance(dim,(tuple,list)) else (dim,)
    return tuple(_layout.integer(d) for d in requested), keepdim


def axes(requested, ndim):
    if not requested:return tuple(range(ndim))
    if len(requested)==1:
        axis = _layout.dimension(requested[0],ndim)
        return (axis,) if ndim else ()
    normalized = tuple(_layout.dimension(d,ndim) for d in requested)
    if len(set(normalized)) != len(normalized):raise RuntimeError('Duplicate reduction dimensions')
    return tuple(range(ndim)) if not requested else normalized if ndim else ()


def expand(seed, dims, keepdim):
    if not keepdim:
        for axis in sorted(dims):seed = np.expand_dims(seed,axis)
    return seed


def divide(value, count, dtype):
    # Half division uses opmath even when an integer count exceeds half's range.
    accumulator = np.float32 if dtype == np.dtype('float16') else dtype
    return np.divide(value,count,dtype=accumulator).astype(dtype,copy=False)


def nan_reduce(input, dim=None, keepdim=False, *, dtype=None, mean=False):
    from . import _autograd, _state
    name = 'nanmean' if mean else 'nansum'
    requested, keepdim = arguments(dim,keepdim,name=name,explicit_dim=False)
    source_dtype = input.dtype
    target = (np.dtype('int64') if source_dtype.kind in 'biu' and not mean else source_dtype) if dtype is None else np.dtype(dtype)
    if mean and (source_dtype.kind not in 'fc' or target.kind not in 'fc'):
        raise RuntimeError('nanmean requires floating input and output dtypes')
    dims = axes(requested,input.ndim)
    if source_dtype.kind == 'c':
        raise RuntimeError(f'{name} does not support complex CPU inputs')
    if source_dtype.kind == 'f' and target.kind == 'c':
        raise NotImplementedError(f'{name} does not support complex CPU kernels')
    if target.kind not in 'fc' and input.requires_grad and _state._grad_enabled:
        raise RuntimeError('Autograd does not support a non-floating nansum dtype')
    if isinstance(input,_autograd.ForwardTensor):
        return _forward(input,dim,keepdim,target,mean,dims)
    shape = input.shape
    with np.errstate(all='ignore'):
        mask = ~np.isnan(input._data)
        if source_dtype.kind == 'f' and target.kind not in 'fc':
            # Native's integer-output path replaces NaNs and infinities before
            # casting. Out-of-range integer values remain platform dependent.
            value = np.nan_to_num(input._data,nan=0.).sum(axis=dims,keepdims=keepdim,dtype=target)
        else:
            value = input._data.sum(axis=dims,keepdims=keepdim,dtype=target,where=mask)
        count = mask.sum(axis=dims,keepdims=keepdim,dtype=np.int64) if mean else None
        if mean:value = divide(value,count,target)

    def backward(seed):
        if mean:seed = divide(seed,count,target)
        # Cast at the forward input dtype before multiplying the current mask.
        # Multiplication deliberately preserves 0*Inf and 0*NaN as NaN.
        seed = expand(seed.astype(source_dtype,copy=False),dims,keepdim)
        return seed * ~np.isnan(input._data)

    return _autograd.result(value,(input,backward),versioned=(input,))


def _forward(input, dim, keepdim, target, mean, dims):
    from . import _autograd, _errors
    if target not in (np.dtype('float32'),np.dtype('float64')):
        raise _errors.BrowserUnsupported('Forward mode requires real float32/float64 outputs')
    primal, tangent = input._primal, input._tangent
    value = nan_reduce(primal,dim,keepdim,dtype=target,mean=mean)
    shape, source_dtype = primal.shape, primal.dtype
    with np.errstate(all='ignore'):
        mask = ~np.isnan(primal._data)
        count = mask.sum(axis=dims,keepdims=keepdim,dtype=np.int64) if mean else None
        # WHERE excludes a NaN tangent only at an ignored primal position.
        raw = np.where(mask,tangent._data,0).sum(axis=dims,keepdims=keepdim,dtype=target)
        if mean:raw = divide(raw,count,target)

    def backward(seed):
        if mean:seed = divide(seed,count,target)
        seed = expand(seed.astype(source_dtype,copy=False),dims,keepdim)
        return np.where(mask,seed,0)

    product = _autograd.result(raw,(tangent,backward),versioned=())
    if mean and primal.requires_grad:
        # Native nanmean's composite JVP has a ZeroTensor primal edge. Keep it
        # exact even at Inf/NaN, with its own guard so target pruning still works.
        zero = _autograd.result(np.zeros((),dtype=target),
            (primal,lambda seed:np.zeros(shape,dtype=source_dtype)),versioned=(primal,))
        product = _autograd.result(raw,(product,lambda seed:seed),
            (zero,lambda seed:np.zeros((),dtype=target)),saves_tensors=False)
    return _autograd.ForwardTensor(value,product)
