"""NumPy-backed linalg implementation; extracted from PaperLeet."""
from collections import namedtuple
import numpy as np


def solve(a, b):
    a, b = _ops.wrap(a), _ops.wrap(b)
    if a.ndim != 2 or b.ndim not in (1, 2): raise _errors.BrowserUnsupported('Batched linear solves require the server')
    if a.dtype != b.dtype or a.dtype.kind != 'f': raise RuntimeError('solve expects matching floating dtypes')
    a_data, b_ndim = a._data, b.ndim
    value = np.linalg.solve(a_data, b._data)
    def db(g): return np.linalg.solve(a_data.T, g)
    def da(g):
        incoming = db(g)
        return -(incoming[:, None] @ value[None, :] if b_ndim == 1 else incoming @ value.T)
    return _autograd.result(value, (a, da), (b, db))


def inv(input, *, out=None):
    if out is not None: raise _errors.BrowserUnsupported('Inverse output buffers require the server')
    real_matrix(input)
    if input.shape[-1] != input.shape[-2]: raise RuntimeError('Inverse expects square matrices')
    try: value = np.linalg.inv(input._data)
    except np.linalg.LinAlgError as error: raise RuntimeError('The matrix is singular') from error
    transpose = np.swapaxes(value, -1, -2)
    return _autograd.result(value, (input, lambda g: -transpose @ g @ transpose))


def real_matrix(input):
    if input.ndim < 2: raise RuntimeError('Linear algebra expects matrices')
    if input.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise _errors.BrowserUnsupported('This linear algebra backend requires real float32/float64 matrices')
    if not np.isfinite(input._data).all(): raise RuntimeError('Linear algebra requires finite inputs')


def svd(input, full_matrices=True, *, driver=None, out=None):
    """Real batched SVD with analytical vector/rectangular-complement VJPs."""
    real_matrix(input)
    source = input._data
    if driver is not None or out is not None: raise _errors.BrowserUnsupported('SVD drivers/output buffers require the server')
    if not isinstance(full_matrices, bool): raise TypeError('full_matrices must be boolean')
    all_u, singular, all_vh = np.linalg.svd(input._data, full_matrices=full_matrices)
    rank = singular.shape[-1]
    u, vh = all_u[..., :, :rank], all_vh[..., :rank, :]
    ut, v = np.swapaxes(u,-1,-2), np.swapaxes(vh,-1,-2)
    saved = tuple(_tensor.Tensor(value) for value in (all_u, singular, all_vh))
    def ds(g): return (u * g[..., None, :]) @ vh
    def vector_factors():
        # Do not fabricate derivatives at rank boundaries or repeated spectra.
        with np.errstate(over='ignore',invalid='ignore'):
            squared = singular ** 2
            gap = squared[..., None, :] - squared[..., :, None]
        diagonal = np.eye(rank, dtype=bool)
        cutoff = np.finfo(source.dtype).eps * max(source.shape[-2:])
        if not np.isfinite(squared).all() or np.any(singular <= cutoff * singular[..., :1]) or np.any((np.abs(gap) <= cutoff * squared[..., :1, None]) & ~diagonal):
            raise _errors.BrowserUnsupported('Degenerate or numerically unresolved SVD vector gradients require the server')
        return np.divide(1, gap, out=np.zeros_like(gap), where=~diagonal)
    def du(g):
        if not rank: return np.zeros_like(source)
        g = g[..., :, :rank]
        inner = ut @ g
        core = vector_factors() * (inner - np.swapaxes(inner,-1,-2)) * singular[..., None, :]
        gradient = u @ core @ vh
        if input.shape[-2] > rank: gradient += ((g - u @ inner) / singular[..., None, :]) @ vh
        return gradient
    def dvh(g):
        if not rank: return np.zeros_like(source)
        g = g[..., :rank, :]
        inner = vh @ np.swapaxes(g,-1,-2)
        core = vector_factors() * (inner - np.swapaxes(inner,-1,-2)) * singular[..., :, None]
        gradient = u @ core @ vh
        if input.shape[-1] > rank: gradient += u @ ((g - (g @ v) @ vh) / singular[..., :, None])
        return gradient
    pair = namedtuple('SVD', ['U','S','Vh'])
    return pair(_autograd.result(all_u,(input,du),saved=saved), _autograd.result(singular,(input,ds),saved=saved), _autograd.result(all_vh,(input,dvh),saved=saved))


def eigh(input, UPLO='L', *, out=None):
    """Real symmetric eigendecomposition with separate value/vector VJPs."""
    real_matrix(input)
    source_dtype = input.dtype
    if out is not None: raise _errors.BrowserUnsupported('eigh output buffers require the server')
    if input.shape[-2] != input.shape[-1] or UPLO not in ('L','U'): raise RuntimeError('Invalid eigh shape/triangle')
    values, vectors = np.linalg.eigh(input._data, UPLO=UPLO)
    count = input.shape[-1]
    transposed = np.swapaxes(vectors, -1, -2)
    saved = (_tensor.Tensor(values), _tensor.Tensor(vectors))
    def spectral_gaps():
        with np.errstate(over='ignore', invalid='ignore'):
            gaps = values[..., None, :] - values[..., :, None]
        scale = np.max(np.abs(values), axis=-1, keepdims=True, initial=0)
        tolerance = np.finfo(source_dtype).eps * max(count, 1) * scale[..., None]
        unresolved = (np.abs(gaps) <= tolerance) & ~np.eye(count, dtype=bool)
        return gaps, unresolved
    def dl(g):
        _, unresolved = spectral_gaps()
        # At repeated spectra a differentiable spectral objective has equal
        # derivatives within each eigenspace. Otherwise the chosen basis would
        # invent a platform-dependent gradient.
        differences = np.abs(g[..., None, :] - g[..., :, None])
        tolerance = 8 * np.finfo(source_dtype).eps * np.maximum(1, np.max(np.abs(g), axis=-1, keepdims=True, initial=0))[..., None]
        if np.any(unresolved & (differences > tolerance)):
            raise _errors.BrowserUnsupported('Basis-dependent repeated-eigenvalue gradients require the server')
        return (vectors * g[..., None, :]) @ transposed
    def dq(g):
        gaps, unresolved = spectral_gaps()
        if np.any(unresolved) or not np.isfinite(gaps).all():
            raise _errors.BrowserUnsupported('Degenerate or unresolved eigenvector gradients require the server')
        inner = transposed @ g
        core = .5 * np.divide(inner - np.swapaxes(inner, -1, -2), gaps,
                              out=np.zeros_like(inner), where=~np.eye(count, dtype=bool))
        # PyTorch assumes symmetric input: the VJP is symmetric, even when
        # UPLO selected just one triangle for the forward computation.
        return vectors @ core @ transposed
    return namedtuple('Eigh',['eigenvalues','eigenvectors'])(
        _autograd.result(values, (input, dl), saved=saved), _autograd.result(vectors, (input, dq), saved=saved))


def eigvalsh(input, UPLO='L', *, out=None):
    if out is not None: raise _errors.BrowserUnsupported('eigvalsh output buffers require the server')
    return eigh(input, UPLO).eigenvalues


def det(input, *, out=None):
    real_matrix(input)
    if out is not None: raise _errors.BrowserUnsupported('Determinant output buffers require the server')
    if input.shape[-2] != input.shape[-1]: raise RuntimeError('det expects square matrices')
    source = input._data
    value = np.linalg.det(source)
    def vjp(g):
        # Products of all other singular values give the cofactor matrix
        # without dividing by a singular determinant or singular value.
        u, s, vh = np.linalg.svd(source, full_matrices=False)
        ones = np.ones(s.shape[:-1] + (1,), dtype=s.dtype)
        prefix = np.concatenate((ones, np.cumprod(s, axis=-1)), axis=-1)
        suffix = np.concatenate((np.flip(np.cumprod(np.flip(s, axis=-1), axis=-1), axis=-1), ones), axis=-1)
        cofactors = prefix[..., :-1] * suffix[..., 1:]
        orientation = np.linalg.det(u) * np.linalg.det(vh)
        return (g * orientation)[..., None, None] * ((u * cofactors[..., None, :]) @ vh)
    return _autograd.result(value, (input, vjp))


def qr(input, mode='reduced', *, out=None):
    real_matrix(input)
    if out is not None or mode not in ('reduced','complete') or _state._grad_enabled and input.requires_grad:
        raise _errors.BrowserUnsupported('Differentiable/other-mode QR or output buffers require the server')
    q, r = np.linalg.qr(input._data, mode=mode)
    return namedtuple('QR',['Q','R'])(_tensor.Tensor(q),_tensor.Tensor(r))


def linalg_norm(input, ord=None, dim=None, keepdim=False, *, out=None, dtype=None):
    if out is not None: raise _errors.BrowserUnsupported('Norm output buffers require the server')
    if dtype is not None and (np.dtype(dtype).kind != 'f' or np.dtype(dtype).itemsize < input.dtype.itemsize):
        raise _errors.BrowserUnsupported('Narrowing/non-floating norm dtypes require the server')
    axes = _ops.reduction_axes(dim, input.ndim)
    if dim is not None and len(axes) not in (1, 2): raise RuntimeError('linalg.norm requires one or two dimensions')
    if ord is not None:
        if dim is None and input.ndim not in (1, 2): raise RuntimeError('An explicit norm order requires a vector or matrix')
        if ord == 'fro' and len(axes) != 2: raise RuntimeError('Frobenius norm requires two dimensions')
        if ord != 'fro' and (ord != 2 or len(axes) != 1):
            raise _errors.BrowserUnsupported('Other vector norms and spectral matrix norms require the server')
    return input.norm(2, dim, keepdim, dtype)


# Late module references break Tensor/autograd/operator cycles without rebinding functions.
from . import _autograd, _errors, _ops, _state, _tensor
