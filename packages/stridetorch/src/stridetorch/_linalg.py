"""NumPy-backed linalg implementation; extracted from PaperLeet."""
from collections import namedtuple
import numpy as np


def _solve_geometry(a, b, left):
    if a.ndim < 2 or a.shape[-2] != a.shape[-1]:
        raise RuntimeError('linalg.solve expects square matrices')
    if b.ndim < 1:raise RuntimeError('linalg.solve expects a vector or matrix right-hand side')
    # Native Torch treats only an exact A.shape[:-1] batch as vectors, in
    # addition to a single vector. Rank alone is not enough to disambiguate B.
    vector = b.ndim == 1 or b.ndim == a.ndim-1 and b.shape == a.shape[:-1]
    if vector and not left:raise RuntimeError('Vector right-hand sides require left=True')
    matrix_shape = b.shape+(1,) if vector else b.shape
    if matrix_shape[-2 if left else -1] != a.shape[-1]:
        raise RuntimeError('linalg.solve matrix dimensions do not match')
    a_batch,b_batch = a.shape[:-2],matrix_shape[:-2]
    try:batch = a_batch if a_batch == b_batch else np.broadcast_shapes(a_batch,b_batch)
    except ValueError:raise RuntimeError('linalg.solve batch dimensions cannot broadcast') from None
    return vector, batch, matrix_shape


def _solve_values(a, b, left):
    # Equal-rank matrix operands behave consistently in NumPy1.26 and2.x.
    # Vector classification has already happened before explicit promotion.
    if not left:a,b = np.swapaxes(a,-1,-2),np.swapaxes(b,-1,-2)
    try:value = np.linalg.solve(a,b)
    except np.linalg.LinAlgError as error:raise RuntimeError('The matrix is singular') from error
    return value if left else np.swapaxes(value,-1,-2)


def solve(A, B, *, left=True, out=None):
    if not isinstance(A,_tensor.Tensor) or not isinstance(B,_tensor.Tensor):
        raise TypeError('linalg.solve inputs must be Tensors')
    if type(left) is not bool:raise TypeError('left must be bool')
    if out is not None:raise _errors.BrowserUnsupported('Solve output buffers require the server')
    dtype = A.dtype
    if dtype != B.dtype:raise RuntimeError('solve expects matching dtypes')
    if dtype.char not in 'fdFD':
        raise _errors.BrowserUnsupported('solve requires float32/64 or complex64/128 inputs')
    vector,batch,bshape = _solve_geometry(A,B,left)
    if any(isinstance(x,_autograd.ForwardTensor) for x in (A,B)):
        a = A._primal if isinstance(A,_autograd.ForwardTensor) else A
        b = B._primal if isinstance(B,_autograd.ForwardTensor) else B
        value = solve(a,b,left=left)
        residual = B._tangent if isinstance(B,_autograd.ForwardTensor) else None
        # A materialized zero direction keeps native zero reverse edges when
        # only B participates in JVP; omitting the product would yield None.
        direction = A._tangent if isinstance(A,_autograd.ForwardTensor) else _tensor.Tensor(np.zeros_like(a._data))
        product = (direction @ value.unsqueeze(-1)).squeeze(-1) if vector else (direction @ value if left else value @ direction)
        residual = -product if residual is None else residual-product
        # A broadcast vector result can look like a matrix to the public parser;
        # make its matrix interpretation explicit for the tangent solve.
        # Native JVP reuses a factorization whose reverse tape is constant.
        # Keep differentiation through X/residual, not through that factor.
        factor = a.detach()
        tangent = solve(factor,residual.unsqueeze(-1),left=True).squeeze(-1) if vector else solve(factor,residual,left=left)
        return _autograd.ForwardTensor(value,_layout.matching_tangent(tangent,value))
    a_shape,b_shape = A.shape,B.shape
    a_data = A._data
    matrix_a = a_data if a_shape[:-2] == batch else np.broadcast_to(a_data,batch+a_shape[-2:])
    matrix_b = B._data.reshape(bshape) if vector else B._data
    if bshape[:-2] != batch:matrix_b = np.broadcast_to(matrix_b,batch+bshape[-2:])
    values = _solve_values(matrix_a,matrix_b,left)
    # LAPACK-style native result geometry: each left-solve matrix is column
    # major; right-solve matrices and vector results are contiguous.
    if vector:values = values[...,0]
    def contiguous(data):
        # NumPy keeps arbitrary singleton strides when an array is already
        # contiguous; copy those descriptors to obtain native conventional ones.
        return _layout.contiguous_empty(data.copy(order='C')) if not data.size or 1 in data.shape else np.ascontiguousarray(data)
    if left and values.ndim >= 2:values = np.swapaxes(contiguous(np.swapaxes(values,-1,-2)),-1,-2)
    else:values = contiguous(values)
    if not _state._grad_enabled or not (A.requires_grad or B.requires_grad):
        return _tensor.Tensor(values,_metadata_owned=True)
    matrix_x = values[...,None] if vector else values
    x_transpose = np.swapaxes(matrix_x,-1,-2) if A.requires_grad else None
    coefficient = (matrix_a.T if matrix_a.ndim == 2 else np.swapaxes(matrix_a,-1,-2)) if left else matrix_a
    def right_gradient(g):
        incoming = g[...,None] if vector else g
        db = np.linalg.solve(coefficient,incoming if left else np.swapaxes(incoming,-1,-2))
        return db if left else np.swapaxes(db,-1,-2)
    def reduce_a(da):return da if da.shape == a_shape else _ops.unbroadcast(da,a_shape)
    def reduce_b(db):
        if vector:db = db[...,0]
        return db if db.shape == b_shape else _ops.unbroadcast(db,b_shape)
    def left_gradient(g):
        db = right_gradient(g)
        return reduce_a(-(db @ x_transpose if left else x_transpose @ db))
    def backward(g,needs):
        db = right_gradient(g)
        return reduce_a(-(db @ x_transpose if left else x_transpose @ db)),reduce_b(db)
    if A.requires_grad and B.requires_grad:
        output = _autograd.joint_result(values,(A,B),backward,versioned=())
    elif A.requires_grad:output = _autograd.result(values,(A,left_gradient),versioned=())
    else:output = _autograd.result(values,(B,lambda g:reduce_b(right_gradient(g))),versioned=())
    if output.requires_grad:
        # Native solve saves A and X, but not B. The shared pullback solves the
        # adjoint system once, irrespective of which input gradients are active.
        counters = {A._storage,output._storage}
        output._grad_node.versions = tuple((counter,counter.version) for counter in counters)
    return output


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
from . import _autograd, _errors, _layout, _ops, _state, _tensor
