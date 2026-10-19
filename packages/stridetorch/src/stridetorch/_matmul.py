"""Qualified two-dimensional and batched contractions for NumPy/WASM."""
import math
import numpy as np


def product(a, b):
    """Use the direct contraction only where its cost and range are qualified.

    Native builds continue using their BLAS-backed matmul at the call site.
    Keep the original operation for vector/integer/complex products,
    small matrices, invalid shapes, and exceptional floating-point ranges.
    """
    if a.ndim != 2 or b.ndim != 2:
        return _batch_product(a, b)
    if (a.shape[1] != b.shape[0]
            or a.dtype != b.dtype or a.dtype not in (np.dtype('float32'), np.dtype('float64'))):
        return a @ b
    m, k = a.shape
    n = b.shape[1]
    if min(m, n) < 32 or k < 64 or m * k * n < 1048576:
        return a @ b
    # A scalar type keeps payload-bearing dtypes out of NumPy's global cache.
    limits = np.finfo(a.dtype.type)
    lower = np.sqrt(limits.tiny)
    upper = np.sqrt(limits.max / k) * 0.5
    for operand in (a, b):
        magnitude = np.abs(operand)
        # A NaN maximum fails the comparison; infinities and extreme finite
        # operands also keep NumPy matmul's original exceptional-value path.
        if not magnitude.max() <= upper:
            return a @ b
        if np.min(magnitude, where=magnitude != 0, initial=upper) < lower:
            return a @ b
        del magnitude
    # Contiguous reduction rows let WASM use the fast dot traversal. Packing
    # may add two temporary operand copies; none survive this contraction.
    # Passing order='C' to einsum disables its fast traversal on WASM, so
    # normalize only the result to retain matmul's C-order output contract.
    return np.ascontiguousarray(np.einsum(
        'ik,jk->ij', np.ascontiguousarray(a), np.ascontiguousarray(b.T),
        optimize=False))


def _unique_values(value, batch_only=False):
    stop = max(0, value.ndim - 2) if batch_only else value.ndim
    axes = [axis for axis in range(stop) if value.strides[axis] == 0 and value.shape[axis] > 1]
    if not axes:
        return value
    selection = [slice(None)] * value.ndim
    for axis in axes:
        selection[axis] = slice(0, 1)
    return value[tuple(selection)]


def _packed_rows(value):
    if value.flags.c_contiguous:
        return value
    unique = _unique_values(value, batch_only=True)
    packed = np.ascontiguousarray(unique)
    return np.broadcast_to(packed, value.shape) if unique.shape != value.shape else packed


def _batch_product(a, b):
    if (a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-2]
            or a.dtype != b.dtype or a.dtype not in (np.dtype('float32'), np.dtype('float64'))):
        return a @ b
    # Shape-only broadcasting has a lower rank limit than matmul on the
    # tested NumPy builds; retain the original operation beyond that limit.
    if max(a.ndim, b.ndim) > 34:
        return a @ b
    m, k = a.shape[-2:]
    n = b.shape[-1]
    if min(m, n) < 32 or k < 64 or m * k * n < 262144:
        return a @ b
    try:
        batch_shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    except ValueError:
        return a @ b
    if math.prod(batch_shape) * m * k * n < 1048576:
        return a @ b
    # A scalar type keeps payload-bearing dtypes out of NumPy's global cache.
    limits = np.finfo(a.dtype.type)
    lower = np.sqrt(limits.tiny)
    upper = np.sqrt(limits.max / k) * 0.5
    for operand in (a, b):
        magnitude = np.abs(_unique_values(operand))
        if not magnitude.max() <= upper:
            return a @ b
        if np.min(magnitude, where=magnitude != 0, initial=upper) < lower:
            return a @ b
        del magnitude
    # Pack actual inputs, keeping zero-stride batch repeats as views. Both
    # operands then present contiguous reduction rows to NumPy's dot kernel.
    left = _packed_rows(a)
    right = _packed_rows(np.swapaxes(b, -1, -2))
    return np.ascontiguousarray(np.einsum('...ik,...jk->...ij', left, right, optimize=False))
