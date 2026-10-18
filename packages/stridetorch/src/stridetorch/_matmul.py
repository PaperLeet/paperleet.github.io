"""Bounded two-dimensional contraction for NumPy's WASM runtime."""
import numpy as np


def product(a, b):
    """Use the direct contraction only where its cost and range are qualified.

    Native builds continue using their BLAS-backed matmul at the call site.
    Keep the original operation for vector/batched/integer/complex products,
    small matrices, invalid shapes, and exceptional floating-point ranges.
    """
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]
            or a.dtype != b.dtype or a.dtype not in (np.dtype('float32'), np.dtype('float64'))):
        return a @ b
    m, k = a.shape
    n = b.shape[1]
    if min(m, n) < 32 or k < 64 or m * k * n < 1048576:
        return a @ b
    limits = np.finfo(a.dtype)
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
    # Let einsum choose its fast traversal. Passing order='C' disables that
    # traversal on WASM even for C-order results. Normalize only the output
    # so transposed/strided inputs still preserve matmul's C-order contract.
    return np.ascontiguousarray(np.einsum('ik,kj->ij', a, b, optimize=False))
