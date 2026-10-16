"""Cholesky factors and real differentiation using triangular systems."""
import numpy as np


def _phi_values(data):
    value = _linalg._triangular_part(data, False, False)
    diagonal = np.arange(data.shape[-1])
    value[..., diagonal, diagonal] *= .5
    return value


def _phi(value):
    # This linear projection saves no operand values, including upper NaNs.
    return _autograd.result(_phi_values(value._data),
                            (value, _phi_values), saves_tensors=False)


def _factor_error(data):
    # NumPy does not expose the failing batch or leading minor. Resolve these
    # only on the exceptional path; successful batches use one LAPACK call.
    for index in np.ndindex(data.shape[:-2]):
        matrix = data[index]
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            lo, hi = 1, matrix.shape[-1]
            while lo < hi:
                mid = (lo + hi) // 2
                try:
                    np.linalg.cholesky(matrix[:mid, :mid])
                    lo = mid + 1
                except np.linalg.LinAlgError:
                    hi = mid
            batch = f' (Batch element {index})' if index else ''
            return RuntimeError(f'linalg.cholesky:{batch} input is not positive-definite '
                                f'(the leading minor of order {lo} is not positive-definite)')
    return RuntimeError('linalg.cholesky: input is not positive-definite')


def cholesky(A, *, upper=False, out=None):
    if not isinstance(A, _tensor.Tensor):
        raise TypeError('linalg.cholesky input must be a Tensor')
    if type(upper) is not bool:
        raise TypeError('upper must be bool')
    if out is not None:
        raise _errors.BrowserUnsupported('Cholesky output buffers require the server')
    data = A._data
    if data.ndim < 2 or data.shape[-1] != data.shape[-2]:
        raise RuntimeError('linalg.cholesky expects square matrices')
    if data.dtype.char not in 'fdFD':
        raise _errors.BrowserUnsupported('cholesky requires float32/64 or complex64/128 inputs')
    if isinstance(A, _autograd.ForwardTensor):
        value = cholesky(A._primal, upper=upper)
        lower = value.transpose(-1, -2) if upper else value
        # Native forward AD uses the untransposed direction for both choices
        # of triangle. Keep L differentiable through both systems and product.
        scaled = _linalg.solve_triangular(lower, A._tangent, upper=False)
        scaled = _linalg.solve_triangular(lower.transpose(-1, -2), scaled,
                                          upper=True, left=False)
        tangent = lower @ _phi(scaled)
        if upper:
            tangent = tangent.transpose(-1, -2)
        if not value.numel():
            # matching_tangent deliberately leaves empty views alone. LAPACK
            # factors still promise conventional column-major empty strides.
            tangent = _autograd.result(_linalg._solve_layout(tangent._data, True),
                                       (tangent, lambda g: g), saves_tensors=False)
        return _autograd.ForwardTensor(value, _layout.matching_tangent(tangent, value))
    # NumPy consumes the lower triangle. A conjugate transpose selects exactly
    # the requested upper triangle without inspecting the unused half.
    selected = (data.conj() if data.dtype.kind == 'c' else data).swapaxes(-1, -2) if upper else data
    try:
        lower = np.linalg.cholesky(selected)
    except np.linalg.LinAlgError as error:
        raise _factor_error(selected) from error
    values = lower.conj().swapaxes(-1, -2) if upper else lower
    values = _linalg._solve_layout(values, True)
    if not _state._grad_enabled or not A.requires_grad:
        return _tensor.Tensor(values, _metadata_owned=True)
    lower = values.swapaxes(-1, -2) if upper else values
    finite_diagonal = np.isfinite(np.diagonal(lower, axis1=-2, axis2=-1)).all()

    def solve(data, rhs, upper, left):
        if finite_diagonal:
            # The saved factor already has zeros in its unused triangle.
            # Small systems need no projection or block dispatch; LAPACK
            # amortizes them in one call, including small matrix batches.
            if data.shape[-1] <= 64:
                return _linalg._solve_values(data, rhs, left)
            return _linalg._triangular_values(data, rhs, upper, left, False)
        # Dense LU can contaminate otherwise finite blocks when a triangular
        # factor has an infinite diagonal. Substitution reads only its triangle.
        if not left:
            data, rhs, upper = data.swapaxes(-1, -2), rhs.swapaxes(-1, -2), not upper
        result = _linalg._triangular_rows(data, rhs, upper, False)
        return result if left else result.swapaxes(-1, -2)

    def backward(g):
        incoming = g.swapaxes(-1, -2) if upper else g
        n = lower.shape[-1]
        if n <= 2 and n and finite_diagonal and np.isfinite(incoming).all():
            if n == 1:
                return incoming / (2 * lower)
            # Scalar adjoints of L00=sqrt(A00), L10=A10/L00 and
            # L11=sqrt(A11-L10**2), with symmetric-domain off-diagonals.
            diagonal, below, last = lower[..., 0, 0], lower[..., 1, 0], lower[..., 1, 1]
            tail = incoming[..., 1, 1] / last
            cross = (incoming[..., 1, 0] - tail * below) / diagonal
            gradient = np.empty_like(g)
            gradient[..., 0, 0] = (incoming[..., 0, 0] - cross * below) / (2 * diagonal)
            gradient[..., 0, 1] = gradient[..., 1, 0] = .5 * cross
            gradient[..., 1, 1] = .5 * tail
            return gradient
        projected = _phi_values(lower.swapaxes(-1, -2) @ incoming)
        symmetric = projected + projected.swapaxes(-1, -2)
        left = solve(lower.swapaxes(-1, -2), symmetric, True, True)
        gradient = solve(lower, left, False, False)
        return gradient * .5

    output = _autograd.result(values, (A, backward), versioned=())
    if output.requires_grad:
        # Backward saves the factor, not A. Native permits changing A after
        # forward, but modifying the factor invalidates its saved values.
        counter = output._storage
        output._grad_node.versions = ((counter, counter.version),)
    return output


from . import _autograd, _errors, _layout, _linalg, _state, _tensor
