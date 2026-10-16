"""Cholesky factors and real differentiation using triangular systems."""
import sys
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


def _factor_values(data):
    values = np.linalg.cholesky(data)
    # Some LAPACK/NumPy builds return NaN factors without raising, including
    # finite non-positive-definite inputs under WASM's floating-point flags.
    # Inspect only the result diagonal: unused input entries remain ignored.
    if not (np.diagonal(values, axis1=-2, axis2=-1).real > 0).all():
        raise np.linalg.LinAlgError('Cholesky factor has a non-positive or NaN diagonal')
    return values


def _factor_error(data):
    # NumPy does not expose the failing batch or leading minor. Resolve these
    # only on the exceptional path; successful batches use one LAPACK call.
    for index in np.ndindex(data.shape[:-2]):
        matrix = data[index]
        try:
            _factor_values(matrix)
        except np.linalg.LinAlgError:
            lo, hi = 1, matrix.shape[-1]
            while lo < hi:
                mid = (lo + hi) // 2
                try:
                    _factor_values(matrix[:mid, :mid])
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
        lower = _factor_values(selected)
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


def _solve_values(factor, rhs, upper, *, save_intermediate=False):
    transpose = (factor.conj() if factor.dtype.kind == 'c' else factor).swapaxes(-1, -2)
    first, second = (transpose, factor) if upper else (factor, transpose)
    intermediate = _linalg._triangular_values(first, rhs, False, True, False)
    value = _linalg._triangular_values(second, intermediate, True, True, False)
    return (value, intermediate) if save_intermediate else value


def _inverse_values(factor, upper):
    n = factor.shape[-1]
    diagonal = np.diagonal(factor, axis1=-2, axis2=-1)
    zero = diagonal == 0
    if zero.any():
        index = tuple(np.argwhere(zero)[0])
        batch = (f' (Batch element {np.ravel_multi_index(index[:-1], factor.shape[:-2])}):'
                 if factor.ndim > 2 else '')
        raise RuntimeError(f'cholesky_inverse:{batch} The diagonal element {index[-1] + 1} '
                           'is zero, the inversion could not be completed because the input matrix is singular.')
    if n and n <= 2:
        with np.errstate(all='ignore'):
            first = 1 / factor[..., 0, 0]
            if n == 1:
                values = (first.conj() * first)[..., None, None]
            else:
                last = 1 / factor[..., 1, 1]
                cross = -(factor[..., 0, 1] if upper else factor[..., 1, 0]) * first * last
                corner = first.conj() * first + cross.conj() * cross
                off = cross * last.conj() if upper else cross.conj() * last
                values = np.empty_like(factor)
                values[..., 0, 0] = corner
                values[..., 0, 1], values[..., 1, 0] = off, off.conj()
                values[..., 1, 1] = last.conj() * last
        return _linalg._solve_layout(values, True)
    if n <= 64:
        # NumPy's inverse entry avoids constructing/broadcasting an identity
        # and the general blocked-solve dispatch for these small factors.
        selected = _linalg._triangular_part(factor, upper, False)
        try:
            inverse = np.linalg.inv(selected)
        except np.linalg.LinAlgError:
            identity = np.broadcast_to(np.eye(n, dtype=factor.dtype), factor.shape)
            inverse = _linalg._triangular_rows(selected, identity, upper, False)
    else:
        identity = np.broadcast_to(np.eye(n, dtype=factor.dtype), factor.shape)
        inverse = None
    if sys.platform == 'emscripten' and n > 64:
        # On measured WASM builds a dense Gram product loses to the blocked
        # second triangular solve. CPU BLAS benefits from the Gram route.
        values = _solve_values(factor, identity, upper)
        if np.isfinite(values).all():
            transpose = values.swapaxes(-1, -2)
            if factor.dtype.kind == 'c':transpose = transpose.conj()
            mask = np.tri(n, k=-1, dtype=bool)
            np.copyto(values, transpose, where=mask if upper else mask.T)
            return _linalg._solve_layout(values, True)
    if inverse is None:inverse = _linalg._triangular_values(factor, identity, upper, True, False)
    finite_inverse = np.isfinite(inverse).all()
    if not finite_inverse:
        # Solve only each column's structural range. A NaN in one diagonal
        # must not enter an unrelated column through a mathematical zero.
        inverse = np.zeros_like(factor)
        for column in range(n):
            section = slice(0, column + 1) if upper else slice(column, n)
            block = factor[..., section, section]
            rhs = np.zeros(block.shape[:-1] + (1,), dtype=factor.dtype)
            rhs[..., -1 if upper else 0, 0] = 1
            inverse[..., section, column] = _linalg._triangular_rows(block, rhs, upper, False)[..., 0]
    adjoint = (inverse.conj() if factor.dtype.kind == 'c' else inverse).swapaxes(-1, -2)
    with np.errstate(all='ignore'):
        values = inverse @ adjoint if upper else adjoint @ inverse
        if not finite_inverse:
            # The exceptional triangular product omits structural zero terms,
            # preserving finite trailing blocks instead of a dense NaN cloud.
            for row in range(n):
                for column in range(row + 1):
                    if upper:
                        tail = inverse[..., column, row:] * inverse[..., row, row:].conj()
                        value = tail.sum(-1)
                        i, j = column, row
                    else:
                        tail = inverse[..., row:, row].conj() * inverse[..., row:, column]
                        value = tail.sum(-1)
                        i, j = row, column
                    values[..., i, j] = value
        transpose = values.swapaxes(-1, -2)
        if factor.dtype.kind == 'c':transpose = transpose.conj()
        mask = np.tri(n, k=-1, dtype=bool)
        np.copyto(values, transpose, where=mask if upper else mask.T)
    return _linalg._solve_layout(values, True)


def _inverse_tangent(value, factor, direction, upper):
    x, f, df = value._data, factor._data, direction._data
    transpose = lambda a: a.swapaxes(-1, -2)
    cf = f if upper else transpose(f)
    dcf = transpose(df) if upper else df
    xd = x @ dcf
    middle = xd @ x
    partial = -(middle @ cf)
    values = partial + transpose(partial)
    if not values.size:values = _layout.contiguous_empty(values)
    if not _state._grad_enabled or not any(t.requires_grad for t in (value, factor, direction)):
        return _tensor.Tensor(values, _metadata_owned=True)
    saved_direction = dcf if value.requires_grad else None
    saved_xd = xd if value.requires_grad else None
    saved_middle = middle if factor.requires_grad else None

    def backward(g, needs):
        h = -(g + transpose(g))
        right = h @ transpose(cf) if needs[0] or needs[2] else None
        dx = (right @ transpose(x) @ transpose(saved_direction)
              + transpose(saved_xd) @ right) if needs[0] else None
        if needs[1]:
            dc = transpose(saved_middle) @ h
            dfactor = dc if upper else transpose(dc)
        else:dfactor = None
        if needs[2]:
            dd = transpose(x) @ right @ transpose(x)
            ddirection = transpose(dd) if upper else dd
        else:ddirection = None
        return dx, dfactor, ddirection

    guarded = [value, factor]
    if value.requires_grad:guarded.append(direction)
    return _autograd.joint_result(values, (value, factor, direction), backward, versioned=guarded)


def cholesky_inverse(input, upper=False, *, out=None):
    if not isinstance(input, _tensor.Tensor):
        raise TypeError('cholesky_inverse input must be a Tensor')
    if type(upper) is not bool:
        raise TypeError('upper must be bool')
    if out is not None:
        raise _errors.BrowserUnsupported('Cholesky inverse output buffers require the server')
    factor = input._data
    if factor.ndim < 2 or factor.shape[-1] != factor.shape[-2]:
        raise RuntimeError('cholesky_inverse expects square factors')
    if factor.dtype.char not in 'fdFD':
        raise _errors.BrowserUnsupported('cholesky_inverse requires float32/64 or complex64/128 inputs')
    if factor.dtype.kind == 'c' and np.any(np.diagonal(factor, axis1=-2, axis2=-1).imag != 0):
        raise _errors.BrowserUnsupported('Complex Cholesky factors require real diagonals')
    if isinstance(input, _autograd.ForwardTensor):
        value = cholesky_inverse(input._primal, upper)
        # Preserve native's ordered JVP expression and its full-factor reverse
        # edges; replacing it by an inverse-system derivative changes behavior.
        tangent = _inverse_tangent(value, input._primal, input._tangent, upper)
        return _autograd.ForwardTensor(value, _layout.matching_tangent(tangent, value))
    values = _inverse_values(factor, upper)
    if not _state._grad_enabled or not input.requires_grad:
        return _tensor.Tensor(values, _metadata_owned=True)

    def backward(g):
        common = values @ ((g + g.swapaxes(-1, -2)) @ values)
        return -(factor @ common if upper else common @ factor)

    return _autograd.result(values, (input, backward))


def _narrow_tangent(value, factor, rhs_direction, factor_direction, upper, intermediate=None):
    """Fuse the real JVP without projecting any full-factor reverse edge."""
    x, f, df = value._data, factor._data, factor_direction._data
    transpose = lambda a: a.swapaxes(-1, -2)

    def factor_product(rhs):
        return (f if upper else transpose(f)) @ rhs

    def direction_product(rhs):
        return (df if upper else transpose(df)) @ rhs

    dfx = direction_product(x)
    reuse = intermediate is not None
    ft = None
    if reuse:
        fx = intermediate
        first_term = (transpose(df) if upper else df) @ fx
        reuse = (np.isfinite(first_term).all() and np.isfinite(dfx).all()
                 and (rhs_direction is None or np.isfinite(rhs_direction._data).all()))
        if reuse:
            first_rhs = -first_term if rhs_direction is None else rhs_direction._data - first_term
            first = _linalg._triangular_values(transpose(f) if upper else f, first_rhs, False, True, False)
            ft = first - dfx
            values = _linalg._triangular_values(f if upper else transpose(f), ft, True, True, False)
            reuse = np.isfinite(ft).all() and np.isfinite(values).all()
    if not reuse:
        fx = factor_product(x)
        residual = -(transpose(df) @ fx + transpose(f) @ dfx if upper
                     else df @ fx + f @ dfx)
        if rhs_direction is not None:
            residual = rhs_direction._data + residual
        values = _solve_values(f, residual, upper)
        ft = None
    values = _linalg._solve_layout(values, True)
    db = rhs_direction if rhs_direction is not None else _tensor.Tensor(np.zeros_like(x))
    inputs = (value, factor, db, factor_direction)
    if not _state._grad_enabled or not any(t.requires_grad for t in inputs):
        return _tensor.Tensor(values, _metadata_owned=True)
    shapes = tuple(t.shape for t in inputs)
    saved_x = x if factor.requires_grad or factor_direction.requires_grad else None
    saved_output = values if factor.requires_grad else None
    saved_fx = fx if factor_direction.requires_grad else None
    saved_dfx = dfx if factor.requires_grad else None
    saved_ft = ft if factor.requires_grad else None
    if value.requires_grad and not factor.requires_grad:
        # Native saves the already-formed coefficient direction. Its source
        # direction may change when no factor adjoint needs that source. Keep
        # a source snapshot without forming the coefficient product.
        df = df.copy()
    elif not value.requires_grad and not factor.requires_grad:
        df = None

    def factor_adjoint(q, solution, product_q, product_solution):
        # The forward and adjoint n-by-k products are shared across all four
        # inputs. Recomputing them per input penalizes wider narrow systems.
        if upper:
            return -(product_q @ transpose(solution) + product_solution @ transpose(q))
        return -(q @ transpose(product_solution) + solution @ transpose(product_q))

    def backward(g, needs):
        product_needed = needs[0] or needs[1] or needs[3]
        if reuse and product_needed:
            q, fq = _solve_values(f, g, upper, save_intermediate=True)
            if not (np.isfinite(q).all() and np.isfinite(fq).all()):
                fq = factor_product(q)
        else:
            q = _solve_values(f, g, upper)
            fq = factor_product(q) if product_needed else None
        dfq = direction_product(q) if needs[0] or needs[1] else None
        dx = (-(transpose(df) @ fq + transpose(f) @ dfq if upper
                else df @ fq + f @ dfq)) if needs[0] else None
        if needs[1]:
            gf = factor_adjoint(q, saved_output, fq,
                                saved_ft if saved_ft is not None else factor_product(saved_output))
            gf += factor_adjoint(q, saved_x, dfq, saved_dfx)
        else:
            gf = None
        gd = factor_adjoint(q, saved_x, fq, saved_fx) if needs[3] else None
        gradients = (dx, gf, q if needs[2] else None, gd)
        return tuple(None if gradient is None else gradient if gradient.shape == shape
                     else _ops.unbroadcast(gradient, shape)
                     for gradient, shape in zip(gradients, shapes))

    # RHS directions enter linearly and their values are not saved. The other
    # selective guards match the operands needed by the analytical pullback.
    guarded = [factor]
    if value.requires_grad or factor.requires_grad or factor_direction.requires_grad:
        guarded.append(value)
    if factor.requires_grad:
        guarded.append(factor_direction)
    output = _autograd.joint_result(values, inputs, backward, versioned=guarded)
    counter = output._storage
    output._grad_node.versions += ((counter, counter.version),)
    return output


def cholesky_solve(input, input2, upper=False, *, out=None):
    if not isinstance(input, _tensor.Tensor) or not isinstance(input2, _tensor.Tensor):
        raise TypeError('cholesky_solve inputs must be Tensors')
    if type(upper) is not bool:
        raise TypeError('upper must be bool')
    if out is not None:
        raise _errors.BrowserUnsupported('Cholesky solve output buffers require the server')
    b, factor = input._data, input2._data
    if b.dtype != factor.dtype:
        raise RuntimeError('cholesky_solve expects matching dtypes')
    if factor.dtype.char not in 'fdFD':
        raise _errors.BrowserUnsupported('cholesky_solve requires float32/64 or complex64/128 inputs')
    if factor.ndim < 2 or factor.shape[-1] != factor.shape[-2]:
        raise RuntimeError('cholesky_solve expects square factors')
    if b.ndim < 2 or b.shape[-2] != factor.shape[-1]:
        raise RuntimeError('cholesky_solve requires matching matrix right-hand sides')
    b_shape, f_shape = b.shape, factor.shape
    try:
        batch = b_shape[:-2] if b_shape[:-2] == f_shape[:-2] else np.broadcast_shapes(b_shape[:-2], f_shape[:-2])
    except ValueError:
        raise RuntimeError('cholesky_solve batch dimensions cannot broadcast') from None
    if isinstance(input, _autograd.ForwardTensor) or isinstance(input2, _autograd.ForwardTensor):
        primal_b = input._primal if isinstance(input, _autograd.ForwardTensor) else input
        primal_f = input2._primal if isinstance(input2, _autograd.ForwardTensor) else input2
        direction = input2._tangent if isinstance(input2, _autograd.ForwardTensor) else _tensor.Tensor(np.zeros_like(factor))
        # Native differentiation extends the full factor, including its unused
        # triangle. Only the primal factor solves select a triangle.
        narrow_shape = f_shape[-1] >= 64 and b_shape[-1] * 4 <= f_shape[-1]
        primal = _solve_primal(primal_b, primal_f, upper, batch, capture=narrow_shape)
        if narrow_shape:
            value, intermediate, finite_values = primal
        else:
            value, intermediate, finite_values = primal, None, False
        narrow_direction = finite_values and np.isfinite(direction._data).all()
        if narrow_direction:
            tangent = _narrow_tangent(value, primal_f,
                input._tangent if isinstance(input, _autograd.ForwardTensor) else None,
                direction, upper, intermediate)
            return _autograd.ForwardTensor(value, _layout.matching_tangent(tangent, value))
        product = direction.transpose(-1, -2) @ primal_f if upper else direction @ primal_f.transpose(-1, -2)
        coefficient_direction = product + product.transpose(-1, -2)
        contribution = coefficient_direction @ value
        residual = -contribution
        if isinstance(input, _autograd.ForwardTensor):
            residual = input._tangent + residual
        tangent = cholesky_solve(residual, primal_f, upper)
        return _autograd.ForwardTensor(value, _layout.matching_tangent(tangent, value))
    return _solve_primal(input, input2, upper, batch)


def _solve_primal(input, input2, upper, batch, *, capture=False):
    # The public entry has validated both primal tensors, including the
    # ForwardTensor case. Reuse that validation instead of recursing through it.
    b, factor = input._data, input2._data
    b_shape, f_shape = b.shape, factor.shape
    matrix_f = factor if f_shape[:-2] == batch else np.broadcast_to(factor, batch + f_shape[-2:])
    matrix_b = b if b_shape[:-2] == batch else np.broadcast_to(b, batch + b_shape[-2:])
    needs_b, needs_f = input.requires_grad, input2.requires_grad
    finite_factor = (capture or _state._grad_enabled and needs_f) and f_shape[-1] >= 64 and np.isfinite(factor).all()
    # Gradient factors already need this classification for their VJP. A new
    # scan for frozen factors pays off on WASM, but loses to native CPU BLAS;
    # very small WASM RHSs also keep the original unclassified path.
    reuse = _state._grad_enabled and needs_f
    if capture and not reuse and sys.platform == 'emscripten':
        reuse = f_shape[-1] > 64 or b_shape[-1] > 4
    triangular_factor = finite_factor and reuse and not np.any(_linalg._triangular_part(factor, not upper, True))
    if triangular_factor:
        values, forward_intermediate = _solve_values(matrix_f, matrix_b, upper, save_intermediate=True)
    else:
        values, forward_intermediate = _solve_values(matrix_f, matrix_b, upper), None
    values = _linalg._solve_layout(values, True)
    finite_values = finite_factor and np.isfinite(values).all()
    triangular_factor = triangular_factor and finite_values and np.isfinite(forward_intermediate).all()
    if not triangular_factor:forward_intermediate = None
    if not _state._grad_enabled or not (needs_b or needs_f):
        output = _tensor.Tensor(values, _metadata_owned=True)
        return (output, forward_intermediate, finite_values) if capture else output
    x_transpose = values.swapaxes(-1, -2) if needs_f else None
    narrow = finite_values and values.shape[-1] * 4 <= f_shape[-1]

    def right_gradient(g):
        return _solve_values(matrix_f, g, upper)

    def reduce_b(db):
        return db if db.shape == b_shape else _ops.unbroadcast(db, b_shape)

    def factor_gradient(db, adjoint_intermediate=None):
        if triangular_factor:
            # Actual triangular factors permit two narrow products, reusing
            # one forward/adjoint intermediate. Full unmasked gradients remain.
            if upper:
                df = -(adjoint_intermediate @ x_transpose + forward_intermediate @ db.swapaxes(-1, -2))
            else:
                df = -(db @ forward_intermediate.swapaxes(-1, -2) + values @ adjoint_intermediate.swapaxes(-1, -2))
            return df if df.shape == f_shape else _ops.unbroadcast(df, f_shape)
        if narrow:
            # Associate through the narrow RHS rather than materializing and
            # multiplying a dense n-by-n common term: O(n*n*k), not O(n**3).
            # Nonfinite factors retain the native evaluation order below.
            if upper:
                df = -((matrix_f @ db) @ x_transpose + (matrix_f @ values) @ db.swapaxes(-1, -2))
            else:
                df = -(db @ (x_transpose @ matrix_f) + values @ (db.swapaxes(-1, -2) @ matrix_f))
            return df if df.shape == f_shape else _ops.unbroadcast(df, f_shape)
        common = db @ x_transpose
        common = common + common.swapaxes(-1, -2)
        df = -(matrix_f @ common if upper else common @ matrix_f)
        return df if df.shape == f_shape else _ops.unbroadcast(df, f_shape)

    def factor_pullback(g):
        if triangular_factor:
            db, intermediate = _solve_values(matrix_f, g, upper, save_intermediate=True)
            return db, factor_gradient(db, intermediate)
        db = right_gradient(g)
        return db, factor_gradient(db)

    if needs_b and needs_f:
        def backward(g, needs):
            db, df = factor_pullback(g)
            return reduce_b(db), df
        output = _autograd.joint_result(values, (input, input2), backward)
    elif needs_b:
        output = _autograd.result(values, (input, lambda g: reduce_b(right_gradient(g))), saved=(input2,))
    else:
        output = _autograd.result(values, (input2, lambda g: factor_pullback(g)[1]), saved=(input,))
    return (output, forward_intermediate, finite_values) if capture else output


from . import _autograd, _errors, _layout, _linalg, _ops, _state, _tensor
