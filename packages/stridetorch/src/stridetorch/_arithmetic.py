"""Operand promotion and Number conversion shared by core arithmetic."""
import math

import numpy as np

from . import _autograd, _defaults, _errors, _ops, _promotion, _state, _tensor

_FLOAT = np.dtype('float32')
_DOUBLE = np.dtype('float64')


def scalar_data(number, dtype):
    # Arithmetic wraps integer operands, unlike checked where/alpha storage.
    if dtype.kind in 'iu':
        bits = dtype.itemsize * 8
        number %= 2**bits
        if dtype.kind == 'i' and number >= 2**(bits-1): number -= 2**bits
    if dtype.kind in 'biu' or dtype == _DOUBLE:
        return np.asarray(number, dtype=dtype)
    if dtype == _FLOAT and math.isfinite(number) and (not number or 1.1754943508222875e-38 <= abs(number) <= 3.4028234663852886e38):
        return np.asarray(number, dtype=dtype)
    with np.errstate(all='ignore'):
        return np.asarray(number, dtype=dtype)


def operands(owner, other, *, floating=False, subtract=False):
    if isinstance(other, _tensor.Tensor):
        if subtract and (owner.dtype.kind == 'b' or other.dtype.kind == 'b'):
            raise _errors.BrowserUnsupported('Subtraction with boolean operands is unsupported')
        dtype = owner.dtype
        if dtype == other.dtype:
            if not floating or dtype.kind not in 'biu':
                return other, owner._data, other._data, dtype
        else: dtype = _promotion.operand_dtype(owner, other)
    elif _promotion.is_scalar(other):
        number, kind = _promotion.scalar_info(other)
        if subtract and (owner.dtype.kind == 'b' or kind.kind == 'b'):
            raise _errors.BrowserUnsupported('Subtraction with boolean operands is unsupported')
        dtype = _promotion._combine(owner.dtype, kind, wrapped=True)
        if floating and dtype.kind in 'biu': dtype = _defaults._float_dtype
        # Numbers are immutable constants, with no history or externally
        # mutable Tensor storage. Keep their array without allocating a Tensor.
        return None, _promotion.array(owner, dtype), scalar_data(number, dtype), dtype
    else:
        # Preserve the NumPy-oriented Tensor class's existing array convenience.
        other = _ops.wrap(other)
        dtype = _promotion.operand_dtype(owner, other)
    if subtract and (owner.dtype.kind == 'b' or other.dtype.kind == 'b'):
        raise _errors.BrowserUnsupported('Subtraction with boolean operands is unsupported')
    if floating and dtype.kind in 'biu': dtype = _defaults._float_dtype
    a = _promotion.array(owner, dtype)
    b = _promotion.array(other, dtype)
    return other, a, b, dtype


def forward_primal(owner, other):
    if isinstance(other, _autograd.ForwardTensor): return other._primal
    if not isinstance(other, _tensor.Tensor) and _promotion.is_scalar(other):
        number, kind = _promotion.scalar_info(other)
        return _tensor.Tensor(scalar_data(number, _promotion._combine(owner.dtype, kind, wrapped=True)), _metadata_owned=True)
    return _ops.wrap(other)


def forward_parts(owner, other):
    if isinstance(other, _autograd.ForwardTensor): return other._primal, other._tangent
    primal = forward_primal(owner, other)
    return primal, _tensor.Tensor(np.zeros_like(primal._data))


def _owner(input, other):
    if not isinstance(other, _tensor.Tensor) and not _promotion.is_scalar(other):
        raise TypeError('Expected a Tensor or Number operand')
    if isinstance(input, _tensor.Tensor): return input, other, False
    if not _promotion.is_scalar(input): raise TypeError('Expected a Tensor or Number input')
    if isinstance(other, _tensor.Tensor): return other, input, True
    if not _promotion.is_scalar(other): raise TypeError('Expected a Tensor or Number operand')
    number, kind = _promotion.scalar_info(input)
    if kind.kind=='c':kind=_defaults.complex_dtype()
    return _tensor.Tensor(scalar_data(number, kind), _metadata_owned=True), other, False


def _alpha(number, kind, dtype):
    if kind.kind == 'b' and dtype.kind != 'b':
        raise RuntimeError('Boolean alpha is only supported for boolean results')
    if kind.kind in 'fc' and dtype.kind in 'biu':
        raise RuntimeError('Integral results require an integral alpha')
    if kind.kind == 'c' and dtype.kind != 'c':
        raise RuntimeError('Real results cannot use a complex alpha')
    if dtype == np.dtype('float16') and np.isfinite(number) and abs(number) > 65504:
        raise RuntimeError('Alpha cannot be converted to float16 without overflow')
    return _promotion.scalar_array(number, dtype, parsed=True)


def _add(input, other, alpha, *, subtract=False):
    owner, argument, reverse = _owner(input, other)
    if not reverse and type(alpha) is int and alpha == 1:
        return owner - argument if subtract else owner + argument
    if isinstance(owner, _autograd.ForwardTensor) or isinstance(argument, _autograd.ForwardTensor):
        alpha_number, alpha_kind = _promotion.scalar_info(alpha)
        alpha = alpha_number
        owner_dual = isinstance(owner, _autograd.ForwardTensor)
        argument_dual = isinstance(argument, _autograd.ForwardTensor)
        op = owner._primal if owner_dual else owner
        raw_argument = argument._primal if argument_dual else argument
        first, second = (raw_argument, op) if reverse else (op, raw_argument)
        primal = _add(first, second, alpha, subtract=subtract)
        if owner_dual and argument_dual:
            tf, ts = (argument._tangent, owner._tangent) if reverse else (owner._tangent, argument._tangent)
            tangent = _add(tf, ts, alpha, subtract=subtract)
        else:
            tangent = _autograd.match_tangent(owner._tangent if owner_dual else argument._tangent, primal)
            if reverse or argument_dual:
                number = -alpha_number if subtract else alpha_number
                scale = _tensor.Tensor(_alpha(number, alpha_kind, primal.dtype), _metadata_owned=True)
                tangent = tangent * scale
        return _autograd.ForwardTensor(primal, tangent)
    argument, a, b, dtype = operands(owner, argument, subtract=subtract)
    number, kind = _promotion.scalar_info(alpha)
    if subtract:
        # Native sub checks alpha's original category, then negates before cast.
        _alpha(number, kind, dtype) if dtype.kind not in 'iu' else None
        if kind.kind == 'b' and dtype.kind != 'b': raise RuntimeError('Boolean alpha requires boolean results')
        if kind.kind in 'fc' and dtype.kind in 'iu': raise RuntimeError('Integral results require an integral alpha')
        number = -number
    scale = _alpha(number, kind, dtype)
    if dtype.kind == 'u' and dtype.itemsize > 1:
        raise _errors.BrowserUnsupported('This unsigned arithmetic kernel requires the server')
    first, second = (b, a) if reverse else (a, b)
    compute_dtype = np.dtype('float32') if dtype == np.dtype('float16') else dtype
    with np.errstate(all='ignore'):
        try:
            data = np.add(first.astype(compute_dtype, copy=False),
                          np.multiply(second.astype(compute_dtype, copy=False), scale.astype(compute_dtype)))
            data = data.astype(dtype, copy=False)
        except ValueError as error: raise RuntimeError('Arithmetic operands have incompatible shapes') from error
    if not _state._grad_enabled or not owner.requires_grad and (argument is None or not argument.requires_grad):
        return _autograd.result(data, saves_tensors=False)
    owner_shape, argument_shape = a.shape, b.shape
    def scaled(g):
        with np.errstate(all='ignore'): return g * scale
    edges = [(owner, lambda g: _ops.unbroadcast(scaled(g) if reverse else g, owner_shape))]
    if argument is not None:
        edges.append((argument, lambda g: _ops.unbroadcast(g if reverse else scaled(g), argument_shape)))
    return _autograd.result(data, *edges, saves_tensors=False)


def add(input, other, *, alpha=1, out=None):
    if out is not None: raise _errors.BrowserUnsupported('Arithmetic output buffers require the server')
    return _add(input, other, alpha)


def sub(input, other, *, alpha=1, out=None):
    if out is not None: raise _errors.BrowserUnsupported('Arithmetic output buffers require the server')
    return _add(input, other, alpha, subtract=True)


def mul(input, other, *, out=None):
    if out is not None: raise _errors.BrowserUnsupported('Arithmetic output buffers require the server')
    owner, argument, _ = _owner(input, other)
    return owner * argument


def div(input, other, *, rounding_mode=None, out=None):
    if out is not None: raise _errors.BrowserUnsupported('Arithmetic output buffers require the server')
    if rounding_mode is not None and not isinstance(rounding_mode, str): raise TypeError('rounding_mode must be a string or None')
    if rounding_mode not in (None, 'floor', 'trunc'): raise RuntimeError('Unknown division rounding mode')
    owner, argument, reverse = _owner(input, other)
    if rounding_mode is None:
        if reverse and not isinstance(owner, _autograd.ForwardTensor):
            # Functional div uses TensorIterator promotion. Python's reflected
            # Tensor division instead promotes integer receivers via reciprocal.
            return owner._binary(argument, np.divide, lambda g,a,b:-g*((b/a)/a),
                                 lambda g,a,b:g/a, floating=True, reverse=True)
        if reverse:
            primal = owner._primal._binary(argument, np.divide,
                lambda g,a,b:-g*((b/a)/a), lambda g,a,b:g/a, floating=True, reverse=True)
            return _autograd.ForwardTensor(primal, (-primal * owner._tangent) / owner._primal)
        return owner / argument
    if isinstance(owner, _autograd.ForwardTensor) or isinstance(argument, _autograd.ForwardTensor):
        op, ot = _autograd.forward_parts(owner)
        ap, at = forward_parts(op, argument)
        first, second = (ap, op) if reverse else (op, ap)
        primal = div(first, second, rounding_mode=rounding_mode)
        return _autograd.ForwardTensor(primal, _tensor.Tensor(np.zeros_like(primal._data)))
    argument, a, b, dtype = operands(owner, argument)
    if dtype.kind in 'bc' or dtype.kind == 'u' and dtype.itemsize > 1:
        raise _errors.BrowserUnsupported('This rounded division dtype requires the server')
    first, second = (b, a) if reverse else (a, b)
    if dtype.kind in 'iu':
        if np.any(second == 0): raise RuntimeError('Integer division by zero')
        if dtype.kind == 'i' and np.any(first == np.iinfo(dtype).min) and np.any(second == -1):
            raise _errors.BrowserUnsupported('Overflowing integer division requires the server')
    compute_dtype = np.dtype('float32') if dtype == np.dtype('float16') else dtype
    with np.errstate(all='ignore'):
        try:
            x, y = first.astype(compute_dtype, copy=False), second.astype(compute_dtype, copy=False)
            if dtype.kind in 'iu':
                data = np.floor_divide(x, y)
                if rounding_mode == 'trunc' and dtype.kind == 'i':
                    correction = ((x < 0) != (y < 0)) & (np.remainder(x, y) != 0)
                    data = np.add(data, correction.astype(dtype))
            else: data = np.floor_divide(x, y) if rounding_mode == 'floor' else np.trunc(np.divide(x, y))
            data = np.asarray(data, dtype=dtype)
        except ValueError as error: raise RuntimeError('Arithmetic operands have incompatible shapes') from error
    if not _state._grad_enabled or not owner.requires_grad and (argument is None or not argument.requires_grad):
        return _autograd.result(data, saves_tensors=False)
    owner_shape, argument_shape = a.shape, b.shape
    edges = [(owner, lambda g: np.zeros(owner_shape, dtype=g.dtype))]
    if argument is not None: edges.append((argument, lambda g: np.zeros(argument_shape, dtype=g.dtype)))
    return _autograd.result(data, *edges, saves_tensors=False)


def true_divide(input, other, *, out=None):
    return div(input, other, out=out)


def reciprocal(input, *, out=None):
    if not isinstance(input, _tensor.Tensor): raise TypeError('reciprocal input must be a Tensor')
    if out is not None: raise _errors.BrowserUnsupported('Reciprocal output buffers require the server')
    return input.reciprocal()


def reciprocal_values(source):
    """Fresh private output; caller owns the numerical error scope."""
    if source.dtype.kind in 'biu': source = source.astype(_defaults._float_dtype)
    if source.dtype.kind != 'c': return np.asarray(np.reciprocal(source))
    data = np.asarray(np.divide(np.ones((), dtype=source.dtype), source))
    zero = (source.real == 0) & (source.imag == 0)
    if np.any(zero):
        data.real[zero] = np.copysign(np.inf, source.real[zero])
        data.imag[zero] = np.nan
    return data


def reflected_number_divide(owner, other):
    # Only exact builtin Numbers use this route: extracting an opaque Number
    # must retain the ordinary reciprocal-before-user-protocol call order.
    number, kind = _promotion.scalar_info(other)
    with np.errstate(all='ignore'):
        inverse = reciprocal_values(owner._data)
        dtype = _promotion._combine(inverse.dtype, kind)
        scalar = scalar_data(number, dtype)
        factor = inverse.astype(dtype, copy=False)
        active = owner.requires_grad and _state._grad_enabled
        if not active:
            np.multiply(factor, scalar, out=factor)
            return _autograd.result(factor, saves_tensors=False)
        data = np.multiply(factor, scalar)
    # Native backward first multiplies the seed by the Number, then applies
    # reciprocal's saved-output derivative. Neither public input nor output
    # storage is saved; the private inverse array is released with this VJP.
    def backward(g): return (g * scalar) * -(inverse * inverse)
    return _autograd.result(data, (owner, backward), versioned=())


def scalar_power_values(source, exponent):
    """Keep public storage arithmetic; use scalar opmath in Tensor pullbacks."""
    if source.dtype == np.dtype('float16') and isinstance(exponent, np.floating):
        # A NumPy scalar must not promote a half Tensor's result storage.
        exponent = float(exponent)
    if source.dtype not in (np.dtype('float32'), np.dtype('float64')) or not isinstance(exponent, (int, float, np.integer, np.floating, bool, np.bool_)):
        return source ** exponent
    if exponent in (.5, -.5, -1, 2, 3, -2):
        output = np.empty_like(source)
        if exponent == .5:
            return np.sqrt(source, out=output)
        if exponent == -.5:
            np.sqrt(source, out=output)
            return np.reciprocal(output, out=output)
        if exponent == -1:
            return np.reciprocal(source, out=output)
        if exponent == 2:
            return np.multiply(source, source, out=output)
        if exponent == 3:
            np.multiply(source, source, out=output)
            return np.multiply(output, source, out=output)
        np.multiply(source, source, out=output)
        return np.reciprocal(output, out=output)
    context = _state._tensor_reverse_context
    if source.dtype == np.dtype('float32') and context is not None and context.reverse_active:
        output = np.empty_like(source)
        np.power(source, np.float64(exponent), out=output, dtype=np.float64)
        return output
    return source ** exponent
