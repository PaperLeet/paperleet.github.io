"""Checked reduction axes and NaN-aware real reductions."""
import numpy as np
from . import _errors, _layout, _state, _tensor


MISSING = _layout.UNSPECIFIED_DIM


_SCAN_ACCUMULATORS = {np.dtype('float16'):np.dtype('float32'),
                      np.dtype('float32'):np.dtype('float64'),
                      np.dtype('complex64'):np.dtype('complex128')}
_SCAN_WORK_BYTES = 256 * 1024
_SCAN_INT64 = np.dtype('int64')


def _scan_array(data, axis, dtype, *, reverse=False):
    if not data.ndim:
        return np.array(data, dtype=dtype, copy=True)
    value = np.empty(data.shape, dtype=dtype)
    if not value.size:
        return _layout.contiguous_empty(value)
    incoming = np.flip(data, axis=axis) if reverse else data
    outgoing = np.flip(value, axis=axis) if reverse else value
    length = data.shape[axis]
    accumulator = _SCAN_ACCUMULATORS.get(dtype, dtype)
    outer_count = data.size // length
    with np.errstate(all='ignore'):
        if length == 1:
            # One native prefix adds +0 after the requested input conversion.
            np.add(incoming, 0, out=outgoing, dtype=dtype, casting='unsafe')
            return value
        if data.size <= 64:
            # Native starts each non-scalar prefix at +0. Adding +0 after a
            # short scan gives the same finite sums and canonical zero signs.
            if accumulator == dtype:
                np.add.accumulate(incoming, axis=axis, dtype=dtype, out=outgoing)
                if dtype.kind in 'fc':np.add(value, 0, out=value)
            else:
                cast = incoming if incoming.dtype == dtype else incoming.astype(dtype)
                work = np.add.accumulate(cast, axis=axis, dtype=accumulator)
                np.add(work, 0, out=work)
                np.copyto(outgoing, work, casting='unsafe')
            return value
        if length <= 8 and outer_count >= 256 and abs(incoming.strides[axis]) != incoming.dtype.itemsize:
            # A few prefixes over many independent rows are vector operations.
            rows = np.moveaxis(incoming, axis, 0)
            outputs = np.moveaxis(outgoing, axis, 0)
            if accumulator == dtype:
                np.add(rows[0], 0, out=outputs[0], dtype=dtype, casting='unsafe')
                for index in range(1, length):
                    np.add(outputs[index-1], rows[index], out=outputs[index], dtype=dtype, casting='unsafe')
            else:
                carry = np.zeros(rows.shape[1:], dtype=accumulator)
                for row, output in zip(rows, outputs):
                    if row.dtype != dtype:row = row.astype(dtype, copy=False)
                    np.add(carry, row, out=carry, dtype=accumulator, casting='unsafe')
                    np.copyto(output, carry, casting='unsafe')
            return value
        if accumulator == dtype:
            np.add.accumulate(incoming, axis=axis, dtype=dtype, out=outgoing)
            if dtype.kind in 'fc' and incoming.dtype.kind in 'fc':
                first_index = [slice(None)] * data.ndim;first_index[axis] = 0
                first = incoming[tuple(first_index)]
                if data.ndim == 1:
                    negative_zero = (first.real == 0 and np.signbit(first.real) or
                                     dtype.kind == 'c' and first.imag == 0 and np.signbit(first.imag))
                else:
                    # A conservative positive result only adds an extra +0.
                    negative_zero = ((first.real == 0).any() and np.signbit(first.real).any() or
                                     dtype.kind == 'c' and (first.imag == 0).any() and np.signbit(first.imag).any())
                if negative_zero:np.add(value, 0, out=value)
            return value
        limit = max(1, _SCAN_WORK_BYTES // accumulator.itemsize)
        if data.size <= limit:
            cast = incoming if incoming.dtype == dtype else incoming.astype(dtype)
            work = np.array(cast, dtype=accumulator, copy=True, order='C');del cast
            first_index = [slice(None)] * data.ndim;first_index[axis] = slice(0,1)
            first = work[tuple(first_index)]
            np.add(first, 0, out=first)
            np.add.accumulate(work, axis=axis, out=work)
            np.copyto(outgoing, work, casting='unsafe')
            return value
        # Carry remains in opmath precision between bounded blocks. Add it to
        # the first input before each scan, retaining native left-to-right order.
        block = max(1, min(length, limit // outer_count))
        shape = list(data.shape);shape[axis] = block
        workspace = np.empty(shape, dtype=accumulator)
        carry = np.zeros(data.shape[:axis]+data.shape[axis+1:], dtype=accumulator)
        expanded_carry = np.expand_dims(carry, axis)
        first_index = [slice(None)] * data.ndim;first_index[axis] = slice(0,1)
        last_index = [slice(None)] * data.ndim;last_index[axis] = -1
        for start in range(0, length, block):
            count = min(block, length-start)
            source_index = [slice(None)] * data.ndim;source_index[axis] = slice(start,start+count)
            work_index = [slice(None)] * data.ndim;work_index[axis] = slice(0,count)
            work = workspace[tuple(work_index)]
            source = incoming[tuple(source_index)]
            if source.dtype != dtype:source = source.astype(dtype)
            np.copyto(work, source, casting='unsafe')
            first = work[tuple(first_index)]
            np.add(first, expanded_carry, out=first)
            if count > 1:np.add.accumulate(work, axis=axis, out=work)
            np.copyto(outgoing[tuple(source_index)], work, casting='unsafe')
            if start+count < length:np.copyto(carry, work[tuple(last_index)])
    return value


def _scan_result(input, axis, dtype, *, reverse=False):
    _autograd = _tensor._autograd
    data = _scan_array(input._data, axis, dtype, reverse=reverse)
    if (not _state._grad_enabled or not input.requires_grad) and not isinstance(input, _autograd.ForwardTensor):
        # The normal result constructor has no parents or recording work here.
        # Keep the same owned Tensor metadata without creating an unused VJP.
        return _tensor.Tensor(data, _metadata_owned=True)
    input_dtype = input.dtype
    def backward(gradient):
        return _scan_array(gradient, axis, input_dtype, reverse=not reverse)
    return _autograd.result(data, (input, backward), saves_tensors=False,
                            tensor_rule=('cumsum', (axis, reverse)))


def cumsum(input, dim, *, dtype=None):
    _autograd = _tensor._autograd
    extent = input.ndim or 1
    if type(dim) is int and -extent <= dim < extent:
        axis = dim % extent
    else:
        if dim is None:raise RuntimeError('cumsum needs an integer dimension')
        dim = _layout.integer(dim)
        if not -(1 << 63) <= dim < (1 << 63):raise RuntimeError('Overflow when unpacking long')
        axis = _layout.dimension(dim, input.ndim)
    source_dtype = input.dtype
    target = (_SCAN_INT64 if source_dtype.kind in 'biu' else source_dtype) if dtype is None else np.dtype(dtype)
    if target.char not in '?bBhHiIlLqQefdFD':
        raise TypeError('cumsum dtype must be a supported numeric dtype')
    dual = isinstance(input, _autograd.ForwardTensor)
    if target.kind not in 'fc' and (dual or _state._grad_enabled and input.requires_grad):
        raise RuntimeError('Autograd does not support a non-floating cumsum dtype')
    if (target.kind == 'c' or source_dtype.kind == 'c') and (dual or _state._grad_enabled and input.requires_grad):
        raise _errors.BrowserUnsupported('Complex cumsum autograd requires the server')
    if (target.kind == 'b' or target.kind == 'u' and target.itemsize > 1) and input.ndim and input.numel():
        raise RuntimeError('cumsum is not implemented for this output dtype')
    if dual:
        primal = cumsum(input._primal, dim, dtype=dtype)
        if target != source_dtype and (_state._functional_jvp_active or _state._hvp_active):
            # The transpose of native backward scans in the original input
            # dtype, then casts into the requested output space.
            tangent = cumsum(input._tangent, dim, dtype=source_dtype).to(dtype=target)
        else:
            tangent = cumsum(input._tangent, dim, dtype=dtype)
        return _autograd.ForwardTensor(primal, tangent)
    return _scan_result(input, axis, target)


def _product_scan(data, axis, dtype):
    if not data.ndim:
        with np.errstate(all='ignore'):
            return np.array(data, dtype=dtype, copy=True)
    if not data.size:
        return _layout.contiguous_empty(np.empty(data.shape,dtype=dtype))
    accumulator = _SCAN_ACCUMULATORS.get(dtype, dtype)
    if accumulator == dtype or data.size <= _SCAN_WORK_BYTES // accumulator.itemsize:
        # A single accumulation needs no carry or block geometry. When the
        # accumulator is also the output dtype its fresh result needs no copy
        # into a separate output array. Wider accumulators stay within the existing
        # planning budget here; larger widened scans keep the blocked path.
        # Still narrow first and retain complex's initial identity product.
        with np.errstate(all='ignore'):
            source = data if data.dtype == dtype else data.astype(dtype)
            if dtype.kind == 'c':
                work = np.array(source,dtype=accumulator,copy=True,order='C')
                first_index = [slice(None)] * data.ndim;first_index[axis] = slice(0,1)
                first = work[tuple(first_index)]
                np.multiply(first,1,out=first)
                np.multiply.accumulate(work,axis=axis,out=work)
            elif data.size > 64 and (accumulator != dtype or
                    dtype.kind == 'f' and data.size > _SCAN_WORK_BYTES // accumulator.itemsize):
                # An explicit opmath buffer avoids NumPy's additional cast
                # temporary for widened medium scans. Large non-widened real
                # scans also avoid the measured WASM out-of-place loop cost.
                work = np.array(source,dtype=accumulator,copy=True,order='C')
                np.multiply.accumulate(work,axis=axis,out=work)
            else:work = np.multiply.accumulate(source,axis=axis,dtype=accumulator)
            return np.ascontiguousarray(work.astype(dtype,copy=False))
    value = np.empty(data.shape, dtype=dtype)
    length = data.shape[axis]
    outer = data.size // length
    block = max(1, min(length, _SCAN_WORK_BYTES // accumulator.itemsize // outer))
    shape = list(data.shape);shape[axis] = block
    workspace = np.empty(shape, dtype=accumulator)
    carry = np.ones(data.shape[:axis]+data.shape[axis+1:], dtype=accumulator)
    expanded = np.expand_dims(carry, axis)
    first_index = [slice(None)] * data.ndim;first_index[axis] = slice(0,1)
    last_index = [slice(None)] * data.ndim;last_index[axis] = -1
    with np.errstate(all='ignore'):
        for start in range(0, length, block):
            count = min(block, length-start)
            source_index = [slice(None)] * data.ndim;source_index[axis] = slice(start,start+count)
            work_index = [slice(None)] * data.ndim;work_index[axis] = slice(0,count)
            work = workspace[tuple(work_index)]
            source = data[tuple(source_index)]
            if source.dtype != dtype:source = source.astype(dtype)
            np.copyto(work, source, casting='unsafe')
            # The native identity multiply is observable for complex Inf/NaN.
            first = work[tuple(first_index)]
            np.multiply(first, expanded, out=first)
            if count > 1:np.multiply.accumulate(work, axis=axis, out=work)
            np.copyto(value[tuple(source_index)], work, casting='unsafe')
            if start+count < length:np.copyto(carry, work[tuple(last_index)])
    return value


def _product_backward(gradient, source, value, axis):
    gradient = gradient.astype(source.dtype, copy=False)
    if not source.ndim or source.shape[axis] == 1:
        return gradient
    if not source.size:
        return _layout.contiguous_empty(np.empty(source.shape, dtype=source.dtype))
    with np.errstate(all='ignore'):
        weighted = gradient * value
        zeros = source == 0
        if not zeros.any():
            return _scan_array(weighted, axis, weighted.dtype, reverse=True) / source
        count = np.cumsum(zeros, axis=axis, dtype=np.int64)
        before = count == 0
        scanned = _scan_array(np.where(before, weighted, 0), axis, weighted.dtype, reverse=True)
        result = np.zeros(source.shape, dtype=source.dtype)
        np.divide(scanned, source, out=result, where=before, casting='unsafe')
        # Only the first zero receives a contribution. Compute its omitted
        # products through the interval ending at the second zero. Preserve
        # native masked-reduction arithmetic, including its Inf*0 behavior.
        first = zeros & (count == 1)
        between = (count == 1) & ~first
        suffix = _product_scan(np.where(between, source, 1), axis, source.dtype)
        terms = suffix * np.where(count == 1, gradient, 0)
        contribution = terms.sum(axis=axis, keepdims=True, dtype=source.dtype)
        index = np.argmax(first, axis=axis, keepdims=True)
        previous = np.take_along_axis(value, np.maximum(index-1,0), axis=axis)
        previous = np.where(index == 0, 1, previous)
        np.copyto(result, contribution * previous, where=first, casting='unsafe')
        return result


def _product_prefix(source, axis):
    """Internal prefix reduction with native nonfinite/zero adjoints.

    The public prod primitive retains its own existing support boundary.
    This reduction is used only by cumprod's differentiable zero path.
    """
    with np.errstate(all='ignore'):
        value = np.prod(source._data,axis=axis,keepdims=True,dtype=source.dtype)
    def backward(gradient):
        data = source._data
        with np.errstate(all='ignore'):
            if data.shape[axis] == 1:return gradient
            if not np.any(data == 0):return gradient * (value / data)
            shape = list(data.shape);shape[axis] = 1
            one = np.ones(shape,dtype=data.dtype)
            key = [slice(None)] * data.ndim;key[axis] = slice(0,-1)
            prefix = _product_scan(np.concatenate((one,data[tuple(key)]),axis),axis,data.dtype)
            key[axis] = slice(1,None)
            suffix = np.concatenate((one,np.flip(data[tuple(key)],axis)),axis)
            suffix = np.flip(_product_scan(suffix,axis,data.dtype),axis)
            return gradient * (prefix * suffix)
    return _tensor._autograd.result(value,(source,backward))


def _product_polynomial_jvp(source, direction, axis):
    """Differentiate omitted products, retaining dependencies at every zero.

    This is the higher-order zero path. Like native's differentiable zero
    pullback it is quadratic in the scan length; ordinary reverse uses the
    linear masked scan above instead.
    """
    ops = _tensor._ops
    length = source.shape[axis]
    shape = list(source.shape);shape[axis] = 1
    one = _tensor.Tensor(np.ones(shape, dtype=source.dtype), _metadata_owned=True)
    result = None
    for index in reversed(range(length)):
        key = [slice(None)] * source.ndim;key[axis] = slice(0,index)
        prefix = _product_prefix(source[tuple(key)],axis) if index else one
        key[axis] = slice(index+1,None)
        if index+1 < length:
            suffix = source[tuple(key)].cumprod(axis)
            omitted = ops.cat((prefix, prefix * suffix), axis)
        else:omitted = prefix
        key[axis] = slice(index,index+1)
        term = direction[tuple(key)] * omitted
        if index:
            padding = list(source.shape);padding[axis] = index
            term = ops.cat((_tensor.Tensor(np.zeros(padding,dtype=source.dtype),_metadata_owned=True),term),axis)
        result = term if result is None else result + term
    return result


def _product_forward(dual, axis, target):
    autograd, ops = _tensor._autograd, _tensor._ops
    source, direction = dual._primal, dual._tangent
    primal = cumprod(source, axis, dtype=target)
    reverse_based = _state._functional_jvp_active or _state._hvp_active
    if reverse_based:
        if not source.ndim or not source.numel() or source.shape[axis] == 1:
            tangent = direction.to(dtype=target)
        elif np.any(source._data == 0):
            tangent = _product_polynomial_jvp(source, direction, axis).to(dtype=target)
        else:
            common = np.promote_types(source.dtype, target)
            ratio = direction.to(dtype=common) / source
            tangent = (ratio.cumsum(axis) * primal).to(dtype=source.dtype).to(dtype=target)
    else:
        tangent = (direction / source).cumsum(axis) * primal
        zeros = source._data == 0
        if not source.ndim:
            tangent = ops.where(_tensor.Tensor(zeros), direction, tangent)
        elif source.numel() and zeros.any():
            count = np.cumsum(zeros, axis=axis, dtype=np.int64)
            first = zeros & (count == 1)
            substituted = ops.where(_tensor.Tensor(first), direction, source)
            replacement = substituted.cumprod(axis)
            tangent = ops.where(_tensor.Tensor(count != 0), replacement, tangent)
        tangent = tangent.to(dtype=target)
    return autograd.ForwardTensor(primal, tangent)


def cumprod(input, dim, *, dtype=None):
    autograd = _tensor._autograd
    extent = input.ndim or 1
    if type(dim) is int and -extent <= dim < extent:
        axis = dim % extent
    else:
        if dim is None:raise RuntimeError('cumprod needs an integer dimension')
        dim = _layout.integer(dim)
        if not -(1 << 63) <= dim < (1 << 63):raise RuntimeError('Overflow when unpacking long')
        axis = _layout.dimension(dim,input.ndim)
    source_dtype = input.dtype
    target = (_SCAN_INT64 if source_dtype.kind in 'biu' else source_dtype) if dtype is None else np.dtype(dtype)
    if target.char not in '?bBhHiIlLqQefdFD':
        raise TypeError('cumprod dtype must be a supported numeric dtype')
    dual = isinstance(input, autograd.ForwardTensor)
    if target.kind not in 'fc' and (dual or _state._grad_enabled and input.requires_grad):
        raise RuntimeError('Autograd does not support a non-floating cumprod dtype')
    if (target.kind == 'c' or source_dtype.kind == 'c') and (dual or _state._grad_enabled and input.requires_grad):
        raise _errors.BrowserUnsupported('Complex cumprod autograd requires the server')
    if (target.kind == 'b' or target.kind == 'u' and target.itemsize > 1) and input.ndim and input.numel():
        raise RuntimeError('cumprod is not implemented for this output dtype')
    if dual:
        if target not in autograd._FUNCTIONAL_JVP_REAL_DTYPES:
            raise _errors.BrowserUnsupported('Forward mode requires real float32/float64 outputs')
        return _product_forward(input, axis, target)
    value = _product_scan(input._data, axis, target)
    if not _state._grad_enabled or not input.requires_grad:
        return _tensor.Tensor(value, _metadata_owned=True)
    return autograd.result(value, (input, lambda gradient:_product_backward(gradient,input._data,value,axis)))


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
