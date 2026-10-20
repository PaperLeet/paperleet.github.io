"""Dense affine matrix/vector products with one analytical reverse node."""
import sys
import numpy as np

_WASM = sys.platform == 'emscripten'
_HALF = np.dtype('float16')
_FLOAT = np.dtype('float32')


def _scalar(value):
    if isinstance(value, _tensor.Tensor):
        dual = isinstance(value, _autograd.ForwardTensor)
        reverse = value.requires_grad and (not dual or _state._functional_jvp_active or _state._hvp_active)
        if value.ndim or reverse:raise TypeError('Product scales must be Numbers or detached scalar Tensors')
        value = (value._primal if dual else value)._data.item()
    return _promotion.scalar_info(value)


def _cast_scalar(value, dtype):
    if dtype.kind == 'b':return np.asarray(bool(value), dtype=dtype)
    if isinstance(value, complex) and dtype.kind != 'c':
        if value.imag:raise RuntimeError('A complex scale cannot be converted to a real dtype')
        value = value.real
    if dtype.kind == 'i':
        maximum = 2**(dtype.itemsize*8-1)-1
        if value < -maximum-1 or value > maximum:raise RuntimeError('Scalar conversion would overflow')
    if dtype == _HALF and np.isfinite(value) and abs(value) > 65504:
        raise RuntimeError('Scalar conversion would overflow float16')
    try:return _promotion.scalar_array(value, dtype, parsed=True)
    except (ValueError, TypeError, OverflowError) as error:raise RuntimeError(str(error)) from None


def _can_cast(source, target):
    ranks = {'b':0, 'i':1, 'u':1, 'f':2, 'c':3}
    return ranks[source.kind] <= ranks[target.kind]


def _shape(input, left, right, matrix):
    if matrix and (input.dtype != left.dtype or left.dtype != right.dtype):
        raise RuntimeError('addmm operands must have the same dtype')
    if left.ndim != 2 or right.ndim != (2 if matrix else 1):
        raise RuntimeError('Fused products require a matrix and a matrix/vector')
    if left.shape[1] != right.shape[0]:raise RuntimeError('Fused product contraction dimensions do not match')
    shape = (left.shape[0], right.shape[1]) if matrix else (left.shape[0],)
    if input.ndim > len(shape) or any(x != y and x != 1 for x,y in zip(reversed(input.shape), reversed(shape))):
        raise RuntimeError('Product input cannot broadcast to the result')
    return shape


def _product(a, b):
    return _matmul.product(a, b) if _WASM and a.size >= 4096 else a @ b


def _value(input, left, right, shape, alpha, beta, beta_kind, matrix):
    c,a,b = input._data,left._data,right._data
    dtype = a.dtype if matrix else b.dtype
    if matrix and (not shape[0] or not shape[1]):return _layout.contiguous_empty(np.empty(shape,dtype=dtype))
    if not a.size:
        if beta == 0:
            value = np.zeros(shape,dtype=dtype)
            return _layout.contiguous_empty(value) if not value.size else value
        if matrix:
            if not _can_cast(beta_kind,dtype):raise RuntimeError('Scaled input cannot be cast to the product dtype')
            compute = _FLOAT if dtype == _HALF else dtype
            scale = _arithmetic.scalar_data(beta,dtype).astype(compute)
            with np.errstate(all='ignore'):
                value = np.ascontiguousarray((np.broadcast_to(c,shape).astype(compute,copy=False)*scale).astype(dtype,copy=False))
        else:
            if not _can_cast(c.dtype,dtype):raise RuntimeError('Scaled input cannot be cast to the product dtype')
            scale = _cast_scalar(beta,c.dtype)
            compute = _FLOAT if c.dtype == _HALF else c.dtype
            with np.errstate(all='ignore'):
                value = (c.astype(compute,copy=False)*scale.astype(compute)).astype(c.dtype,copy=False).astype(dtype,copy=False)
        return _layout.contiguous_empty(value) if not value.size else value
    if a.dtype != b.dtype:raise RuntimeError('Product factors must have the same dtype')
    if a.dtype.kind == 'b' or a.dtype.kind == 'u' and a.dtype.itemsize > 1:
        raise RuntimeError('This matrix product kernel is unavailable for the dtype')
    compute = _FLOAT if dtype == _HALF else dtype
    scale_dtype = compute if matrix else dtype
    aa,bb = _cast_scalar(alpha,scale_dtype),_cast_scalar(beta,scale_dtype)
    aa,bb = aa.astype(compute),bb.astype(compute)
    if matrix and dtype != _HALF and aa == 0 and bb == 1:return np.broadcast_to(c,shape).copy(order='C')
    with np.errstate(all='ignore'):
        prescale = matrix and dtype.kind == 'c' and not np.isfinite(aa)
        first = a.astype(compute,copy=False)
        if prescale:first = first*aa
        evaluate = aa != 0 or dtype == _HALF or not matrix and dtype.kind == 'c'
        value = _product(first,b.astype(compute,copy=False)) if evaluate else np.zeros(shape,dtype=compute)
        if not prescale and evaluate and aa != 1:np.multiply(value,aa,out=value)
        if bb != 0:
            bias = c.astype(dtype,copy=False).astype(compute,copy=False)
            if bb != 1 or dtype.kind == 'c':bias = bias*bb
            np.add(value,bias,out=value)
        value = value.astype(dtype,copy=False)
    return _layout.contiguous_empty(value) if not value.size else value


def _scaled(value, number):
    return value if not isinstance(number,complex) and number == 1 else value * number


def _gradient_scale(value, number):
    # Reverse rules use ordinary wrapped-scalar multiplication, not the checked
    # BLAS argument conversion. Even complex(1, 0) produces a complex gradient,
    # which cannot be delivered to a real leaf.
    if isinstance(number,complex) and value.dtype.kind != 'c':
        raise RuntimeError('A complex product gradient cannot be delivered to a real input')
    return value if number == 1 else value * _arithmetic.scalar_data(number,value.dtype)


def _fused(input, left, right, beta, alpha, matrix):
    if not all(isinstance(v,_tensor.Tensor) for v in (input,left,right)):
        raise TypeError('Fused product operands must be Tensors')
    alpha,alpha_kind = _scalar(alpha);beta,beta_kind = _scalar(beta)
    shape = _shape(input,left,right,matrix)
    duals = [isinstance(v,_autograd.ForwardTensor) for v in (input,left,right)]
    if any(duals):
        values = tuple(v._primal if dual else v for v,dual in zip((input,left,right),duals))
        c,a,b = values;primal = _fused(c,a,b,beta,alpha,matrix);terms=[]
        reverse_based = _state._functional_jvp_active or _state._hvp_active
        partial = not all(duals)
        tangent_shapes = [tuple(input.shape) if i == 0 else shape for i,dual in enumerate(duals) if dual]
        if not reverse_based and partial:tangent_shapes.append(shape)
        try:tangent_shape = np.broadcast_shapes(*tangent_shapes)
        except ValueError:raise RuntimeError('Product tangent shapes do not broadcast') from None
        if np.broadcast_shapes(tangent_shape,primal.shape) != primal.shape:
            raise RuntimeError('Product tangent shape does not match its primal')
        if reverse_based and (isinstance(alpha,complex) or isinstance(beta,complex)):
            raise RuntimeError('A complex product gradient cannot be delivered to a real input')
        if duals[0]:terms.append(_scaled(input._tangent,beta))
        operation = _ops.mm if matrix else _ops.mv
        if duals[1]:
            terms.append(operation(_scaled(left._tangent,alpha),b) if reverse_based
                         else _scaled(operation(left._tangent,b),alpha))
        if duals[2]:
            terms.append(operation(a,_scaled(right._tangent,alpha)) if reverse_based
                         else _scaled(operation(a,right._tangent),alpha))
        tangent = terms[0]
        for term in terms[1:]:tangent = tangent+term
        if not reverse_based and partial:
            # Undefined tangents are efficient ZeroTensors in native forward
            # AD. They still determine the broadcast shape and promoted dtype,
            # and matrix products retain zero reverse edges to their factors.
            bias_dtype = input.dtype if not isinstance(beta,complex) and beta == 1 else _promotion._combine(input.dtype,beta_kind,wrapped=True)
            product_dtype = right.dtype if not isinstance(alpha,complex) and alpha == 1 else _promotion._combine(right.dtype,alpha_kind,wrapped=True)
            dtype = _construction.promote_types(bias_dtype,product_dtype)
            dtype = _construction.promote_types(tangent.dtype,dtype)
            if tangent.dtype != dtype:tangent = tangent.to(dtype=dtype)
            if not duals[1]:tangent = _autograd._forward_zero_edge(tangent,b)
            if not duals[2]:tangent = _autograd._forward_zero_edge(tangent,a)
        try:tangent = tangent.expand(primal.shape) if tangent.shape != primal.shape else tangent
        except RuntimeError:raise RuntimeError('Product tangent shape does not match its primal') from None
        return _autograd.ForwardTensor(primal,tangent)
    value = _value(input,left,right,shape,alpha,beta,beta_kind,matrix)
    needs_grad = _state._grad_enabled and any(v.requires_grad for v in (input,left,right))
    if needs_grad and value.dtype.kind not in 'fc':
        raise RuntimeError('An integer or boolean product result cannot carry gradients')
    if not needs_grad:return _tensor.Tensor(value,_metadata_owned=True)
    if any(v.requires_grad and v.dtype == _HALF for v in (input,left,right)):
        raise _errors.BrowserUnsupported('Fused product gradients require float32/float64')
    shape_input,shape_left,shape_right = tuple(input.shape),tuple(left.shape),tuple(right.shape)
    def bias_vjp(g):
        with np.errstate(all='ignore'):return _ops.unbroadcast(_gradient_scale(g,beta),shape_input)
    def factor_vjp(g,first):
        try:
            if not matrix and g.ndim != 1:raise RuntimeError('addmv gradients require a vector')
            if matrix:gradient = _product(g,right._data.T) if first else _product(left._data.T,g)
            else:gradient = g[:,None]*right._data if first else _product(left._data.T,g)
            expected = shape_left if first else shape_right
            if gradient.shape != expected:raise RuntimeError('Product gradient shape does not match its input')
            with np.errstate(all='ignore'):return _gradient_scale(gradient,alpha)
        except ValueError as error:raise RuntimeError(str(error)) from None
    saved=[]
    if left.requires_grad:saved.append(right)
    if right.requires_grad:saved.append(left)
    return _autograd.result(value,(input,bias_vjp),(left,lambda g:factor_vjp(g,True)),
        (right,lambda g:factor_vjp(g,False)),versioned=saved)


def addmm(input,mat1,mat2,*,beta=1,alpha=1,out=None):
    if out is not None:
        if not isinstance(out,_tensor.Tensor):raise TypeError('Product output must be a Tensor')
        raise _errors.BrowserUnsupported('Product output buffers require the server')
    return _fused(input,mat1,mat2,beta,alpha,True)


def addmv(input,mat,vec,*,beta=1,alpha=1,out=None):
    if out is not None:
        if not isinstance(out,_tensor.Tensor):raise TypeError('Product output must be a Tensor')
        raise _errors.BrowserUnsupported('Product output buffers require the server')
    return _fused(input,mat,vec,beta,alpha,False)


from . import _arithmetic,_autograd,_construction,_errors,_layout,_matmul,_ops,_promotion,_state,_tensor
