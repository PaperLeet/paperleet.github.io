"""Dense CPU normalization, shared analytical VJPs and forward-mode rules."""
import numbers
import weakref
import numpy as np

from . import _autograd, _defaults, _device, _errors, _ops, _tensor
from ._module import Module, Parameter


def _shape(value, *, constructor=False):
    if constructor and isinstance(value, numbers.Integral) and not isinstance(value, (bool,np.bool_)):
        value = (int(value),)
    elif constructor:
        try:value = tuple(value)
        except TypeError:raise TypeError('normalized_shape must be an integer or iterable') from None
    if not isinstance(value, (tuple,list)):
        raise TypeError('normalized_shape must be a tuple or list of integers')
    if any(not isinstance(n,numbers.Integral) or isinstance(n,(bool,np.bool_)) for n in value):
        raise TypeError('normalized_shape must contain integers')
    shape = tuple(int(n) for n in value)
    if any(n < 0 for n in shape):raise RuntimeError('Negative normalization dimensions')
    return shape


def parameter_dtype(dtype, device, *, allocate=True):
    _device.check_device(device)
    if not allocate:return None
    dtype = _defaults._float_dtype if dtype is None else np.dtype(dtype)
    if dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise _errors.BrowserUnsupported('Neural-network parameters require float32/float64')
    return dtype


def _validate(input, shape, weight, bias, eps):
    if not isinstance(input,_tensor.Tensor):raise TypeError('Normalization input must be a Tensor')
    if not shape or input.ndim<len(shape) or input.shape[-len(shape):]!=shape:
        raise RuntimeError('normalized_shape does not match the input trailing dimensions')
    if input.dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise _errors.BrowserUnsupported('Normalization requires real float32/float64')
    for name, value in (('weight',weight),('bias',bias)):
        if value is None:continue
        if not isinstance(value,_tensor.Tensor):raise TypeError(f'{name} must be a Tensor')
        if value.shape!=shape or value.dtype!=input.dtype:
            raise RuntimeError(f'Normalization {name} shape/dtype does not match the input')
    if not isinstance(eps,numbers.Real):raise TypeError('eps must be a real number')


def _layer_statistics(source, axes, shape):
    if source.dtype==np.dtype('float32'):
        # Accumulate before rounding. Small ordinary mean/variance errors can
        # amplify through successive normalization mixed derivatives even when
        # a row does not meet a large-offset cancellation heuristic.
        high = source.astype(np.float64)
        centered = high-high.mean(axis=axes,keepdims=True)
        variance = np.mean(centered*centered,axis=axes,keepdims=True)
        return centered.astype(np.float32),variance.astype(np.float32)
    centered = source-source.mean(axis=axes,keepdims=True)
    return centered,np.mean(centered*centered,axis=axes,keepdims=True)


def _forward(kind, input, shape, weight, bias, eps):
    x,dx = _autograd.forward_parts(input)
    w,dw = _autograd.forward_parts(weight) if weight is not None else (None,None)
    b,db = _autograd.forward_parts(bias) if bias is not None else (None,None)
    primal = _normalize(kind,x,shape,w,b,eps)
    if not x.numel():
        dependencies = (x,dx,w,dw,db)
        edges = [(v,lambda g,shape=v.shape,dtype=v.dtype:np.zeros(shape,dtype=dtype))
                 for v in dependencies if v is not None]
        tangent = _autograd.result(np.empty_like(x._data),*edges)
        return _autograd.ForwardTensor(primal,tangent)
    axes = tuple(range(x.ndim-len(shape),x.ndim))
    if kind=='layer' and _autograd._state._hvp_active:
        # Functional HVP differentiates the full statistics. Ordinary native
        # JVP below deliberately retains its saved-statistic reverse tape.
        high = x.to(dtype=np.float64) if x.dtype==np.dtype('float32') else x
        direction = dx.to(dtype=high.dtype)
        centered = high-high.mean(axes,True)
        inverse = (centered.square().mean(axes,True)+eps).rsqrt()
        projected = direction-direction.mean(axes,True)
        tangent = projected*inverse-centered*inverse.pow(3)*(projected*centered).mean(axes,True)
        if w is not None:tangent = tangent*w.to(dtype=high.dtype)+(centered*inverse)*dw.to(dtype=high.dtype)
        if b is not None:tangent = tangent+db.to(dtype=high.dtype)
        return _autograd.ForwardTensor(primal,tangent.to(dtype=x.dtype))
    if kind=='layer':
        if (x.dtype==np.dtype('float32') and np.isfinite(eps)
                and eps>=np.finfo(np.float32).eps and np.isfinite(x._data).all()
                and np.isfinite(primal._data).all()):
            # Keep the public dtype, but avoid accumulating several FP32
            # rounding stages in the LayerNorm tangent and its reverse tape.
            # The mean/inverse remain frozen on that tape, as in the native
            # LayerNorm JVP. Tiny/non-positive eps and non-finite inputs retain
            # the original IEEE path below, as do non-finite primal results.
            data,variance = _layer_statistics(x._data.astype(np.float64),axes,shape)
            inverse = _tensor.Tensor(1/np.sqrt(variance+eps))
            centered = _autograd.result(data,(x,lambda g:g))
            direction = dx.to(dtype=np.float64)
            projected = direction-direction.mean(axes,True)
            inverse_tangent = (-inverse.pow(3)*projected*centered).mean(axes,True)
            tangent = projected*inverse+centered*inverse_tangent
            if w is not None:
                tangent = tangent*w.to(dtype=np.float64)+(centered*inverse)*dw.to(dtype=np.float64)
            if b is not None:tangent = tangent+db.to(dtype=np.float64)
            return _autograd.ForwardTensor(primal,tangent.to(dtype=np.float32))
        with np.errstate(over='ignore',invalid='ignore'):
            data,variance = _layer_statistics(x._data,axes,shape)
            inverse = _tensor.Tensor(1/np.sqrt(variance+eps))
        # Native LayerNorm's JVP holds its saved mean/inverse statistics fixed
        # on the reverse tape. Keep the explicit centered-tangent projection;
        # algebraic simplification would change reverse-over-forward gradients.
        centered = _autograd.result(data,(x,lambda g:g))
        projected = dx-dx.mean(axes,True)
        inverse_tangent = (-inverse.pow(3)*projected*centered).mean(axes,True)
        tangent = projected*inverse+centered*inverse_tangent
    else:
        centered = x
        inverse = (centered.square().mean(axes,True)+eps).rsqrt()
        tangent_variance = (2*x*dx).mean(axes,True)
        tangent = dx*inverse+x*(tangent_variance*(-.5)*inverse.pow(3))
    normalized = centered*inverse
    if w is not None:tangent = tangent*w+normalized*dw
    if b is not None:tangent = tangent+db
    return _autograd.ForwardTensor(primal,tangent)


def _normalize(kind, input, shape, weight, bias, eps):
    _validate(input,shape,weight,bias,eps)
    if any(isinstance(v,_autograd.ForwardTensor) for v in (input,weight,bias)):
        return _forward(kind,input,shape,weight,bias,eps)
    source = input._data
    axes = tuple(range(input.ndim-len(shape),input.ndim))
    prefix = tuple(range(input.ndim-len(shape)))
    scale = weight._data if weight is not None else 1
    if source.size:
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            if kind=='layer':centered,variance = _layer_statistics(source,axes,shape)
            else:centered,variance = source,np.mean(source*source,axis=axes,keepdims=True)
            inverse = 1/np.sqrt(variance+eps)
            normalized = centered*inverse
    else:
        normalized = np.empty_like(source)
        inverse = None
    value = normalized*scale
    if bias is not None:value = value+bias._data
    operands = [input]
    if weight is not None:operands.append(weight)
    if bias is not None:operands.append(bias)
    original = tuple((weakref.ref(t),weakref.ref(t._data)) for t in operands)
    size,input_shape,input_dtype = source.size,input.shape,input.dtype
    has_weight,has_bias = weight is not None,bias is not None
    active_input = input.requires_grad
    backward_normalized = normalized if active_input or has_weight and weight.requires_grad else None
    backward_inverse = inverse if active_input else None
    backward_source = source if active_input and kind=='rms' else None
    backward_scale = scale if active_input else None
    def backward(g, needs):
        for tensor_ref,data_ref in original:
            tensor = tensor_ref()
            if tensor is not None and tensor._data is not data_ref():
                raise _errors.BrowserUnsupported('Normalization backward after storage rebinding is unsupported')
        values = [None]*len(needs)
        if needs[0]:
            if not size:values[0] = np.empty(input_shape,dtype=input_dtype)
            else:
                scaled = g*backward_scale
                if kind=='layer':
                    # Keep the subtractive order: an equivalent projection sum
                    # is measurably slower for float64 arrays on real WASM.
                    values[0] = backward_inverse*(scaled-scaled.mean(axis=axes,keepdims=True)-backward_normalized*np.mean(scaled*backward_normalized,axis=axes,keepdims=True))
                else:
                    # Preserve the composite native rule's underflow/overflow
                    # order instead of algebraically canceling inverse powers.
                    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                        reduced=(scaled*backward_source).sum(axis=axes,keepdims=True)
                        second=(reduced*(-.5)*(backward_inverse**3))/int(np.prod(shape))
                        values[0]=scaled*backward_inverse+(second*2)*backward_source
        index = 1
        if has_weight:
            if needs[index]:values[index] = (g*backward_normalized).sum(axis=prefix) if prefix else g*backward_normalized
            index += 1
        if has_bias and needs[index]:values[index] = g.sum(axis=prefix) if prefix else g
        return values
    return _autograd.joint_result(value,operands,backward)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return _normalize('layer',input,_shape(normalized_shape),weight,bias,eps)


def rms_norm(input, normalized_shape, weight=None, eps=None):
    valid = isinstance(input,_tensor.Tensor) and input.dtype in (np.dtype('float32'),np.dtype('float64'))
    epsilon = (np.finfo(input.dtype).eps if valid else 0) if eps is None else eps
    return _normalize('rms',input,_shape(normalized_shape),weight,None,epsilon)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
        super().__init__()
        dtype = parameter_dtype(dtype,device,allocate=elementwise_affine)
        self.normalized_shape = _shape(normalized_shape,constructor=True)
        self.eps, self.elementwise_affine = eps, elementwise_affine
        self.register_parameter('weight',Parameter(_tensor.Tensor(np.empty(self.normalized_shape,dtype=dtype))) if elementwise_affine else None)
        self.register_parameter('bias',Parameter(_tensor.Tensor(np.empty(self.normalized_shape,dtype=dtype))) if elementwise_affine and bias else None)
        self.reset_parameters()

    def reset_parameters(self):
        with _autograd.no_grad():
            if self.weight is not None:self.weight.fill_(1)
            if self.bias is not None:self.bias.zero_()

    def forward(self,input):return layer_norm(input,self.normalized_shape,self.weight,self.bias,self.eps)

    def extra_repr(self):
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, bias={self.bias is not None}'


class RMSNorm(Module):
    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        dtype = parameter_dtype(dtype,device,allocate=elementwise_affine)
        self.normalized_shape = _shape(normalized_shape,constructor=True)
        self.eps, self.elementwise_affine = eps, elementwise_affine
        self.register_parameter('weight',Parameter(_tensor.Tensor(np.ones(self.normalized_shape,dtype=dtype))) if elementwise_affine else None)

    def reset_parameters(self):
        if self.weight is not None:
            with _autograd.no_grad():self.weight.fill_(1)

    def forward(self,input):return rms_norm(input,self.normalized_shape,self.weight,self.eps)

    def extra_repr(self):
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
