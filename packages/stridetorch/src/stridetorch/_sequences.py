"""Bounded CPU sequence factories with Torch scalar and rounding conventions."""
import math
import operator
from functools import lru_cache

import numpy as np

from . import _construction, _conversion, _defaults, _device, _errors, _layout, _ops, _tensor


_LIMIT = 16_000_000
_REAL = tuple(np.dtype(n) for n in ('float16','float32','float64','int8','uint8','int16','int32','int64'))
_COMPLEX = (np.dtype('complex64'),np.dtype('complex128'))
_UNIT_INTERVALS = ((0,1),(-1,1),(1,0),(1,-1))


def _options(device, requires_grad, layout, pin_memory, out):
    _device.check_device(device)
    if type(requires_grad) is not bool:raise TypeError('requires_grad must be bool')
    if pin_memory is not None and type(pin_memory) is not bool:raise TypeError('pin_memory must be bool')
    if pin_memory or layout is not None or out is not None:
        raise _errors.BrowserUnsupported('Pinned/non-default-layout/output-buffer sequence factories require the server')


def _steps(value):
    if isinstance(value,(bool,np.bool_)):raise TypeError('steps must be an integer')
    if isinstance(value,_tensor.Tensor) and (value.ndim or value.dtype.kind not in 'iu'):
        raise TypeError('steps must be a scalar integer')
    result=operator.index(value)
    if not -(2**63)<=result<2**63:raise ValueError('steps must fit a signed 64-bit integer')
    if result<0:raise RuntimeError('number of steps must be non-negative')
    if result>_LIMIT:raise _errors.BrowserUnsupported('Sequence allocation exceeds the browser limit')
    return result


def _dtype(dtype, *kinds, arange=False):
    if dtype is not None:return _conversion.dtype(dtype)
    if arange:return _defaults._float_dtype if any(k.kind in 'fc' for k in kinds) else np.dtype('int64')
    return _defaults.complex_dtype() if any(k.kind=='c' for k in kinds) else _defaults._float_dtype


def _real(value):
    if isinstance(value,complex):
        if value.imag:raise RuntimeError('A complex value with a nonzero imaginary part cannot be converted to real')
        return value.real
    return value


def _endpoint(value, dtype, *, fill=False):
    if dtype.kind not in 'cb':value=_real(value)
    if dtype==np.dtype('int64') and type(value) is int:
        if not -(2**63)<=value<2**63:raise RuntimeError('Endpoint is outside int64 storage')
        return value
    if dtype==np.dtype('float64'):return float(value)
    if dtype==np.dtype('complex128'):return complex(value)
    if dtype.kind in 'fc' and value in (0,1,-1):
        return complex(value) if dtype.kind=='c' else float(value)
    if fill and dtype==np.dtype('float16'):
        value=_endpoint(value,np.dtype('float32'))
        with np.errstate(all='ignore'):return np.asarray(value,dtype=dtype).item()
    if dtype.kind in 'biu':
        try:
            if dtype==np.dtype('uint64'):
                if isinstance(value,float):
                    if not math.isfinite(value) or value<0 or value>=2**64:raise RuntimeError('Value is outside uint64 storage')
                    value=math.trunc(value)
                else:value=operator.index(value)
                value%=2**64
            else:value=_construction._scalar(value,dtype)
        except (ValueError,OverflowError) as error:raise RuntimeError(str(error)) from error
    else:
        number=complex(value) if dtype.kind=='c' else float(value)
        limit=float(np.finfo(dtype).max)
        parts=(number.real,number.imag) if dtype.kind=='c' else (number,)
        if any(math.isfinite(part) and abs(part)>limit for part in parts):
            raise RuntimeError(f'Endpoint cannot be converted to {dtype} without overflow')
        value=number
    with np.errstate(all='ignore'):return np.asarray(value,dtype=dtype).item()


def _result(value, requires_grad):
    if not value.size:value=_layout.contiguous_empty(value)
    return _tensor.Tensor(value,requires_grad=requires_grad,_metadata_owned=True)


def _arange_arguments(args, kwargs):
    if len(args)<=1 and not any(k!='end' for k in kwargs):
        if len(args)==1 and not kwargs:return 0,args[0],1
        if not args and 'end' in kwargs:return 0,kwargs['end'],1
    names=('start','end','step')
    if len(args)>3 or any(k not in names or names.index(k)<len(args) for k in kwargs):
        raise TypeError('arange received invalid or duplicate arguments')
    values=dict(zip(names,args));values.update(kwargs)
    if 'start' not in values or 'end' not in values:raise TypeError('arange expects end or start, end[, step]')
    return values['start'],values['end'],values.get('step',1)


def arange(*args,dtype=None,device=None,requires_grad=False,out=None,layout=None,pin_memory=False,**kwargs):
    _options(device,requires_grad,layout,pin_memory,out)
    arguments=_arange_arguments(args,kwargs)
    if any(isinstance(value,_tensor.Tensor) and value.requires_grad and not isinstance(value,_ops._autograd.ForwardTensor) for value in arguments):
        raise TypeError('arange scalar controls cannot be ordinary Tensors requiring gradients')
    parsed=[_ops._factory_number(value) for value in arguments]
    target=_dtype(dtype,*(kind for _,kind in parsed),arange=True)
    if target not in _REAL:raise _errors.BrowserUnsupported(f'arange has no CPU/WASM kernel for {target}')
    start,end,step=(_real(value) for value,_ in parsed)
    a,b,c=float(start),float(end),float(step)
    if math.isnan(c) or not c:raise RuntimeError('step must be nonzero and not NaN')
    if not math.isfinite(a) or not math.isfinite(b):raise RuntimeError('Range endpoints must be finite')
    if c>0 and b<a or c<0 and b>a:raise RuntimeError('upper bound and lower bound inconsistent with step sign')
    if target==np.dtype('int64') and all(type(v) is int for v in (start,end,step)):
        for value in (start,end,step):_endpoint(value,target)
        size=(end-start+step-(1 if step>0 else -1))//step
    else:
        quotient=(b-a)/c
        if not math.isfinite(quotient) or quotient>2**63-1:raise RuntimeError('Invalid range size')
        size=math.ceil(quotient)
    if size>_LIMIT:raise _errors.BrowserUnsupported('Sequence allocation exceeds the browser limit')
    if size<0:raise RuntimeError('Invalid range size')
    if target.kind in 'iu':
        first,increment=_endpoint(start,np.dtype('int64')),_endpoint(step,np.dtype('int64'))
        with np.errstate(all='ignore'):
            if first==0 and increment==1:values=np.arange(size,dtype=target)
            else:
                values=np.arange(size,dtype=np.int64)
                np.multiply(values,np.int64(increment),out=values)
                np.add(values,np.int64(first),out=values)
                values=values.astype(target,copy=False)
    else:
        accumulator=np.dtype('float32') if target==np.dtype('float16') else np.dtype('float64')
        with np.errstate(all='ignore'):
            if start==0 and step==1:values=np.arange(size,dtype=target)
            else:
                first=np.asarray(start,dtype=accumulator);increment=np.asarray(step,dtype=accumulator)
                values=np.arange(size,dtype=accumulator)
                np.multiply(values,increment,out=values);np.add(values,first,out=values)
                values=values.astype(target,copy=False)
    return _result(values,requires_grad)


def _arguments(start,end,steps,dtype):
    count=_steps(steps)
    if any(isinstance(value,_tensor.Tensor) and value.ndim for value in (start,end)):
        raise RuntimeError('linspace/logspace endpoints must be zero-dimensional tensors')
    (start,a),(end,b)=_ops._factory_number(start),_ops._factory_number(end)
    target=_dtype(dtype,a,b)
    if (a.kind=='c' or b.kind=='c') and target.kind!='c':
        raise RuntimeError('Inferred complex dtype cannot be safely cast to a real output dtype')
    return start,end,count,target


def _interpolate(start,end,count,dtype,*,complex_indices=False):
    # Two-ended evaluation avoids compounding drift near the final endpoint.
    half=count//2
    with np.errstate(all='ignore'):
        if dtype.kind=='c' and not complex_indices:
            real=np.dtype('float32') if dtype==np.dtype('complex64') else np.dtype('float64')
            values=np.empty(count,dtype=dtype)
            index=np.arange(count,dtype=real);reverse=np.asarray(count-1,dtype=real)-index[half:]
            for field in ['real','imag']:
                a=np.asarray(getattr(start,field),dtype=real);b=np.asarray(getattr(end,field),dtype=real)
                step=np.divide(b-a,np.asarray(count-1,dtype=real));view=getattr(values,field)
                np.multiply(index[:half],step,out=view[:half]);np.add(view[:half],a,out=view[:half])
                np.multiply(reverse,step,out=view[half:]);np.subtract(b,view[half:],out=view[half:])
            return values
        left=np.asarray(start,dtype=dtype);right=np.asarray(end,dtype=dtype)
        delta=np.divide(right-left,np.asarray(count-1,dtype=dtype))
        values=np.arange(count,dtype=dtype)
        np.multiply(values[:half],delta,out=values[:half]);np.add(values[:half],left,out=values[:half])
        np.subtract(np.asarray(count-1,dtype=dtype),values[half:],out=values[half:])
        np.multiply(values[half:],delta,out=values[half:]);np.subtract(right,values[half:],out=values[half:])
    return values


@lru_cache(maxsize=64)
def _unit_grid_bytes(dtype,count,interval):
    # Only fixed public intervals and <=256 elements enter this cache: at most
    # 128 KiB of numeric bytes, never arbitrary endpoint objects or Tensors.
    if type(dtype) is not str or type(count) is not int or not 2<=count<=256 or type(interval) is not int or not 0<=interval<4:
        raise ValueError('Invalid fixed-interval template key')
    target=np.dtype(dtype)
    if target not in _REAL[:3]:raise ValueError('Fixed-interval templates require a real floating dtype')
    start,end=_UNIT_INTERVALS[interval]
    compute=np.dtype('float32') if target==np.dtype('float16') else target
    with np.errstate(all='ignore'):return _interpolate(start,end,count,compute).astype(target,copy=False).tobytes()


def linspace(start,end,steps,*,dtype=None,device=None,requires_grad=False,out=None,layout=None,pin_memory=False):
    _options(device,requires_grad,layout,pin_memory,out)
    start,end,count,target=_arguments(start,end,steps,dtype)
    if count==0:return _result(np.empty(0,dtype=target),requires_grad)
    if count==1:return _result(np.array([_endpoint(start,target,fill=True)],dtype=target),requires_grad)
    if target not in _REAL+_COMPLEX:raise _errors.BrowserUnsupported(f'linspace has no CPU/WASM kernel for {target}')
    start,end=_endpoint(start,target),_endpoint(end,target)
    if target.kind=='f' and count<=256 and (start,end) in _UNIT_INTERVALS:
        data=_unit_grid_bytes(target.str,count,_UNIT_INTERVALS.index((start,end)))
        return _result(np.frombuffer(data,dtype=target).copy(),requires_grad)
    compute=np.dtype('float64') if target.kind in 'iu' else np.dtype('float32') if target==np.dtype('float16') else target
    values=_interpolate(start,end,count,compute)
    if compute!=target:
        with np.errstate(all='ignore'):values=values.astype(target)
    return _result(values,requires_grad)


def logspace(start,end,steps,base=10.,*,dtype=None,device=None,requires_grad=False,out=None,layout=None,pin_memory=False):
    _options(device,requires_grad,layout,pin_memory,out)
    if isinstance(base,(str,bytes,np.ndarray)) or isinstance(base,complex) and not isinstance(base,np.complexfloating):raise TypeError('base must be real')
    if isinstance(base,_tensor.Tensor) and base.ndim:raise TypeError('base must be scalar')
    if isinstance(base,_tensor.Tensor):base=_real(base.item())
    base=float(base)
    start,end,count,target=_arguments(start,end,steps,dtype)
    if count==0:return _result(np.empty(0,dtype=target),requires_grad)
    if count==1:
        with np.errstate(all='ignore'):
            value=np.exp(np.log(np.complex128(base))*complex(start)) if target.kind=='c' else np.power(np.float64(base),float(_real(start)))
        return _result(np.array([_endpoint(value.item(),target,fill=True)],dtype=target),requires_grad)
    if target not in _REAL+_COMPLEX:raise _errors.BrowserUnsupported(f'logspace has no CPU/WASM kernel for {target}')
    start,end=_endpoint(start,target),_endpoint(end,target)
    with np.errstate(all='ignore'):
        if target.kind=='c':
            exponent=_interpolate(start,end,count,target,complex_indices=True)
            np.multiply(exponent,np.log(np.asarray(base,dtype=target)),out=exponent)
            np.exp(exponent,out=exponent);values=exponent
        else:
            # Unlike linspace, native real logspace rounds the endpoint
            # difference in the storage type, then computes powers in double.
            difference=end-start if target.kind in 'iu' and target.itemsize<4 else np.subtract(np.asarray(end,dtype=target),np.asarray(start,dtype=target)).item()
            delta=float(difference)/(count-1);half=count//2
            exponent=np.empty(count,dtype=np.float64)
            exponent[:half]=float(start)+np.arange(half,dtype=np.float64)*delta
            exponent[half:]=float(end)-np.arange(count-half-1,-1,-1,dtype=np.float64)*delta
            np.power(np.float64(base),exponent,out=exponent)
            values=exponent.astype(target,copy=False)
    return _result(values,requires_grad)
