"""Dense indexed selection and accumulation with selective analytical tapes."""
import math
import numpy as np
from . import _autograd, _errors, _indexing, _layout, _ops, _promotion, _state, _tensor

_INDICES = (np.dtype('int32'), np.dtype('int64'))
_HALF = np.dtype('float16')
_DTYPES = tuple(np.dtype(name) for name in ('bool','uint8','int8','int16','int32',
                                         'int64','float16','float32','float64','complex64','complex128'))


def _tensor_arg(value,name):
    if not isinstance(value,_tensor.Tensor):raise TypeError(name+' must be a Tensor')
    return value


def _output(out):
    if out is not None:
        _tensor_arg(out,'out')
        raise _errors.BrowserUnsupported('Indexed output buffers are unsupported')


def _dense(value):
    if value.dtype not in _DTYPES:
        raise _errors.BrowserUnsupported('This indexed dtype is unsupported')


def _bounds(index,size,*,scatter=False):
    if _indexing._out_of_bounds(index,size):
        if scatter:raise RuntimeError('scatter_add index out of bounds')
        raise IndexError('Index out of range')


def _vector(index):
    _tensor_arg(index,'index')
    if index.ndim > 1:raise RuntimeError('Index must be a vector or scalar')
    if index.dtype not in _INDICES:raise RuntimeError('Index must have int32 or int64 dtype')
    return index._data.reshape(-1)


def _array(data):return data.reshape(1) if not data.ndim else data


def _take(data,index,axis):
    values=index._data.reshape(-1)
    _bounds(values,data.shape[axis] if data.ndim else 1)
    result=np.take(_array(data),values,axis=axis)
    return result.reshape(()) if not data.ndim else _layout.contiguous_empty(result)


def _alpha(value,dtype):
    if isinstance(value,_tensor.Tensor):
        if value.ndim or value.requires_grad:
            raise TypeError('alpha must be a Number or non-differentiable zero-dimensional Tensor')
        value=value._data.item()
    number,_=_promotion.scalar_info(value)
    original=number
    if dtype.kind=='b':return original,np.asarray(bool(number),dtype=dtype)
    if isinstance(number,complex) and dtype.kind!='c':
        if number.imag:raise RuntimeError('alpha cannot be converted to the input dtype')
        number=number.real
    if dtype.kind in 'iu':
        limits=np.iinfo(dtype)
        if isinstance(number,(float,np.floating)):
            if not math.isfinite(number) or not limits.min<=number<=limits.max:
                raise RuntimeError('alpha cannot be converted without overflow')
            number=int(number)
        elif not (-(limits.max) <= number <= limits.max if dtype.kind=='u' else limits.min<=number<=limits.max):
            raise RuntimeError('alpha cannot be converted without overflow')
        converted=number%(1<<(dtype.itemsize*8)) if dtype.kind=='u' else number
        return original,np.asarray(converted,dtype=dtype)
    limits=np.finfo(dtype)
    parts=(number.real,number.imag) if isinstance(number,complex) else (number,)
    if any(math.isfinite(part) and abs(part)>float(limits.max) for part in parts):
        raise RuntimeError('alpha cannot be converted without overflow')
    with np.errstate(all='ignore'):return original,np.asarray(number,dtype=dtype)


def _index_values(data,index,source,axis,alpha,number=1,*,zero_shape=None):
    dtype=alpha.dtype
    rank=data.ndim if zero_shape is None else len(zero_shape)
    # Native CPU uses the scatter kernel only for these index_add cases.
    # Half fallback paths round after every update; scatter rounds at the end.
    scatter_half=(dtype==_HALF and rank>1 and axis in (0,rank-1)
                  and index.dtype==np.dtype('int64') and number==1)
    accumulator=np.dtype('float32') if scatter_half else dtype
    result=(np.array(data,dtype=accumulator,order='C',copy=True) if zero_shape is None
            else np.zeros(zero_shape,dtype=accumulator))
    left=np.moveaxis(_array(result),axis,0)
    right=np.moveaxis(_array(source),axis,0)
    ids=index._data.reshape(-1)
    if right.size:
        _bounds(ids,left.shape[0])
        with np.errstate(all='ignore'):
            # Half slice-add, like its scalar fallback, rounds the scaled
            # source before each half addition. Only scatter_half widens.
            row_bytes=max(1,int(np.prod(right.shape[1:],dtype=np.int64))*accumulator.itemsize)
            step=max(1,(1024*1024)//row_bytes)
            for start in range(0,len(ids),step):
                values=right[start:start+step].astype(accumulator,copy=False)
                if alpha!=1:values=values*alpha.astype(accumulator,copy=False)
                np.add.at(left,ids[start:start+step],values)
    return _layout.contiguous_empty(result.astype(dtype,copy=False))


def _scatter_values(data,index,source,axis):
    dtype=data.dtype;accumulator=np.dtype('float32') if dtype==_HALF else dtype
    result=np.array(data,dtype=accumulator,order='C',copy=True)
    if not index.numel():return _layout.contiguous_empty(result.astype(dtype,copy=False))
    target=_array(result);indices=_array(index._data);values=_array(source)
    _bounds(indices,target.shape[axis],scatter=True)
    selected=values[tuple(slice(0,size) for size in indices.shape)]
    coordinates=tuple(indices if dimension==axis else
                      np.arange(size,dtype=np.intp).reshape((1,)*dimension+(size,)+(1,)*(indices.ndim-dimension-1))
                      for dimension,size in enumerate(indices.shape))
    with np.errstate(all='ignore'):np.add.at(target,coordinates,selected.astype(accumulator,copy=False))
    return _layout.contiguous_empty(result.astype(dtype,copy=False))


def _gradient_scale(values,number,dtype):
    if isinstance(number,complex) and dtype.kind!='c':
        raise RuntimeError('Complex alpha has no real source gradient')
    if number==1:return values
    with np.errstate(all='ignore'):
        if dtype==_HALF:return (values.astype(np.float32)*np.float32(number)).astype(dtype)
        return values*np.asarray(number,dtype=dtype)


def _record(value,inputs,backward):
    context=_state._tensor_reverse_context
    return context.record(value,inputs,backward) if context is not None else value


def index_select(input,dim,index,*,out=None):
    _tensor_arg(input,'input');_output(out);_dense(input)
    axis=_layout.dimension(dim,input.ndim);indices=_vector(index)
    if not input.ndim and indices.size!=1:raise RuntimeError('Index to a scalar must contain one value')
    _bounds(indices,input.shape[axis] if input.ndim else 1)
    if isinstance(input,_autograd.ForwardTensor):
        return _autograd.ForwardTensor(index_select(input._primal,axis,index),
                                      index_select(input._tangent,axis,index))
    shape,dtype=input.shape,input.dtype
    values=_take(input._data,index,axis)
    def backward(seed):return _index_values(None,index,seed,axis,np.asarray(1,dtype=dtype),zero_shape=shape)
    result=_autograd.result(values,(input,backward),versioned=(index,))
    def tensor_backward(seed):
        return (index_add(_tensor.Tensor(np.zeros(shape,dtype=dtype)),axis,index,seed),)
    return _record(result,(input,),tensor_backward)


def index_add(input,dim,index,source,*,alpha=1,out=None):
    _tensor_arg(input,'input');_tensor_arg(source,'source');_output(out);_dense(input)
    axis=_layout.dimension(dim,input.ndim);indices=_vector(index)
    if input.dtype!=source.dtype:raise RuntimeError('index_add input and source dtypes must match')
    if source.ndim!=input.ndim:raise RuntimeError('index_add source rank must match input')
    source_shape=source.shape or (1,);input_shape=input.shape or (1,)
    if source_shape[axis]!=indices.size or any(a!=b for dimension,(a,b) in enumerate(zip(input_shape,source_shape)) if dimension!=axis):
        raise RuntimeError('index_add source shape does not match index and input')
    if source.numel():_bounds(indices,input_shape[axis])
    number,coefficient=_alpha(alpha,input.dtype)
    if isinstance(input,_autograd.ForwardTensor) or isinstance(source,_autograd.ForwardTensor):
        a=input._primal if isinstance(input,_autograd.ForwardTensor) else input
        b=source._primal if isinstance(source,_autograd.ForwardTensor) else source
        da=input._tangent if isinstance(input,_autograd.ForwardTensor) else _tensor.Tensor(np.zeros(input.shape,dtype=input.dtype))
        db=source._tangent if isinstance(source,_autograd.ForwardTensor) else _tensor.Tensor(np.zeros(source.shape,dtype=source.dtype))
        value=index_add(a,axis,index,b,alpha=number)
        tangent=index_add(da,axis,index,db,alpha=number)
        return _autograd.ForwardTensor(value,tangent)
    dtype=source.dtype
    values=_index_values(input._data,index,source._data,axis,coefficient,number)
    def source_backward(seed):return _gradient_scale(_take(seed,index,axis),number,dtype)
    versioned=(index,source) if source.requires_grad else ()
    result=_autograd.result(values,(input,lambda g:g),(source,source_backward),versioned=versioned)
    needs_input,needs_source=input.requires_grad,source.requires_grad
    def tensor_backward(seed):
        if needs_source and isinstance(number,complex) and dtype.kind!='c':
            raise RuntimeError('Complex alpha has no real source gradient')
        return (seed if needs_input else None,
                index_select(seed,axis,index)*number if needs_source else None)
    return _record(result,(input,source),tensor_backward)


def _scatter_take(input,axis,index):
    result=_indexing.gather(input,axis,index)
    shape,dtype=input.shape,input.dtype
    def backward(seed):return (scatter_add(_tensor.Tensor(np.zeros(shape,dtype=dtype)),axis,index,seed),)
    return _record(result,(input,),backward)


def scatter_add(input,dim,index,src,*,out=None):
    _tensor_arg(input,'input');_tensor_arg(src,'src');_tensor_arg(index,'index');_output(out);_dense(input)
    axis=_layout.dimension(dim,input.ndim)
    if input.dtype!=src.dtype:raise RuntimeError('scatter_add input and source dtypes must match')
    if index.numel():
        if index.dtype not in _INDICES:raise RuntimeError('scatter_add index dtype must be int32 or int64')
        shapes=[value.shape or (1,) for value in (input,index,src)]
        if len(set(map(len,shapes)))!=1:raise RuntimeError('scatter_add ranks must match')
        if any(n>s or dimension!=axis and n>a for dimension,(a,n,s) in enumerate(zip(*shapes))):
            raise RuntimeError('scatter_add index dimensions exceed input or source')
        _bounds(index._data,shapes[0][axis],scatter=True)
    if isinstance(input,_autograd.ForwardTensor) or isinstance(src,_autograd.ForwardTensor):
        a=input._primal if isinstance(input,_autograd.ForwardTensor) else input
        b=src._primal if isinstance(src,_autograd.ForwardTensor) else src
        da=input._tangent if isinstance(input,_autograd.ForwardTensor) else _tensor.Tensor(np.zeros(input.shape,dtype=input.dtype))
        db=src._tangent if isinstance(src,_autograd.ForwardTensor) else _tensor.Tensor(np.zeros(src.shape,dtype=src.dtype))
        return _autograd.ForwardTensor(scatter_add(a,axis,index,b),scatter_add(da,axis,index,db))
    source_shape=src.shape
    values=_scatter_values(input._data,index,src._data,axis)
    def checked_shape():
        try:valid=np.broadcast_shapes(source_shape,index.shape)==index.shape
        except ValueError:valid=False
        if not valid:raise RuntimeError('scatter_add backward cannot reduce to source shape')
    def source_backward(seed):
        checked_shape()
        if index.numel():
            _bounds(index._data,seed.shape[axis] if seed.ndim else 1,scatter=True)
            gradient=_indexing._take(seed,index._data,axis)
        else:gradient=np.empty(index.shape,dtype=seed.dtype)
        return _ops.unbroadcast(gradient,source_shape)
    needs_input,needs_source=input.requires_grad,src.requires_grad
    result=_autograd.result(values,(input,lambda g:g),(src,source_backward),versioned=(index,) if needs_source else ())
    def tensor_backward(seed):
        source_gradient=None
        if needs_source:
            checked_shape()
            source_gradient=_scatter_take(seed,axis,index)
            while source_gradient.ndim>len(source_shape):source_gradient=source_gradient.sum(dim=0)
            for dimension,size in enumerate(source_shape):
                if size==1 and source_gradient.shape[dimension]!=1:
                    source_gradient=source_gradient.sum(dim=dimension,keepdim=True)
            source_gradient=source_gradient.reshape(source_shape)
        return seed if needs_input else None,source_gradient
    return _record(result,(input,src),tensor_backward)


def inplace(input,dim,index,source,*,alpha=1,scatter=False):
    _tensor_arg(source,'source');_tensor_arg(index,'index')
    if _state._jvp_active:raise _errors.BrowserUnsupported('Indexed mutation during forward AD is unsupported')
    if _state._grad_enabled and (input.requires_grad or source.requires_grad):
        raise _errors.BrowserUnsupported('Differentiable indexed mutation is unsupported')
    if input.requires_grad and input._parents:
        raise _errors.BrowserUnsupported('Indexed updates of nonleaf tensors are unsupported')
    if any(size>1 and step==0 for size,step in zip(input.shape,input.stride())):
        raise RuntimeError('Indexed destination has overlapping elements')
    if input.numel():
        for value in (source,index):
            try:overlap=np.shares_memory(input._data,value._data,max_work=1000)
            except np.exceptions.TooHardError:
                raise _errors.BrowserUnsupported('Indexed overlap cannot be determined within its budget') from None
            if overlap:raise RuntimeError('Indexed source or index overlaps its destination')
    result=scatter_add(input,dim,index,source) if scatter else index_add(input,dim,index,source,alpha=alpha)
    input.copy_(result)
    return input
