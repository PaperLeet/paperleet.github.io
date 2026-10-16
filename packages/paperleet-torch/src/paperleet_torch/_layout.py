"""Shape-only storage views and forward-tangent layout materialization."""
import math
import operator
import numpy as np
from . import _errors


def shape_args(args, count):
    if len(args)==1 and isinstance(args[0],(tuple,list)):args=tuple(args[0])
    elif not args:raise TypeError('view requires a shape')
    shape=[];known=1;unknown=None
    for axis,value in enumerate(args):
        if type(value) is not int:
            if isinstance(value,(bool,np.bool_)):raise TypeError('Shape dimensions must be integers')
            try:value=operator.index(value)
            except TypeError:raise TypeError('Shape dimensions must be integers') from None
        if value < -1:raise RuntimeError('Invalid negative shape dimension')
        if value==-1:
            if unknown is not None:raise RuntimeError('Only one dimension can be inferred')
            unknown=axis
        else:known*=value
        shape.append(value)
    if unknown is not None:
        if not known:raise RuntimeError('Cannot infer a dimension with zero known product')
        if count%known:raise RuntimeError('View shape does not match element count')
        shape[unknown]=count//known
    elif known!=count:raise RuntimeError('View shape does not match element count')
    return tuple(shape)


def view_strides(old_shape, old_stride, shape):
    count=math.prod(old_shape)
    if not count:
        if old_shape==shape:return old_stride
        stride=[0]*len(shape);step=1
        for axis in reversed(range(len(shape))):
            stride[axis]=step;step*=max(shape[axis],1)
        return tuple(stride)
    if not old_shape:return (1,)*len(shape)
    result=[0]*len(shape);vdim=len(shape)-1;base=old_stride[-1];old_count=new_count=1
    for dim in reversed(range(len(old_shape))):
        old_count*=old_shape[dim]
        end=(dim==0 or (old_shape[dim-1]!=1 and old_stride[dim-1]!=old_count*base))
        if not end:continue
        while vdim>=0 and (new_count<old_count or shape[vdim]==1):
            result[vdim]=new_count*base;new_count*=shape[vdim];vdim-=1
        if new_count!=old_count:raise RuntimeError('View crosses non-contiguous storage chunks; use reshape')
        if dim:
            base=old_stride[dim-1];old_count=new_count=1
    if vdim!=-1:raise RuntimeError('View crosses non-contiguous storage chunks; use reshape')
    return tuple(result)


def view(data,*args):
    raw=args[0] if len(args)==1 and isinstance(args[0],(tuple,list)) else args
    # For these geometries NumPy's shape validation cannot trigger a data copy.
    # Nonbuiltin dimensions and invalid negatives keep the checked parser below.
    if args and data.size>1 and all(type(v) is int and v>=-1 for v in raw):
        target=raw if type(raw) is tuple else tuple(raw)
        direct=False
        if data.ndim==1:
            if data.strides[0]%data.itemsize:
                raise _errors.BrowserUnsupported('Byte strides must be whole element strides')
            direct=True
        elif data.flags.c_contiguous and 1 not in data.shape:direct=True
        elif target==data.shape and 1 not in target:
            if any(v%data.itemsize for v in data.strides):
                raise _errors.BrowserUnsupported('Byte strides must be whole element strides')
            return data.view()
        if direct:
            try:return data.reshape(target)
            except ValueError as error:raise RuntimeError(str(error)) from None
    shape=shape_args(args,data.size)
    # These layouts always admit a no-copy C-order regrouping. Reuse NumPy's
    # inexpensive descriptor rather than creating a general stride wrapper.
    if data.size>1:
        if data.ndim==1:
            if data.strides[0]%data.itemsize:
                raise _errors.BrowserUnsupported('Byte strides must be whole element strides')
            return data.reshape(shape)
        if data.flags.c_contiguous and 1 not in data.shape:return data.reshape(shape)
        if shape==data.shape and 1 not in shape:
            element_strides(data)
            return data.view()
    strides=view_strides(data.shape,element_strides(data),shape)
    raw_strides=tuple(v*data.itemsize for v in strides)
    if shape==data.shape and raw_strides==data.strides:return data.view()
    return np.lib.stride_tricks.as_strided(data,shape=shape,
        strides=raw_strides,writeable=data.flags.writeable)


def contiguous_empty(data):
    if data.size:return data
    # A fresh empty factory has conventional C strides; imported empty arrays
    # retain their original metadata instead.
    step=1;values=[]
    for size in reversed(data.shape):values.append(step);step*=max(size,1)
    return np.lib.stride_tricks.as_strided(data,shape=data.shape,
        strides=tuple(v*data.itemsize for v in reversed(values)),writeable=data.flags.writeable)



def arguments(args, kwargs):
    if not kwargs and (len(args)!=1 or not isinstance(args[0],np.dtype)):return args
    options=dict(kwargs)
    if 'dtype' in options or (len(args)==1 and isinstance(args[0],np.dtype)):
        raise _errors.BrowserUnsupported('Dtype-reinterpreting view is unsupported')
    if 'size' in options:
        if args:raise TypeError('view received duplicate size arguments')
        args=(options.pop('size'),)
    if options:raise TypeError('Unexpected view arguments: '+', '.join(sorted(options)))
    return args


def element_strides(data):
    if any(v%data.itemsize for v in data.strides):
        raise _errors.BrowserUnsupported('Byte strides must be whole element strides')
    return tuple(v//data.itemsize for v in data.strides)


def stride(data, dim=None):
    values=element_strides(data)
    if dim is None:return values
    if isinstance(dim,(bool,np.bool_)):raise TypeError('Stride dimension must be an integer')
    dim=operator.index(dim)
    if not -data.ndim<=dim<data.ndim:raise IndexError('Stride dimension out of range')
    return values[dim]


def matching_tangent(tangent, primal):
    """Copy logical values into a nonoverlapping primal layout when required."""
    if tangent._data.strides==primal._data.strides:return tangent
    shape=primal.shape;strides=primal.stride()
    if primal.numel():
        extent=1
        for step,size in sorted((s,n) for n,s in zip(shape,strides) if n>1):
            if step<extent:
                raise _errors.BrowserUnsupported('Materializing a tangent into overlapping storage is unsupported')
            extent+=(size-1)*step
    else:extent=0
    buffer=np.empty(extent,dtype=tangent.dtype)
    data=np.ndarray(shape=shape,dtype=tangent.dtype,buffer=buffer,strides=tuple(v*buffer.itemsize for v in strides))
    data[...]=tangent._data
    from . import _autograd
    return _autograd.result(data,(tangent,lambda g:g),saves_tensors=False)
