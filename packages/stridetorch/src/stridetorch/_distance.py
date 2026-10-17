"""Direct Minkowski distances with bounded tiles and shared first-order VJPs."""
import math
import weakref

import numpy as np

from . import _autograd, _errors, _layout, _promotion, _tensor

_REAL = (np.dtype('float32'),np.dtype('float64'))
_WORKSPACE_BYTES = 8 * 1024 * 1024
_OUTPUT_LIMIT = 16_000_000
_WORK_LIMIT = 512_000_000
_MODES = ('use_mm_for_euclid_dist_if_necessary','use_mm_for_euclid_dist','donot_use_mm_for_euclid_dist')


def _order(value):
    if isinstance(value,_tensor.Tensor):
        if value.ndim or value.requires_grad:raise TypeError('Distance order must be a scalar Tensor without gradients')
        value=value.item()
        if isinstance(value,complex):
            if value.imag:raise RuntimeError('Distance order cannot contain an imaginary component')
            value=value.real
    elif not _promotion.is_scalar(value):raise TypeError('Distance order must be a real scalar')
    value=float(value)
    if math.isnan(value) or value<0:raise RuntimeError('Distance order must be nonnegative')
    return value


def _check(input):
    if not isinstance(input,_tensor.Tensor):raise TypeError('Distances expect Tensor inputs')
    if input.dtype not in _REAL:raise _errors.BrowserUnsupported('Distances require real float32/float64')


def _budget(pairs,features):
    if pairs>_OUTPUT_LIMIT or pairs*features>_WORK_LIMIT:
        raise _errors.BrowserUnsupported('Pairwise distance output/work exceeds the browser budget')


def _plan(batch,rows,columns,features,dtype,triangular=False):
    width=min(max(features,1),4096,max(1,(_WORKSPACE_BYTES-64)//(8*dtype.itemsize)))
    pairs=max(1,_WORKSPACE_BYTES//(8*width*dtype.itemsize+64))
    batches=min(max(batch,1),8,pairs)
    side=max(1,math.isqrt(pairs//batches))
    left=min(max(rows,1),side)
    right=min(max(columns,1),left if triangular else max(1,pairs//batches//left))
    return batches,left,right,width


def _blocks(batch,rows,columns,plan,triangular=False):
    block_batch,left,right,_=plan
    for b in range(0,batch,block_batch):
        for i in range(0,rows,left):
            for j in range(i if triangular else 0,columns,right):
                yield slice(b,min(b+block_batch,batch)),slice(i,min(i+left,rows)),slice(j,min(j+right,columns))


def _coordinates(block,batch_shape):
    return np.unravel_index(np.arange(block.start,block.stop),batch_shape) if batch_shape else ()


def _source(view,coords,points,features):
    return view[(*coords,points,features)] if coords else view[points,features][None]


def _norm_chunk(difference,p,*,triangular=False):
    np.abs(difference,out=difference)
    if p==0:
        np.ceil(difference,out=difference);np.minimum(difference,1,out=difference)
    elif p==2:np.square(difference,out=difference)
    elif math.isinf(p):
        return (np.maximum if triangular else np.fmax).reduce(difference,axis=-1,initial=0)
    elif p!=1:np.power(difference,p,out=difference)
    return difference.sum(axis=-1)


def _finish(value,p):
    if p==2:np.sqrt(value,out=value)
    elif p not in (0,1) and not math.isinf(p):
        # Native scalar finish computes the reciprocal/power in double.
        value=np.power(value.astype('float64'),1./p).astype(value.dtype)
    return value


def _direction(difference,lengths,seed,p):
    if p==1:
        np.sign(difference,out=difference);np.multiply(difference,seed[...,None],out=difference)
        return difference
    if p==2:
        np.multiply(difference,seed[...,None],out=difference)
        np.divide(difference,lengths[...,None],out=difference,where=lengths[...,None]!=0)
    elif math.isinf(p):
        # Preserve native IEEE behavior at infinite ties and NaN components.
        weight=np.abs(difference);np.subtract(weight,lengths[...,None],out=weight)
        np.abs(weight,out=weight);np.ceil(weight,out=weight);np.minimum(weight,1,out=weight)
        np.subtract(1,weight,out=weight);np.sign(difference,out=difference)
        np.multiply(difference,seed[...,None],out=difference);np.multiply(difference,weight,out=difference)
        return difference
    else:
        scalar=np.asarray(p,dtype=difference.dtype)
        if p<2:
            zero=difference==0 if p<1 else None
            weight=np.abs(difference);np.power(weight,scalar-1,out=weight)
            np.sign(difference,out=difference);np.multiply(difference,weight,out=difference)
        else:
            zero=None;weight=np.abs(difference);np.power(weight,scalar-2,out=weight)
            np.multiply(difference,weight,out=difference)
        np.multiply(difference,seed[...,None],out=difference)
        denominator=np.power(lengths,scalar-1)
        np.divide(difference,denominator[...,None],out=difference,where=lengths[...,None]!=0)
        if zero is not None:np.copyto(difference,0,where=zero)
    np.copyto(difference,0,where=lengths[...,None]==0)
    return difference


def _add_gradient(gradient,coords,points,features,value,batch_shape):
    shape=gradient.shape[:-2]
    if not shape:
        gradient[points,features]+=value.sum(axis=0)
    else:
        mapped=tuple(np.zeros_like(index) if size==1 else index for index,size in zip(coords[-len(shape):],shape))
        np.add.at(gradient,(*mapped,points,features),value)


def _unbroadcast(value,shape):
    while value.ndim>len(shape):value=value.sum(axis=0)
    for axis,size in enumerate(shape):
        if size==1 and value.shape[axis]!=1:value=value.sum(axis=axis,keepdims=True)
    return value.reshape(shape)


def _small_cdist(x1,x2,p):
    first,second=x1._data,x2._data;shapes=(x1.shape,x2.shape)
    with np.errstate(all='ignore'):
        kernel_p=np.asarray(p,dtype=x1.dtype).item()
        lengths=_finish(_norm_chunk(first[..., :,None,:]-second[...,None,:,:],kernel_p),kernel_p)
    def backward(seed,needs):
        if p==0 or not lengths.size or not shapes[0][-1]:
            return [np.zeros(shape,dtype=first.dtype) if active else None for shape,active in zip(shapes,needs)]
        unit=_direction(first[..., :,None,:]-second[...,None,:,:],lengths,seed,p)
        return [_unbroadcast(unit.sum(axis=-2),shapes[0]) if needs[0] else None,
                _unbroadcast(-unit.sum(axis=-3),shapes[1]) if needs[1] else None]
    if not lengths.size:lengths=_layout.contiguous_empty(lengths)
    return _autograd.joint_result(lengths,(x1,x2),backward)


def _small_pdist(input,p):
    source=input._data;dtype=input.dtype;i,j=np.triu_indices(input.shape[0],k=1);owner=weakref.ref(input)
    with np.errstate(all='ignore'):
        kernel_p=np.asarray(p,dtype=dtype).item()
        lengths=_finish(_norm_chunk(source[i]-source[j],kernel_p,triangular=True),kernel_p)
    def backward(seed):
        current=owner()
        if current is not None and current.dtype!=dtype:raise RuntimeError('pdist backward input and gradient dtypes must match')
        gradient=np.zeros(source.shape,dtype=dtype)
        if p!=0 and lengths.size and source.shape[-1]:
            unit=_direction(source[i]-source[j],lengths,seed,p)
            np.add.at(gradient,i,unit);np.add.at(gradient,j,-unit)
        return gradient
    if not lengths.size:lengths=_layout.contiguous_empty(lengths)
    return _autograd.result(lengths,(input,backward))


def _run(x1,x2,p,*,triangular=False):
    dtype=x1.dtype;features=x1.shape[-1];rows,columns=x1.shape[-2],x2.shape[-2]
    batch_shape=() if triangular else np.broadcast_shapes(x1.shape[:-2],x2.shape[:-2]);batch=math.prod(batch_shape)
    pairs=rows*(rows-1)//2 if triangular else batch*rows*columns
    _budget(pairs,features)
    if pairs*(8*features*dtype.itemsize+64)<=_WORKSPACE_BYTES:
        if not triangular:return _small_cdist(x1,x2,p)
        if rows<=16:return _small_pdist(x1,p)
    sources=(x1._data,x2._data);shapes=(x1.shape,x2.shape)
    views=tuple(np.broadcast_to(data,(*batch_shape,*data.shape[-2:])) for data in sources)
    shape=(pairs,) if triangular else (*batch_shape,rows,columns)
    output=np.zeros(shape,dtype=dtype)
    flat=output if triangular else output.reshape(batch,rows,columns)
    plan=_plan(batch,rows,columns,features,dtype,triangular)
    with np.errstate(all='ignore'):
        kernel_p=np.asarray(p,dtype=dtype).item()
        if features:
            for b,i,j in _blocks(batch,rows,columns,plan,triangular):
                coords=_coordinates(b,batch_shape);value=np.zeros((b.stop-b.start,i.stop-i.start,j.stop-j.start),dtype=dtype)
                for f in range(0,features,plan[-1]):
                    span=slice(f,min(f+plan[-1],features))
                    delta=_source(views[0],coords,i,span)[:,:,None,:]-_source(views[1],coords,j,span)[:,None,:,:]
                    part=_norm_chunk(delta,kernel_p,triangular=triangular)
                    if math.isinf(kernel_p):(np.maximum if triangular else np.fmax)(value,part,out=value)
                    else:np.add(value,part,out=value)
                    del delta,part
                value=_finish(value,kernel_p)
                if triangular:
                    ii=np.arange(i.start,i.stop)[:,None];jj=np.arange(j.start,j.stop)[None,:];mask=jj>ii
                    index=ii*(2*rows-ii-1)//2+jj-ii-1
                    output[index[mask]]=value[0][mask]
                else:flat[b,i,j]=value
    owner=weakref.ref(x1) if triangular else None
    def backward(seed,needs):
        current=owner() if triangular else None
        if current is not None and current.dtype!=dtype:
            raise RuntimeError('pdist backward input and gradient dtypes must match')
        if triangular:
            gradient=np.zeros(shapes[0],dtype=dtype) if needs[0] else None
            gradients=[gradient,gradient]
        else:gradients=[np.zeros(s,dtype=dtype) if active else None for s,active in zip(shapes,needs)]
        if not features or not pairs or p==0:return gradients
        upstream=seed if triangular else np.asarray(seed).reshape(batch,rows,columns)
        for b,i,j in _blocks(batch,rows,columns,plan,triangular):
            coords=_coordinates(b,batch_shape);mask=None
            if triangular:
                ii=np.arange(i.start,i.stop)[:,None];jj=np.arange(j.start,j.stop)[None,:];mask=jj>ii
                index=ii*(2*rows-ii-1)//2+jj-ii-1
                distances=np.zeros(mask.shape,dtype=dtype);g=np.zeros_like(distances)
                distances[mask]=output[index[mask]];g[mask]=upstream[index[mask]]
                distances,g=distances[None],g[None]
            else:distances,g=flat[b,i,j],upstream[b,i,j]
            for f in range(0,features,plan[-1]):
                span=slice(f,min(f+plan[-1],features))
                delta=_source(views[0],coords,i,span)[:,:,None,:]-_source(views[1],coords,j,span)[:,None,:,:]
                direction=_direction(delta,distances,g,p)
                if mask is not None:np.copyto(direction,0,where=~mask[None,:,:,None])
                if gradients[0] is not None:_add_gradient(gradients[0],coords,i,span,direction.sum(axis=-2),batch_shape)
                if gradients[1] is not None:_add_gradient(gradients[1],coords,j,span,-direction.sum(axis=-3),batch_shape)
                del delta,direction
        return gradients
    if not output.size:output=_layout.contiguous_empty(output)
    if triangular:
        def triangular_backward(seed,needs):
            first,second=backward(seed,(needs[0],needs[0]))
            return [first] if needs[0] else [None]
        return _autograd.joint_result(output,(x1,),triangular_backward)
    return _autograd.joint_result(output,(x1,x2),backward)


def cdist(x1,x2,p=2.,compute_mode='use_mm_for_euclid_dist_if_necessary'):
    _check(x1);_check(x2);p=_order(p)
    if compute_mode not in _MODES:raise _errors.BrowserUnsupported('Unknown cdist computation mode')
    if x1.ndim<2 or x2.ndim<2 or x1.shape[-1]!=x2.shape[-1] or x1.dtype!=x2.dtype:
        raise RuntimeError('cdist expects matching floating feature dimensions and dtypes')
    try:np.broadcast_shapes(x1.shape[:-2],x2.shape[:-2])
    except ValueError as error:raise RuntimeError('cdist batch dimensions are not broadcastable') from error
    return _run(x1,x2,p)


def pdist(input,p=2.):
    _check(input);p=_order(p)
    if input.ndim!=2:raise RuntimeError('pdist expects a two-dimensional Tensor')
    rows,features=input.shape
    if rows<=16 and rows*(rows-1)//2*(8*features*input.dtype.itemsize+64)<=_WORKSPACE_BYTES:
        _budget(rows*(rows-1)//2,features)
        return _small_pdist(input,p)
    return _run(input,input,p,triangular=True)
