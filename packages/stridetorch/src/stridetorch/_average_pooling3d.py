"""Ordered volumetric averages and linear adjoints with bounded packing."""
from collections import namedtuple
from functools import lru_cache
import itertools
import math
import numbers
import numpy as np
from . import _autograd, _average_pooling, _errors, _layout, _pooling3d, _tensor
from ._module import Module

_ALGORITHM = 'auto'  # Private numerical-route comparison, not an API option.
_BACKWARD_ALGORITHM = 'auto'
_WORKSPACE_BYTES = 32 * 1024 * 1024
_Geometry = namedtuple('_Geometry', 'start end out divisor adaptive partition pack', defaults=(None,))


def _input(input, adaptive, allow_empty=False):
    if not isinstance(input, _tensor.Tensor): raise TypeError('Average pooling input must be a Tensor')
    if input.ndim not in (4,5): raise RuntimeError('3D average pooling expects CDHW/NCDHW input')
    if not allow_empty and any(n==0 for n in input.shape[-4:]):
        raise RuntimeError('3D average pooling requires nonempty channels/spatial axes')
    supported = _average_pooling._FLOATS if adaptive else (np.dtype('float32'),np.dtype('float64'),np.dtype('int64'))
    if input.dtype not in supported: raise _errors.BrowserUnsupported('This average-pooling dtype requires the server')


def _allocation(shape,out):
    if max(out)>4_000_000 or math.prod(out)*math.prod(max(1,n) for n in shape[:-3])>8_000_000:
        raise _errors.BrowserUnsupported('This average-pooling output requires the server')
    if math.prod(shape[-3:])>np.iinfo(np.intp).max:
        raise _errors.BrowserUnsupported('These average-pooling spatial indices require the server')


def _output_size(value,spatial):
    if isinstance(value,numbers.Integral) and not isinstance(value,(bool,np.bool_)):value=(value,)*3
    if not isinstance(value,(tuple,list)):raise TypeError('output_size must be an integer or triple')
    if len(value)!=3:raise RuntimeError('output_size must contain three dimensions')
    result=[]
    for n,current in zip(value,spatial):
        if n is None:n=current
        if not isinstance(n,numbers.Integral) or isinstance(n,(bool,np.bool_)):raise TypeError('output_size entries must be integers or None')
        if n<0:raise RuntimeError('output_size entries must be nonnegative')
        result.append(int(n))
    return tuple(result)


def _geometry(shape,out,kernel=None,stride=None,padding=None,include=True,override=None):
    if sum(out)<=2048:
        return _cached_geometry(shape,out,kernel,stride,padding,include,override,_WORKSPACE_BYTES)
    return _make_geometry(shape,out,kernel,stride,padding,include,override,_WORKSPACE_BYTES)


@lru_cache(maxsize=32)
def _cached_geometry(shape,out,kernel,stride,padding,include,override,budget):
    return _make_geometry(shape,out,kernel,stride,padding,include,override,budget)


def _make_geometry(shape,out,kernel,stride,padding,include,override,budget):
    adaptive=kernel is None;start=[];end=[];counts=[]
    # Empty batches/outputs need only shape metadata, not coordinate grids.
    if not math.prod(shape[:-3]) or not math.prod(out):return _Geometry((),(),out,override,adaptive,None)
    for axis,(size,count) in enumerate(zip(shape[-3:],out)):
        index=np.arange(count,dtype=np.int64)
        if adaptive:
            first=index*size//count;last=((index+1)*size+count-1)//count
        else:
            raw=index*stride[axis]-padding[axis];stop=np.minimum(raw+kernel[axis],size+padding[axis])
            first=np.maximum(raw,0);last=np.minimum(stop,size)
            counts.append(stop-raw if include else last-first)
        start.append(first);end.append(last)
    partition=(tuple(size//n for size,n in zip(shape[-3:],out)) if adaptive and all(size%n==0 for size,n in zip(shape[-3:],out))
               else kernel if not adaptive and stride==kernel and padding==(0,0,0) and tuple(n*k for n,k in zip(out,kernel))==shape[-3:] else None)
    divisor=None if adaptive else override if override is not None else tuple(counts)
    for array in (*start,*end,*counts):array.setflags(write=False)
    geometry=_Geometry(tuple(start),tuple(end),out,divisor,adaptive,partition)
    return geometry._replace(pack=_pack_plan(shape[-3:],geometry,budget))


def _pack_plan(spatial,g,budget):
    """At most 128 KiB per readonly plan; never retain input values/storage."""
    taps=tuple(int((b-a).max()) for a,b in zip(g.start,g.end))
    points=math.prod(g.out)*math.prod(taps)
    if points*(np.dtype(np.intp).itemsize+1)>min(128*1024,budget//4):return None
    coords=[a[:,None]+np.arange(k,dtype=np.intp) for a,k in zip(g.start,taps)]
    valid=[c<b[:,None] for c,b in zip(coords,g.end)]
    coords=[np.minimum(c,n-1) for c,n in zip(coords,spatial)]
    z=coords[0][:,None,None,:,None,None];y=coords[1][None,:,None,None,:,None];x=coords[2][None,None,:,None,None,:]
    flat=(((z*spatial[1])+y)*spatial[2]+x).astype(np.intp).reshape(*g.out,math.prod(taps))
    mask=(valid[0][:,None,None,:,None,None]&valid[1][None,:,None,None,:,None]&valid[2][None,None,:,None,None,:]).reshape(flat.shape)
    flat.setflags(write=False);mask.setflags(write=False)
    return flat,mask


def _divide(value,g):
    if g.adaptive:
        for axis,(a,b) in enumerate(zip(g.start,g.end)):
            shape=[1]*3;shape[axis]=len(a)
            np.divide(value,(b-a).astype(value.dtype).reshape(shape),out=value)
        return value
    divisor=g.divisor
    if isinstance(divisor,tuple):divisor=divisor[0][:,None,None]*divisor[1][None,:,None]*divisor[2]
    if value.dtype.kind=='i':return _average_pooling._divide(value,(None,None,None,None,divisor,False))
    np.divide(value,np.asarray(divisor,dtype=value.dtype),out=value);return value


def _subgeometry(g,axis,region):
    start=list(g.start);end=list(g.end);out=list(g.out)
    start[axis]=start[axis][region];end[axis]=end[axis][region];out[axis]=len(start[axis])
    divisor=g.divisor
    if isinstance(divisor,tuple):
        divisor=list(divisor);divisor[axis]=divisor[axis][region];divisor=tuple(divisor)
    return _Geometry(tuple(start),tuple(end),tuple(out),divisor,g.adaptive,None)


def _scan(data,g):
    value=np.zeros((*data.shape[:-3],*g.out),dtype=_average_pooling._accumulator(data.dtype))
    taps=tuple(int((b-a).max()) for a,b in zip(g.start,g.end))
    for z,y,x in itertools.product(*(range(n) for n in taps)):
        coords=[a+offset for a,offset in zip(g.start,(z,y,x))]
        valid=[c<b for c,b in zip(coords,g.end)]
        source=data[...,np.minimum(coords[0],data.shape[-3]-1)[:,None,None],
                    np.minimum(coords[1],data.shape[-2]-1)[None,:,None],np.minimum(coords[2],data.shape[-1]-1)]
        mask=valid[0][:,None,None]&valid[1][None,:,None]&valid[2]
        np.add(value,source,out=value,where=mask)
    return _divide(value,g).astype(data.dtype,copy=False)


def _packing_bytes(data,g):
    taps=math.prod(int((b-a).max()) for a,b in zip(g.start,g.end))
    points=math.prod(data.shape[:-3])*math.prod(g.out);acc=_average_pooling._accumulator(data.dtype)
    return points*(taps*(data.itemsize+acc.itemsize+2)+64)+sum(g.out)*taps*24


def _packed(data,g):
    if g.pack is not None and data.flags.c_contiguous:
        indices,mask=g.pack
        packed=np.take(data.reshape(*data.shape[:-3],math.prod(data.shape[-3:])),indices,axis=-1)
        packed=packed.astype(_average_pooling._accumulator(data.dtype),copy=False)
        np.copyto(packed,0,where=~mask)
        np.add.accumulate(packed,axis=-1,out=packed)
        return _divide(np.add(packed[...,-1],0),g).astype(data.dtype,copy=False)
    lengths=[b-a for a,b in zip(g.start,g.end)];taps=tuple(int(n.max()) for n in lengths)
    coords=[a[:,None]+np.arange(k) for a,k in zip(g.start,taps)]
    valid=[c<b[:,None] for c,b in zip(coords,g.end)]
    coords=[np.minimum(c,n-1) for c,n in zip(coords,data.shape[-3:])]
    z=coords[0][:,None,None,:,None,None];y=coords[1][None,:,None,None,:,None];x=coords[2][None,None,:,None,None,:]
    packed=np.array(data[...,z,y,x],dtype=_average_pooling._accumulator(data.dtype),copy=True,order='C')
    mask=valid[0][:,None,None,:,None,None]&valid[1][None,:,None,None,:,None]&valid[2][None,None,:,None,None,:]
    np.copyto(packed,0,where=~mask)
    packed=packed.reshape(*data.shape[:-3],*g.out,math.prod(taps));np.add.accumulate(packed,axis=-1,out=packed)
    value=np.add(packed[...,-1],0);return _divide(value,g).astype(data.dtype,copy=False)


def _stream(data,g):
    region=tuple(slice(int(a[0]),int(b[0])) for a,b in zip(g.start,g.end));flat=data[region].flat
    acc=_average_pooling._accumulator(data.dtype);total=acc.type(0)
    count=max(1,(_WORKSPACE_BYTES-acc.itemsize)//(data.itemsize+acc.itemsize))
    for first in range(0,len(flat),count):
        buffer=np.empty(min(count,len(flat)-first)+1,dtype=acc);buffer[0]=total;buffer[1:]=flat[first:first+count]
        np.add.accumulate(buffer,out=buffer);total=buffer[-1]
    return _divide(np.asarray(total).reshape(1,1,1),g).astype(data.dtype,copy=False)


def _forward(data,g):
    if not data.size or not math.prod(g.out):return np.empty((*data.shape[:-3],*g.out),dtype=data.dtype)
    if _ALGORITHM not in ('auto','scan','packed','block'):raise ValueError('Unknown internal 3D average algorithm')
    if g.partition is not None and math.prod(g.partition)<=8 and _ALGORITHM in ('auto','block'):
        value=np.zeros((*data.shape[:-3],*g.out),dtype=_average_pooling._accumulator(data.dtype))
        for offsets in itertools.product(*(range(k) for k in g.partition)):
            region=tuple(slice(o,n*k,k) for o,n,k in zip(offsets,g.out,g.partition))
            np.add(value,data[(Ellipsis,)+region],out=value)
        return _divide(value,g).astype(data.dtype,copy=False)
    estimate=_packing_bytes(data,g)
    if estimate<=_WORKSPACE_BYTES:return _scan(data,g) if _ALGORITHM=='scan' else _packed(data,g)
    # Tile both packed and scan routes; a single region streams in input order.
    value=np.empty((*data.shape[:-3],*g.out),dtype=data.dtype)
    if data.ndim>3:
        for i in range(data.shape[0]):value[i]=_forward(data[i],g)
    elif max(g.out)>1:
        axis=next(i for i,n in enumerate(g.out) if n>1);mid=g.out[axis]//2
        for region in (slice(None,mid),slice(mid,None)):
            key=[slice(None)]*3;key[axis]=region;value[tuple(key)]=_forward(data,_subgeometry(g,axis,region))
    else:return _stream(data,g)
    return value


def _backward(seed,shape,g):
    if not math.prod(g.out):raise RuntimeError('Average-pooling backward requires nonempty output spatial axes')
    result=np.zeros(shape,dtype=seed.dtype)
    if not seed.size:return result
    points=math.prod(g.out);planes=math.prod(shape[:-3]);taps=math.prod(int((b-a).max()) for a,b in zip(g.start,g.end))
    if _BACKWARD_ALGORITHM not in ('auto','loop','scatter','block'):raise ValueError('Unknown internal 3D average backward algorithm')
    if g.partition is not None and _BACKWARD_ALGORITHM in ('auto','block') and seed.nbytes<=_WORKSPACE_BYTES:
        delta=_divide(np.array(seed,copy=True,order='C'),g);kd,kh,kw=g.partition
        view=result.reshape(*shape[:-3],g.out[0],kd,g.out[1],kh,g.out[2],kw)
        np.add(delta[..., :,None,:,None,:,None],0,out=view);return result
    if _BACKWARD_ALGORITHM=='loop' or points<=16 or taps*96>_WORKSPACE_BYTES:
        for z,y,x in np.ndindex(g.out):
            sub=_Geometry(tuple(a[i:i+1] for a,i in zip(g.start,(z,y,x))),tuple(b[i:i+1] for b,i in zip(g.end,(z,y,x))),(1,1,1),
                          tuple(c[i:i+1] for c,i in zip(g.divisor,(z,y,x))) if isinstance(g.divisor,tuple) else g.divisor,g.adaptive,None)
            delta=_divide(seed[...,z:z+1,y:y+1,x:x+1].copy(),sub)
            region=tuple(slice(int(a[0]),int(b[0])) for a,b in zip(sub.start,sub.end))
            result[(Ellipsis,)+region]+=delta
        return result
    # Flatten only an existing C-order seed view. A strided/F-order seed can
    # exceed the packing budget; gather its plane/output blocks directly.
    flat=seed.reshape(planes,points) if seed.flags.c_contiguous else None
    target=result.reshape(planes,math.prod(shape[-3:]));widths=[int((b-a).max()) for a,b in zip(g.start,g.end)]
    offsets=np.arange(taps,dtype=np.int64);oz=offsets//(widths[1]*widths[2]);oy=(offsets//widths[2])%widths[1];ox=offsets%widths[2]
    batch=max(1,min(planes,_WORKSPACE_BYTES//max(1,taps*96)))
    for first in range(0,planes,batch):
        last=min(first+batch,planes);block=max(1,min(points,_WORKSPACE_BYTES//max(1,(last-first)*taps*96)))
        for begin in range(0,points,block):
            stop=min(begin+block,points);positions=np.arange(begin,stop);z=positions//(g.out[1]*g.out[2]);y=(positions//g.out[2])%g.out[1];x=positions%g.out[2]
            coords=[a[p,None]+o for a,p,o in zip(g.start,(z,y,x),(oz,oy,ox))];valid=np.ones(coords[0].shape,dtype=bool)
            for c,b,p in zip(coords,g.end,(z,y,x)):valid&=c<b[p,None]
            if flat is not None:delta=flat[first:last,begin:stop].copy()
            else:
                leading=np.unravel_index(np.arange(first,last),shape[:-3])
                delta=seed[tuple(c[:,None] for c in leading)+(z[None,:],y[None,:],x[None,:])]
            if g.adaptive:
                for a,b,p in zip(g.start,g.end,(z,y,x)):np.divide(delta,(b[p]-a[p]).astype(seed.dtype),out=delta)
            else:
                divisor=g.divisor if not isinstance(g.divisor,tuple) else g.divisor[0][z]*g.divisor[1][y]*g.divisor[2][x]
                np.divide(delta,np.asarray(divisor,dtype=seed.dtype),out=delta)
            indices=((coords[0]*shape[-2]+coords[1])*shape[-1]+coords[2])[valid]
            values=np.broadcast_to(delta[...,None],(last-first,stop-begin,taps))[:,valid]
            np.add.at(target,(np.arange(first,last)[:,None],indices),values)
    return result


def _apply(input,g):
    if isinstance(input,_autograd.ForwardTensor):return _autograd.ForwardTensor(_apply(input._primal,g),_apply(input._tangent,g))
    with np.errstate(all='ignore'):value=_forward(input._data,g)
    if not value.size:value=_layout.contiguous_empty(value)
    def backward(seed):
        with np.errstate(all='ignore'):return _backward(seed,input.shape,g)
    return _autograd.result(value,(input,backward),versioned=(input,))


def avg_pool3d(input,kernel_size,stride=None,padding=0,ceil_mode=False,count_include_pad=True,divisor_override=None):
    _input(input,False)
    if type(count_include_pad) is not bool:raise TypeError('count_include_pad must be bool')
    if divisor_override is not None:
        if not isinstance(divisor_override,numbers.Integral) or isinstance(divisor_override,(bool,np.bool_)):raise TypeError('divisor_override must be an integer')
        divisor_override=int(divisor_override)
        if not -(2**63)<=divisor_override<2**63:raise OverflowError('divisor_override exceeds int64')
        if divisor_override==0:raise RuntimeError('divisor_override must not be zero')
    kernel=_pooling3d._triple(kernel_size,'kernel_size')
    if any(n<k for n,k in zip(input.shape[-3:],kernel)):raise RuntimeError('avg_pool3d input is smaller than its kernel')
    kernel,stride,padding,_,out=_pooling3d._geometry(input,kernel,stride,padding,1,ceil_mode);_allocation(input.shape,out)
    return _apply(input,_geometry(input.shape,out,kernel,stride,padding,count_include_pad,divisor_override))


def adaptive_avg_pool3d(input,output_size):
    _input(input,True,True);out=_output_size(output_size,input.shape[-3:]);_allocation(input.shape,out)
    if out==(1,1,1):return _average_pooling._global_mean(input,spatial_dims=3,format_output=_format_global)
    _input(input,True)
    return _apply(input,_geometry(input.shape,out))


def _format_global(source,value):
    # Native global pooling uses mean but canonicalizes its output layout.
    # NumPy can retain Fortran-order leading axes or noncanonical singleton
    # strides even when the reduced result is marked C-contiguous.
    if not value.size:return _layout.contiguous_empty(value)
    if _pooling3d._channels_last(source):
        return value.transpose(0,2,3,4,1).copy(order='C').transpose(0,4,1,2,3)
    return value.copy(order='C')


class AvgPool3d(Module):
    def __init__(self,kernel_size,stride=None,padding=0,ceil_mode=False,count_include_pad=True,divisor_override=None):
        super().__init__();self.kernel_size=kernel_size;self.stride=kernel_size if stride is None else stride
        self.padding,self.ceil_mode=padding,ceil_mode;self.count_include_pad,self.divisor_override=count_include_pad,divisor_override
    def forward(self,input):return avg_pool3d(input,self.kernel_size,self.stride,self.padding,self.ceil_mode,self.count_include_pad,self.divisor_override)
    def extra_repr(self):return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'


class AdaptiveAvgPool3d(Module):
    def __init__(self,output_size):super().__init__();self.output_size=output_size
    def forward(self,input):return adaptive_avg_pool3d(input,self.output_size)
    def extra_repr(self):return f'output_size={self.output_size}'
