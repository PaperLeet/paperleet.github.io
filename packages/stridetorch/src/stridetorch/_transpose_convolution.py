"""Grouped transposed convolutions using the bounded convolution adjoint kernels."""
import itertools
import math
import weakref
import numpy as np
from . import _autograd, _convolution as _conv, _errors, _layout, _state, _tensor

_ALGORITHM = 'auto'
_MATRIX_WORKSPACE_BYTES = 32 * 1024 * 1024
_PAD_WORKSPACE_BYTES = 32 * 1024 * 1024


def _validate(input, weight, bias, stride, padding, output_padding, groups, dilation, dim):
    if not isinstance(input, _tensor.Tensor) or not isinstance(weight, _tensor.Tensor):
        raise TypeError('Transposed convolution input and weight must be Tensors')
    if input.ndim not in (dim+1,dim+2) or weight.ndim != dim+2:
        raise RuntimeError('Transposed convolution input/weight rank does not match its spatial dimension')
    if input.dtype not in _conv._REAL:
        raise _errors.BrowserUnsupported('Transposed convolution requires real float32/float64')
    if weight.dtype != input.dtype: raise RuntimeError('Input and weight dtypes must match')
    groups = _conv._integer(groups,'groups')
    if groups < 1: raise RuntimeError('groups must be positive')
    unbatched = input.ndim == dim+1
    shape = (1,*input.shape) if unbatched else input.shape
    channels = shape[1];outputs = weight.shape[1]*groups
    if channels != weight.shape[0] or not channels or channels%groups or not outputs:
        raise RuntimeError('Transposed convolution channels and grouped weights do not match')
    if bias is not None:
        if not isinstance(bias,_tensor.Tensor): raise TypeError('bias must be a Tensor')
        if bias.shape != (outputs,) or bias.dtype != input.dtype:
            raise RuntimeError('Transposed convolution bias shape/dtype does not match output channels')
    kernel = tuple(weight.shape[2:])
    if any(v < 1 for v in kernel): raise RuntimeError('Kernel dimensions must be positive')
    stride = _conv._tuple(stride,dim,'stride',minimum=1)
    if isinstance(padding,str) or isinstance(padding,(tuple,list)) and any(isinstance(v,str) for v in padding):
        raise TypeError('padding must be an integer or a sequence of integers')
    padding = _conv._tuple(padding,dim,'padding',minimum=0)
    output_padding = _conv._tuple(output_padding,dim,'output_padding',minimum=0)
    dilation = _conv._tuple(dilation,dim,'dilation',minimum=1)
    if any(op >= s and op >= d for op,s,d in zip(output_padding,stride,dilation)):
        raise RuntimeError('output padding must be smaller than either stride or dilation')
    effective = tuple(d*(k-1)+1 for d,k in zip(dilation,kernel))
    out = tuple((n-1)*s-2*p+e+op for n,s,p,e,op in zip(shape[2:],stride,padding,effective,output_padding))
    if any(v < 1 for v in out): raise RuntimeError('Calculated output spatial size must be positive')
    if shape[0] and not math.prod(shape[2:]): raise RuntimeError('Only empty batch inputs are supported')
    padded = tuple(n+2*p for n,p in zip(out,padding))
    return shape,unbatched,groups,kernel,stride,dilation,padding,padded,effective,out,outputs


def _use_matrix(input,weight,geometry):
    shape,_,_,kernel,_,_,_,padded,_,out,outputs = geometry
    if _ALGORITHM not in ('auto','direct','matrix'): raise ValueError('Unknown internal transposed convolution algorithm')
    if _ALGORITHM == 'direct' or not input.numel(): return False
    work = shape[0]*math.prod(shape[2:])*weight.numel()
    if _ALGORITHM == 'auto' and (math.prod(kernel)==1 or work<8192): return False
    original = (shape[0],outputs,*out)
    if _conv._matrix_workspace(original,weight.shape,kernel,out,shape[2:],input._data.itemsize)>_MATRIX_WORKSPACE_BYTES:
        return False
    return bool(np.isfinite(input._data).all() and np.isfinite(weight._data).all())


def _windows(data,shape,groups,kernel,stride,dilation,effective,outputs):
    dim = len(kernel)
    raw = np.lib.stride_tricks.sliding_window_view(data,effective,axis=tuple(range(2,dim+2)))
    # output_padding can exceed stride when dilation is larger. In that case
    # ordinary correlation has additional points; the adjoint uses only the
    # original input extent, rather than reshaping or summing those extra points.
    key = (slice(None),slice(None),*[slice(0,n*s,s) for n,s in zip(shape[2:],stride)],*[slice(None,None,d) for d in dilation])
    return raw[key].reshape(shape[0],groups,outputs//groups,*shape[2:],*kernel)


def _channels_last(data):
    if data.ndim!=4 or not data.size:return False
    distinct = not data.flags.c_contiguous or (data.shape[1]==1 and data.strides[1]!=math.prod(data.shape[2:])*data.itemsize)
    return distinct and np.moveaxis(data,1,-1).flags.c_contiguous


def _tap(position,shape,stride,dilation,padding,out):
    source,destination = [],[]
    for k,n,s,d,p,size in zip(position,shape[2:],stride,dilation,padding,out):
        start = k*d-p
        lo = max(0,(-start+s-1)//s)
        hi = min(n,(size-1-start)//s+1)
        if lo>=hi:return None
        source.append(slice(lo,hi));destination.append(slice(start+lo*s,start+hi*s,s))
    return tuple(source),tuple(destination)


def _gather_patches(raw,geometry):
    shape,_,groups,kernel,stride,dilation,padding,_,_,_,outputs = geometry
    dim = len(kernel)
    # Allocate directly in packed order. _conv._pack then returns a view rather
    # than allocating a second im2col buffer. Huge dilation/padding never cause
    # a huge zero-padded activation allocation here.
    packed = np.zeros((shape[0],groups,*shape[2:],outputs//groups,*kernel),dtype=raw.dtype)
    axes = (0,1,2+dim,*range(2,2+dim),*range(3+dim,3+2*dim))
    patches = packed.transpose(axes)
    grouped = raw.reshape(shape[0],groups,outputs//groups,*geometry[-2])
    for position in itertools.product(*[range(k) for k in kernel]):
        slices = _tap(position,shape,stride,dilation,padding,geometry[-2])
        if slices is None:continue
        source,destination = slices
        patches[(slice(None),slice(None),slice(None),*source,*position)] = grouped[(slice(None),slice(None),slice(None),*destination)]
    return patches


def _tapped_backward(raw,saved_input,saved_weights,geometry,needs):
    shape,_,groups,kernel,stride,dilation,padding,_,_,out,outputs = geometry
    grouped = raw.reshape(shape[0],groups,outputs//groups,*out)
    spatial = 'pqr'[:len(kernel)]
    dx = np.zeros((shape[0],groups,shape[1]//groups,*shape[2:]),dtype=raw.dtype) if needs[0] else None
    dw = np.zeros((groups,shape[1]//groups,outputs//groups,*kernel),dtype=raw.dtype) if needs[1] else None
    finite = bool(np.isfinite(raw).all() and (not needs[0] or np.isfinite(saved_weights).all())
                  and (not needs[1] or np.isfinite(saved_input).all()))
    for position in itertools.product(*[range(k) for k in kernel]):
        slices = _tap(position,shape,stride,dilation,padding,out)
        wt_key = (slice(None),slice(None),slice(None),*position)
        if not finite:
            # Implicit padded zeros participate in native backward's IEEE
            # products: 0*Inf must remain NaN even when a tap misses the output.
            seed = np.zeros((shape[0],groups,outputs//groups,*shape[2:]),dtype=raw.dtype)
            if slices is not None:
                source,destination = slices
                seed[(slice(None),slice(None),slice(None),*source)] = grouped[(slice(None),slice(None),slice(None),*destination)]
            if needs[0]:dx += np.einsum('gio,bgo...->bgi...',saved_weights[wt_key],seed,optimize=False)
            if needs[1]:dw[wt_key] = np.einsum(f'bgi{spatial},bgo{spatial}->gio',saved_input,seed,optimize=False)
        elif slices is not None:
            source,destination = slices
            key = (slice(None),slice(None),slice(None),*source)
            seed = grouped[(slice(None),slice(None),slice(None),*destination)]
            if needs[0]:dx[key] += np.einsum('gio,bgo...->bgi...',saved_weights[wt_key],seed,optimize=False)
            if needs[1]:dw[wt_key] = np.einsum(f'bgi{spatial},bgo{spatial}->gio',saved_input[key],seed,optimize=False)
    return dx,dw


def _primal(input,weight,bias,geometry):
    shape,unbatched,groups,kernel,stride,dilation,padding,padded,effective,out,outputs = geometry
    dim = len(kernel);dtype = input.dtype;empty = not input.numel()
    grouped = input._data.reshape(shape[0],groups,shape[1]//groups,*shape[2:])
    weights = weight._data.reshape(groups,shape[1]//groups,outputs//groups,*kernel)
    matrix = _use_matrix(input,weight,geometry)
    with np.errstate(all='ignore'):
        if empty: value = _layout.contiguous_empty(np.empty((shape[0],outputs,*out),dtype=dtype))
        else:
            value = np.zeros((shape[0],groups,outputs//groups,*out),dtype=dtype)
            if matrix:
                columns = np.swapaxes(grouped.reshape(shape[0],groups,shape[1]//groups,-1),-1,-2) @ weights.reshape(groups,shape[1]//groups,-1)
                packed = columns.reshape(shape[0],groups,*shape[2:],outputs//groups,*kernel)
                axes = (0,1,2+dim,*range(2,2+dim),*range(3+dim,3+2*dim))
                patch_gradient = packed.transpose(axes)
            for position in itertools.product(*[range(k) for k in kernel]):
                slices = _tap(position,shape,stride,dilation,padding,out)
                if slices is None:continue
                source,destination = slices
                if matrix:
                    delta = patch_gradient[(slice(None),slice(None),slice(None),*source,*position)]
                else:
                    wt = weights[(slice(None),slice(None),slice(None),*position)]
                    delta = np.einsum('gio,bgi...->bgo...',wt,grouped[(slice(None),slice(None),slice(None),*source)],optimize=False)
                value[(slice(None),slice(None),slice(None),*destination)] += delta
            value = value.reshape(shape[0],outputs,*out)
            if bias is not None: value += bias._data.reshape(1,outputs,*([1]*dim))
            if dim==2 and ((not unbatched and _channels_last(input._data)) or _channels_last(weight._data)):
                value = np.moveaxis(np.ascontiguousarray(np.moveaxis(value,1,-1)),-1,1)
    active_x = input.requires_grad and _state._grad_enabled
    active_w = weight.requires_grad and _state._grad_enabled
    saved_input = grouped if active_w else None
    saved_weights = weights if active_x else None
    input_shape,weight_shape = input.shape,weight.shape
    guarded = (input,weight)
    original = [(weakref.ref(v),weakref.ref(v._data)) for v in guarded]
    spatial,taps = 'pqr'[:dim],'uvw'[:dim]

    def backward(seed,needs):
        for owner_ref,data_ref in original:
            owner = owner_ref()
            if owner is not None and owner._data is not data_ref():
                raise _errors.BrowserUnsupported('Transposed convolution backward after saved storage rebinding is unsupported')
        values = [None]*len(needs)
        if empty:
            if needs[0]:values[0] = np.zeros(input_shape,dtype=dtype)
            if needs[1]:values[1] = np.zeros(weight_shape,dtype=dtype)
            if len(needs)>2 and needs[2]:values[2] = np.zeros(outputs,dtype=dtype)
            return values
        raw = np.asarray(seed).reshape(shape[0],outputs,*out)
        matrix_backward = matrix and bool(np.isfinite(raw).all())
        with np.errstate(all='ignore'):
            if needs[0] or needs[1]:
                padded_bytes = math.prod((shape[0],outputs,*padded))*raw.itemsize
                gathered_bytes = shape[0]*outputs*math.prod(shape[2:])*math.prod(kernel)*raw.itemsize
                huge_padding = padded_bytes>_PAD_WORKSPACE_BYTES or padded_bytes>4*gathered_bytes
                if huge_padding and not matrix_backward:
                    dx,dw = _tapped_backward(raw,saved_input,saved_weights,geometry,needs)
                    if needs[0]:values[0] = dx.reshape(input_shape)
                    if needs[1]:values[1] = dw.reshape(weight_shape)
                else:
                    if huge_padding:patches = _gather_patches(raw,geometry)
                    else:
                        padded_seed,_ = _conv._pad(raw,tuple((p,p) for p in padding),'zeros')
                        patches = _windows(padded_seed,shape,groups,kernel,stride,dilation,effective,outputs)
                    if needs[0]:
                        dx = (_conv._matrix_forward(patches,saved_weights,shape[2:]) if matrix_backward else
                            np.einsum(f'bgo{spatial}{taps},gio{taps}->bgi{spatial}',patches,saved_weights,optimize=False))
                        values[0] = dx.reshape(input_shape)
                    if needs[1]:
                        values[1] = (_conv._matrix_weight_gradient(saved_input,patches,weight_shape) if matrix_backward else
                            np.einsum(f'bgi{spatial},bgo{spatial}{taps}->gio{taps}',saved_input,patches,optimize=False).reshape(weight_shape))
            if len(needs)>2 and needs[2]:values[2] = raw.sum(axis=(0,*range(2,dim+2)))
        return values

    if unbatched:value = value[0]
    return _autograd.joint_result(value,[input,weight,*([bias] if bias is not None else [])],backward,versioned=guarded)


def _transpose(input,weight,bias,stride,padding,output_padding,groups,dilation,dim):
    geometry = _validate(input,weight,bias,stride,padding,output_padding,groups,dilation,dim)
    if not any(isinstance(v,_autograd.ForwardTensor) for v in (input,weight,bias)):
        return _primal(input,weight,bias,geometry)
    def parts(v):return (v._primal,v._tangent) if isinstance(v,_autograd.ForwardTensor) else (v,None)
    x,dx = parts(input);w,dw = parts(weight);b,db = parts(bias)
    primal = _primal(x,w,b,geometry)
    if dx is None:dx = _tensor.Tensor(np.zeros(x.shape,dtype=x.dtype))
    if dw is None:dw = _tensor.Tensor(np.zeros(w.shape,dtype=w.dtype))
    tangent = _primal(dx,w,None,geometry)+_primal(x,dw,None,geometry)
    if db is not None:
        broadcast = (geometry[-1],*([1]*dim)) if geometry[1] else (1,geometry[-1],*([1]*dim))
        tangent = tangent+db.reshape(broadcast).expand_as(primal)
    if not primal.numel():
        tangent = _autograd.view_result(_layout.contiguous_empty(tangent._data),tangent,lambda g:g)
    return _autograd.ForwardTensor(primal,tangent)


def conv_transpose1d(input,weight,bias=None,stride=1,padding=0,output_padding=0,groups=1,dilation=1):
    return _transpose(input,weight,bias,stride,padding,output_padding,groups,dilation,1)


def conv_transpose2d(input,weight,bias=None,stride=1,padding=0,output_padding=0,groups=1,dilation=1):
    return _transpose(input,weight,bias,stride,padding,output_padding,groups,dilation,2)


def conv_transpose3d(input,weight,bias=None,stride=1,padding=0,output_padding=0,groups=1,dilation=1):
    return _transpose(input,weight,bias,stride,padding,output_padding,groups,dilation,3)


class _ConvTransposeNd(_conv._ConvNd):
    _transposed = True

    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0,
                 groups=1,bias=True,dilation=1,padding_mode='zeros',device=None,dtype=None):
        if padding_mode!='zeros':raise ValueError('Only zeros padding mode is supported for transposed convolution')
        super().__init__(in_channels,out_channels,kernel_size,stride,0 if isinstance(padding,str) else padding,
            dilation,groups,bias,padding_mode,device,dtype)
        if isinstance(padding,str):
            self.padding = tuple(padding)
            self._reversed_padding_repeated_twice = tuple(v for v in reversed(padding) for _ in range(2))
        self.output_padding = _conv._tuple(output_padding,self._dimension,'output_padding')

    def _output_padding(self,input,output_size,stride,padding,kernel_size,num_spatial_dims,dilation=None):
        if output_size is None:return self.output_padding
        dim = num_spatial_dims;nonspatial = 2 if input.ndim==dim+2 else 1
        output_size = tuple(output_size)
        if len(output_size)==dim+nonspatial:output_size = output_size[nonspatial:]
        if len(output_size)!=dim:raise ValueError('output_size must contain the spatial dimensions or full output shape')
        output_size = tuple(_conv._integer(v,'output_size') for v in output_size)
        dilation = (1,)*dim if dilation is None else dilation
        minimum = tuple((n-1)*s-2*p+d*(k-1)+1 for n,s,p,d,k in zip(input.shape[nonspatial:],stride,padding,dilation,kernel_size))
        maximum = tuple(n+s-1 for n,s in zip(minimum,stride))
        if any(v<lo or v>hi for v,lo,hi in zip(output_size,minimum,maximum)):
            raise ValueError(f'requested output_size {output_size}, valid sizes range from {minimum} to {maximum}')
        return tuple(v-lo for v,lo in zip(output_size,minimum))

    def forward(self,input,output_size=None):
        dim = self._dimension
        if self.padding_mode!='zeros':raise ValueError('Only zeros padding mode is supported for transposed convolution')
        if input.ndim not in (dim+1,dim+2):raise RuntimeError('Transposed convolution input rank does not match its spatial dimension')
        op = self._output_padding(input,output_size,self.stride,self.padding,self.kernel_size,dim,self.dilation)
        return _transpose(input,self.weight,self.bias,self.stride,self.padding,op,self.groups,self.dilation,dim)

class ConvTranspose1d(_ConvTransposeNd):
    _dimension = 1


class ConvTranspose2d(_ConvTransposeNd):
    _dimension = 2


class ConvTranspose3d(_ConvTransposeNd):
    _dimension = 3
