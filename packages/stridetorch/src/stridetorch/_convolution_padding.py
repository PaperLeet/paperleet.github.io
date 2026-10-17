"""Clipped convolution gather/scatter without a full zero-padded activation."""
import itertools
import numpy as np


def tap_slices(position,shape,stride,dilation,padding,out):
    source,destination = [],[]
    for k,n,s,d,p,size in zip(position,shape[2:],stride,dilation,padding,out):
        start = k*d-p
        lo = max(0,(-start+s-1)//s)
        hi = min(n,(size-1-start)//s+1)
        if lo>=hi:return None
        source.append(slice(lo,hi));destination.append(slice(start+lo*s,start+hi*s,s))
    return tuple(source),tuple(destination)


def gather(data,shape,groups,kernel,stride,dilation,padding,out):
    dim = len(kernel);channels = shape[1]//groups
    packed = np.zeros((shape[0],groups,*out,channels,*kernel),dtype=data.dtype)
    axes = (0,1,2+dim,*range(2,2+dim),*range(3+dim,3+2*dim))
    patches = packed.transpose(axes)
    grouped = data.reshape(shape[0],groups,channels,*shape[2:])
    points = (shape[0],shape[1],*out)
    for position in itertools.product(*[range(k) for k in kernel]):
        slices = tap_slices(position,points,stride,dilation,padding,shape[2:])
        if slices is None:continue
        source,destination = slices
        patches[(slice(None),slice(None),slice(None),*source,*position)] = grouped[(slice(None),slice(None),slice(None),*destination)]
    return patches


def scatter(seed,weights,shape,kernel,stride,dilation,padding,points,matrix):
    n,groups = seed.shape[:2];channels = weights.shape[2];dim = len(kernel)
    value = np.zeros((n,groups,channels,*shape[2:]),dtype=seed.dtype)
    if matrix:
        columns = np.swapaxes(seed.reshape(n,groups,seed.shape[2],-1),-1,-2) @ weights.reshape(groups,weights.shape[1],-1)
        packed = columns.reshape(n,groups,*points,channels,*kernel)
        axes = (0,1,2+dim,*range(2,2+dim),*range(3+dim,3+2*dim))
        patches = packed.transpose(axes)
    grid = (n,groups*seed.shape[2],*points)
    for position in itertools.product(*[range(k) for k in kernel]):
        slices = tap_slices(position,grid,stride,dilation,padding,shape[2:])
        if slices is None:continue
        source,destination = slices
        if matrix:delta = patches[(slice(None),slice(None),slice(None),*source,*position)]
        else:
            wt = weights[(slice(None),slice(None),slice(None),*position)]
            delta = np.einsum('goi,bgo...->bgi...',wt,seed[(slice(None),slice(None),slice(None),*source)],optimize=False)
        value[(slice(None),slice(None),slice(None),*destination)] += delta
    return value.reshape(shape)


def forward_taps(data,weights,shape,kernel,stride,dilation,padding,out):
    groups,outputs = weights.shape[:2];dim = len(kernel)
    grouped = data.reshape(shape[0],groups,shape[1]//groups,*shape[2:])
    value = np.zeros((shape[0],groups,outputs,*out),dtype=data.dtype)
    grid = (shape[0],shape[1],*out)
    # Missing input locations are implicit zeros. Nonfinite weights must still
    # multiply those zeros; finite weights can skip the out-of-bounds region.
    finite_weights = bool(np.isfinite(weights).all())
    for position in itertools.product(*[range(k) for k in kernel]):
        slices = tap_slices(position,grid,stride,dilation,padding,shape[2:])
        wt = weights[(slice(None),slice(None),slice(None),*position)]
        if not finite_weights:
            tile = np.zeros((shape[0],groups,shape[1]//groups,*out),dtype=data.dtype)
            if slices is not None:
                source,destination = slices
                tile[(slice(None),slice(None),slice(None),*source)] = grouped[(slice(None),slice(None),slice(None),*destination)]
            value += np.einsum('goi,bgi...->bgo...',wt,tile,optimize=False)
        elif slices is not None:
            source,destination = slices
            value[(slice(None),slice(None),slice(None),*source)] += np.einsum('goi,bgi...->bgo...',wt,grouped[(slice(None),slice(None),slice(None),*destination)],optimize=False)
    return value.reshape(shape[0],groups*outputs,*out)


def weight_gradient_taps(seed,data,shape,weight_shape,groups,kernel,stride,dilation,padding,out):
    dim = len(kernel);spatial = 'pqr'[:dim]
    grouped = data.reshape(shape[0],groups,shape[1]//groups,*shape[2:])
    gradient = np.zeros((groups,weight_shape[0]//groups,shape[1]//groups,*kernel),dtype=data.dtype)
    finite_seed = bool(np.isfinite(seed).all())
    grid = (shape[0],shape[1],*out)
    for position in itertools.product(*[range(k) for k in kernel]):
        slices = tap_slices(position,grid,stride,dilation,padding,shape[2:])
        target = (slice(None),slice(None),slice(None),*position)
        if not finite_seed:
            tile = np.zeros((shape[0],groups,shape[1]//groups,*out),dtype=data.dtype)
            if slices is not None:
                source,destination = slices
                tile[(slice(None),slice(None),slice(None),*source)] = grouped[(slice(None),slice(None),slice(None),*destination)]
            gradient[target] = np.einsum(f'bgo{spatial},bgi{spatial}->goi',seed,tile,optimize=False)
        elif slices is not None:
            source,destination = slices
            gradient[target] = np.einsum(f'bgo{spatial},bgi{spatial}->goi',seed[(slice(None),slice(None),slice(None),*source)],grouped[(slice(None),slice(None),slice(None),*destination)],optimize=False)
    return gradient.reshape(weight_shape)
