"""Checked indexing and value selection, with compact analytical pullbacks."""
from collections import namedtuple
import numpy as np
from . import _autograd, _errors, _layout, _state, _tensor

_INDEX_DTYPES = (np.dtype('int32'), np.dtype('int64'))
_HALF = np.dtype('float16')


Sort = namedtuple('Sort', ['values', 'indices'])
TopK = namedtuple('TopK', ['values', 'indices'])


def sort_arguments(args, dim, descending, stable):
    missing=_layout.UNSPECIFIED_DIM
    if len(args)>2 or stable is not missing and args:
        raise TypeError('stable sorting requires keyword dimensions and flags')
    if args:
        if dim is not missing:raise TypeError('Duplicate sorting dimension')
        dim=args[0]
        if len(args)==2:
            if descending is not missing:raise TypeError('Duplicate descending flag')
            descending=args[1]
    return (-1 if dim is missing else dim,False if descending is missing else descending,
            False if stable is missing else stable)


def _flag(value, name):
    if type(value) is not bool:raise TypeError(name+' must be bool')
    return value


def _out_of_bounds(index, size):
    # Integer reductions avoid an index-sized boolean temporary for each
    # comparison, including on noncontiguous and broadcast index views.
    if index.dtype.kind == 'i':
        return bool(index.size and (index.min() < 0 or index.max() >= size))
    # Preserve comparison semantics for other arrays (notably NaNs) that may
    # reach a saved-index pullback after raw NumPy storage metadata changes.
    return np.any(index < 0) or np.any(index >= size)


def _bounds(index, size):
    if _out_of_bounds(index, size):raise RuntimeError('gather index out of bounds')


def _take(data, index, axis):
    if max(data.ndim,index.ndim)<=1:
        return np.take(data.reshape(-1),index,axis=0)
    coordinates=tuple(np.arange(n,dtype=np.intp).reshape((1,)*d+(n,)+(1,)*(index.ndim-d-1))
                      if d!=axis else index for d,n in enumerate(index.shape))
    return data[coordinates]


def _scatter(seed, index, shape, dtype, axis, *, unique=False):
    # CPU half scatter-add accumulates in float32, then rounds once. NumPy's
    # half add.at instead rounds every addition and drifts on repeated indices.
    accumulator=np.float32 if dtype==_HALF and not unique else dtype
    gradient=np.zeros(shape,dtype=accumulator)
    if not index.size:return gradient.astype(dtype,copy=False)
    size=shape[axis] if shape else 1
    _bounds(index,size)
    if max(len(shape),index.ndim)<=1:
        if unique:gradient.reshape(-1)[index.reshape(-1)]=seed.reshape(-1)
        else:np.add.at(gradient.reshape(-1),index.reshape(-1),seed.reshape(-1))
    elif unique:np.put_along_axis(gradient,index,seed,axis=axis)
    else:
        # Broadcast coordinate vectors instead of rank * selected-size grids.
        coordinates=[np.arange(n,dtype=np.intp).reshape((1,)*d+(n,)+(1,)*(index.ndim-d-1))
                     if d!=axis else index for d,n in enumerate(index.shape)]
        np.add.at(gradient,tuple(coordinates),seed)
    return gradient.astype(dtype,copy=False)


def gather(input, dim, index, *, sparse_grad=False):
    requested=_layout.integer(dim)
    _flag(sparse_grad,'sparse_grad')
    if not isinstance(index,_tensor.Tensor):raise TypeError('gather index must be a Tensor')
    data,indices=input._data,index._data
    axis=_layout.dimension(requested,data.ndim)
    if indices.size:
        if indices.dtype not in _INDEX_DTYPES:
            raise RuntimeError('gather requires int32 or int64 indices')
        if indices.ndim!=data.ndim and max(indices.ndim,data.ndim)>1:
            raise RuntimeError('gather index rank must match input')
        if any(indices.shape[d]>data.shape[d] for d in range(data.ndim) if d!=axis):
            raise RuntimeError('gather dimensions do not match')
        _bounds(indices,data.shape[axis] if data.ndim else 1)
    if isinstance(input,_autograd.ForwardTensor):
        return _autograd.ForwardTensor(gather(input._primal,axis,index,sparse_grad=sparse_grad),
            gather(input._tangent,axis,index,sparse_grad=sparse_grad))
    if sparse_grad and input.requires_grad and _state._grad_enabled:
        raise _errors.BrowserUnsupported('Sparse gather gradients are unsupported')
    shape,dtype=data.shape,data.dtype
    value=_take(data,indices,axis) if indices.size else np.empty(indices.shape,dtype=dtype)
    def backward(seed):return _scatter(seed,index._data,shape,dtype,axis)
    # Native gather saves its input and indices; its output is not saved.
    return _autograd.result(value,(input,backward),versioned=(input,index))


def argsort(input, dim=-1, descending=False, *, stable=False):
    requested=_layout.integer(dim)
    _flag(descending,'descending');_flag(stable,'stable')
    axis=_layout.dimension(requested,input.ndim)
    if input.dtype.kind not in 'fiub':raise _errors.BrowserUnsupported('Sorting this dtype is unsupported')
    if not input.ndim:return _tensor.Tensor(0,dtype=np.int64)
    kind='stable' if stable else 'quicksort'
    if descending:
        indices=input.shape[axis]-1-np.flip(np.argsort(np.flip(input._data,axis=axis),axis=axis,kind=kind),axis=axis)
    else:indices=np.argsort(input._data,axis=axis,kind=kind)
    return _tensor.Tensor(indices,dtype=np.int64)


def _selected(input, axis, indices):
    shape,dtype=input.shape,input.dtype
    value=_take(input._data,indices._data,axis)
    def backward(seed):return _scatter(seed,indices._data,shape,dtype,axis,unique=True)
    # Sort/topk save only the selected indices and metadata, not input values.
    return _autograd.result(value,(input,backward),versioned=(indices,))


def sort(input, dim=-1, descending=False, *, stable=False, out=None):
    if out is not None:raise _errors.BrowserUnsupported('Sort output buffers are unsupported')
    indices=argsort(input,dim,descending,stable=False if stable is None else stable)
    axis=_layout.dimension(dim,input.ndim)
    if isinstance(input,_autograd.ForwardTensor):
        value=_autograd.ForwardTensor(_selected(input._primal,axis,indices),gather(input._tangent,axis,indices))
    else:value=_selected(input,axis,indices)
    return Sort(value,indices)


def topk(input, k, dim=-1, largest=True, sorted=True, *, out=None):
    if out is not None:raise _errors.BrowserUnsupported('Topk output buffers are unsupported')
    count=_layout.integer(k,'k');requested=_layout.integer(dim)
    _flag(largest,'largest');_flag(sorted,'sorted')
    axis=_layout.dimension(requested,input.ndim)
    size=input.shape[axis] if input.ndim else 1
    if not 0<=count<=size:raise RuntimeError('topk selected index k out of range')
    if input.dtype.kind not in 'fiu':raise _errors.BrowserUnsupported('Topk for this dtype is unsupported')
    if not input.ndim:selected=np.asarray(0,dtype=np.int64)
    elif not count:
        shape=list(input.shape);shape[axis]=0;selected=np.empty(shape,dtype=np.int64)
    else:
        slices=[slice(None)]*input.ndim
        if size>=64 and count<=size//4:
            cut=size-count if largest else count-1
            partition=np.argpartition(input._data,cut,axis=axis)
            slices[axis]=slice(size-count,None) if largest else slice(0,count)
            selected=partition[tuple(slices)]
            if sorted:
                values=_take(input._data,selected,axis)
                order=np.argsort(values,axis=axis,kind='stable')
                if largest:order=np.flip(order,axis=axis)
                selected=np.take_along_axis(selected,order,axis=axis)
        else:
            ordered=argsort(input,axis,descending=largest,stable=True)
            slices[axis]=slice(0,count);selected=ordered._data[tuple(slices)]
        # A narrow view must not retain the whole partition/permutation array.
        selected=selected.astype(np.int64,copy=True)
    indices=_tensor.Tensor(selected,_metadata_owned=True)
    if isinstance(input,_autograd.ForwardTensor):
        value=_autograd.ForwardTensor(_selected(input._primal,axis,indices),gather(input._tangent,axis,indices))
    else:value=_selected(input,axis,indices)
    return TopK(value,indices)
