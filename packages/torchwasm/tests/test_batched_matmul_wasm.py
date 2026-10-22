"""Broadcast derivatives, layout, range and memory invariants for WASM batches."""
import numpy as np
import pytest
import gc
import weakref
from torchwasm import torch,_matmul,_tensor,_attention


@pytest.fixture(autouse=True)
def wasm_matrix_kernels(monkeypatch):
    monkeypatch.setattr(_tensor,'_WASM_MATMUL',True)
    monkeypatch.setattr(_attention,'_WASM_ATTENTION',True)


def arrange(value,kind):
    if kind=='F':return np.asfortranarray(value)
    if kind=='stride':return np.repeat(value,2,axis=-1)[...,::2]
    if kind=='repeat':return np.broadcast_to(value[:1],value.shape)
    return value


def independent_products(a,b,ta,tb,seed):
    """Per-output-batch float64 algebra, with explicit gradient accumulation."""
    a,b,ta,tb,seed=[x.astype('float64') for x in (a,b,ta,tb,seed)]
    shape=np.broadcast_shapes(a.shape[:-2],b.shape[:-2]);m,k=a.shape[-2:];n=b.shape[-1]
    y=np.empty(shape+(m,n));dy=np.empty_like(y)
    ga=np.zeros_like(a);gb=np.zeros_like(b);ha=np.zeros_like(a);hb=np.zeros_like(b)
    def source_index(index,prefix):
        offset=len(index)-len(prefix)
        return tuple(0 if extent==1 else index[offset+i] for i,extent in enumerate(prefix))
    for index in np.ndindex(shape):
        ai=source_index(index,a.shape[:-2]);bi=source_index(index,b.shape[:-2])
        x,w,dx,dw=a[ai],b[bi],ta[ai],tb[bi]
        value=x@w;direction=dx@w+x@dw;y[index]=value;dy[index]=direction
        ga[ai]+=seed[index]@w.T;gb[bi]+=x.T@seed[index]
        ha[ai]+=direction@w.T+value@dw.T;hb[bi]+=dx.T@value+x.T@direction
    return [y,ga,gb,dy,ha,hb,(y*y).sum()*.5]


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shapes',[
    ((4,64,64),(4,64,64)),((2,1,64,64),(1,4,64,64)),
    ((1,64,64),(4,64,64)),((4,64,64),(64,64)),
    ((64,64),(4,64,64)),((4,32,128),(4,128,64))])
@pytest.mark.parametrize('order',['C','F','stride','repeat'])
def test_batch_values_vjp_jvp_hvp_and_layout(dtype,shapes,order,monkeypatch):
    rng=np.random.default_rng(77071)
    a,b,ta,tb=[arrange((rng.integers(-16,17,size=s,dtype=np.int32).astype(dtype)/128),order) for s in [*shapes,*shapes]]
    shape=np.broadcast_shapes(a.shape[:-2],b.shape[:-2])+(a.shape[-2],b.shape[-1])
    seed=rng.integers(-16,17,size=shape,dtype=np.int32).astype(dtype)/128
    expected=independent_products(a,b,ta,tb,seed)
    x,w,dx,dw=[torch.from_numpy(data).requires_grad_(i<2) for i,data in enumerate((a,b,ta,tb))]
    calls=[];original=np.einsum
    def observed(*args,**kwargs):calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(np,'einsum',observed)
    y=x@w;ga,gb=torch.autograd.grad(y,(x,w),torch.from_numpy(seed))
    primal,direction=torch.func.jvp(lambda x,w:x@w,(x,w),(dx,dw))
    loss,(ha,hb)=torch.autograd.functional.hvp(lambda x,w:(x@w).square().sum()*.5,(x,w),(dx,dw))
    assert calls and all(call=='...ik,...jk->...ij' for call in calls)
    tolerance=3e-5 if dtype==np.float32 else 3e-11
    for value,control in zip([y,ga,gb,direction,ha,hb,loss],expected):
        np.testing.assert_allclose(value.detach().numpy(),control,rtol=tolerance,atol=tolerance)
    np.testing.assert_array_equal(primal.numpy(),y.numpy())
    assert y.is_contiguous() and tuple(y.shape)==shape


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('operand',[0,1])
@pytest.mark.parametrize('kind',['nan','inf','large','tiny','subnormal','small_normal'])
def test_exceptional_batches_keep_original_matmul(dtype,operand,kind,monkeypatch):
    a=np.ones((4,64,64),dtype=dtype);b=np.ones_like(a);finfo=np.finfo(dtype)
    [a,b][operand][0,0,0]={'nan':np.nan,'inf':np.inf,'large':finfo.max/2,'tiny':finfo.tiny,
                           'subnormal':finfo.smallest_subnormal,'small_normal':np.sqrt(finfo.tiny)/2}[kind]
    def unexpected(*args,**kwargs):raise AssertionError('Exceptional batch reached contraction')
    monkeypatch.setattr(np,'einsum',unexpected)
    with np.errstate(all='ignore'):
        expected=a@b;actual=(torch.from_numpy(a)@torch.from_numpy(b)).numpy()
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('active',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('changed',['a','b','output'])
def test_batch_keeps_minimal_saved_value_guards(active,changed):
    a=torch.ones(4,64,64,requires_grad=active[0]);b=torch.ones(4,64,64,requires_grad=active[1]);y=a@b
    with torch.no_grad():{'a':a,'b':b,'output':y}[changed].detach().add_(1)
    guarded=changed=='a' and active[1] or changed=='b' and active[0]
    if guarded:
        with pytest.raises(RuntimeError,match='modified'):y.sum().backward()
    else:
        y.sum().backward()
        if active[0]:np.testing.assert_array_equal(a.grad.numpy(),np.full((4,64,64),64))
        if active[1]:np.testing.assert_array_equal(b.grad.numpy(),np.full((4,64,64),64))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_zero_stride_batch_does_not_expand_input_scans_or_packing(dtype,monkeypatch):
    a=np.broadcast_to(np.ones((1,64,64),dtype=dtype),(8,64,64));b=np.broadcast_to(np.ones((1,64,64),dtype=dtype),(8,64,64))
    scans=[];packs=[];calls=[];absolute=np.abs;contiguous=np.ascontiguousarray;einsum=np.einsum
    def scan(value,*args,**kwargs):scans.append(value.shape);return absolute(value,*args,**kwargs)
    def pack(value,*args,**kwargs):
        if np.shares_memory(value,a) or np.shares_memory(value,b):packs.append(value.shape)
        return contiguous(value,*args,**kwargs)
    def product(expression,left,right,**kwargs):
        calls.append((left.strides[0],right.strides[0]));return einsum(expression,left,right,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(np,'abs',scan);patch.setattr(np,'ascontiguousarray',pack);patch.setattr(np,'einsum',product)
        result=_matmul.product(a,b)
    assert scans==[(1,64,64),(1,64,64)]
    assert packs and all(shape[0]==1 for shape in packs)
    assert calls==[(0,0)]
    np.testing.assert_array_equal(result,a@b)


@pytest.mark.parametrize('rank',[3,34,35,64])
def test_high_rank_fallback_remains_available(rank,monkeypatch):
    try:
        a=np.ones((1,)*(rank-2)+(128,64),dtype='float32');b=np.ones((1,)*(rank-2)+(64,128),dtype='float32')
    except ValueError as error:
        if 'dimension' in str(error):pytest.skip('Installed NumPy cannot construct this rank')
        raise
    if rank>34:
        def unexpected(*args,**kwargs):raise AssertionError('High rank reached shape-only broadcast or contraction')
        monkeypatch.setattr(np,'broadcast_shapes',unexpected);monkeypatch.setattr(np,'einsum',unexpected)
    np.testing.assert_array_equal(_matmul.product(a,b),a@b)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(128,64),(4,64,64)])
def test_cold_limits_cache_does_not_retain_dtype_metadata(dtype,shape):
    class Owner:pass
    saved=dict(np.finfo._finfo_cache)
    try:
        np.finfo._finfo_cache.clear()
        owner=Owner();ref=weakref.ref(owner)
        descriptor=np.dtype(dtype,metadata={'owner':owner})
        a=np.ones(shape,dtype=descriptor)
        b=np.ones((64,128) if len(shape)==2 else shape,dtype=dtype)
        x=torch.from_numpy(a);w=torch.from_numpy(b);y=x@w
        del owner,descriptor,a,b,x,w,y
        gc.collect()
        assert ref() is None
        assert not any(getattr(key,'metadata',None) for key in np.finfo._finfo_cache)
    finally:
        np.finfo._finfo_cache.clear();np.finfo._finfo_cache.update(saved)
