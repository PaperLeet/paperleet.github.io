"""Ordered plan equivalence, ownership, live aliases and bounded metadata."""
import gc
import weakref
import numpy as np
import pytest
from torchwasm import torch, _average_pooling as pooling


CASES = [((2,3,17,23),(5,3)), ((2,3,5,7),(12,14))]


def layout(array, kind):
    if kind == 'fortran': return np.asfortranarray(array)
    if kind == 'strided': return np.repeat(array, 2, axis=-1)[..., ::2]
    return array


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('dtype', [np.float16,np.float32,np.float64])
@pytest.mark.parametrize('kind', ['finite','nonfinite','cancellation'])
@pytest.mark.parametrize('order', ['contiguous','fortran','strided'])
def test_cached_and_uncached_ordered_values_and_pullbacks(monkeypatch,case,dtype,kind,order):
    shape,size=case; rng=np.random.default_rng(4003)
    a=rng.normal(size=shape).astype(dtype)
    if kind=='nonfinite': a[...,0,0]=np.inf; a[...,-1,-1]=np.nan
    if kind=='cancellation':
        scale=100 if dtype is np.float16 else 1e6
        a[...] = np.resize(np.asarray([scale,1.3,-scale,1.3],dtype=dtype),a.shape)
    a=layout(a,order); original=a.copy(); rows=[]; seed=None
    monkeypatch.setattr(pooling,'_ALGORITHM','packed')
    monkeypatch.setattr(pooling,'_BACKWARD_ALGORITHM','packed')
    for cached in [False,True]:
        monkeypatch.setattr(pooling,'_CACHE_PLANS',cached)
        x=torch.from_numpy(a).requires_grad_(dtype is not np.float16)
        y=torch.nn.functional.adaptive_avg_pool2d(x,size)
        if seed is None:
            seed=rng.normal(size=y.shape).astype(dtype)
            if kind=='nonfinite':seed[...,0,0]=np.inf;seed[...,-1,-1]=np.nan
        if dtype is not np.float16:y.backward(torch.tensor(seed))
        rows.append((y.numpy().copy(),None if x.grad is None else x.grad.numpy().copy()))
        np.testing.assert_array_equal(a,original)
    for left,right in zip(*rows):
        if left is None:assert right is None
        else:
            np.testing.assert_array_equal(left,right)
            np.testing.assert_array_equal(np.signbit(left),np.signbit(right))


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('order', ['contiguous','fortran'])
@pytest.mark.parametrize('nonfinite', [False,True])
def test_cached_jvp_and_tangent_reverse_match_ordered_path(monkeypatch,case,dtype,order,nonfinite):
    shape,size=case;rng=np.random.default_rng(4004)
    a=layout(rng.normal(size=shape).astype(dtype),order)
    direction=rng.normal(size=shape).astype(dtype)
    if nonfinite:direction[...,0,0]=np.inf;direction[...,-1,-1]=np.nan
    rows=[];monkeypatch.setattr(pooling,'_ALGORITHM','packed');monkeypatch.setattr(pooling,'_BACKWARD_ALGORITHM','packed')
    for cached in [False,True]:
        monkeypatch.setattr(pooling,'_CACHE_PLANS',cached)
        x=torch.from_numpy(a).requires_grad_();v=torch.tensor(direction,requires_grad=True)
        p,t=torch.func.jvp(lambda z:torch.nn.functional.adaptive_avg_pool2d(z,size),(x,),(v,))
        t.sum().backward();assert x.grad is None
        rows.append([z.numpy().copy() for z in (p,t,v.grad)])
    for left,right in zip(*rows):np.testing.assert_array_equal(left,right)


@pytest.mark.parametrize('dtype', [np.float16,np.float32,np.float64])
@pytest.mark.parametrize('order', ['contiguous','fortran','strided'])
def test_readonly_input_and_owned_output_do_not_enter_cache(monkeypatch,dtype,order):
    monkeypatch.setattr(pooling,'_ALGORITHM','packed')
    a=layout(np.arange(17*23,dtype=dtype).reshape(1,1,17,23),order)
    before=a.copy();a.setflags(write=False);ref=weakref.ref(a)
    x=torch.from_numpy(a);y=torch.nn.functional.adaptive_avg_pool2d(x,(5,3))
    assert not np.shares_memory(x._data,y._data)
    assert y._data.base is None, 'Output must not retain every packed prefix'
    y._data[...] = 3;np.testing.assert_array_equal(a,before)
    del x,a;gc.collect();assert ref() is None
    np.testing.assert_array_equal(y.numpy(),3)
    del y


@pytest.mark.parametrize('dtype', [np.float16,np.float32,np.float64])
@pytest.mark.parametrize('inverse', [False,True])
def test_plan_arrays_are_readonly_and_each_plan_is_bounded(dtype,inverse):
    p=pooling._cached_pack_plan(17,23,5,3,np.dtype(dtype).str,inverse)
    assert p is not None
    arrays=[v for v in p if isinstance(v,np.ndarray)]
    assert sum(v.nbytes for v in arrays)<=pooling._PLAN_CACHE_BYTES
    for v in arrays:
        assert not v.flags.writeable
        if isinstance(v.base,np.ndarray):assert not v.base.flags.writeable


def test_lru_plan_count_is_bounded():
    pooling._cached_pack_plan.cache_clear()
    for n in range(90):
        pooling._cached_pack_plan(17,23,2+n%11,2+n//11,np.dtype('float32').str,False)
    assert pooling._cached_pack_plan.cache_info().currsize<=64


def test_large_plan_is_rejected_before_index_construction(monkeypatch):
    pooling._cached_geometry(8192,8192,2,3)
    original=np.arange
    def guard(*args,**kwargs):
        assert args[0]<=2048,'Oversized packed indices must not be constructed'
        return original(*args,**kwargs)
    monkeypatch.setattr(np,'arange',guard)
    assert pooling._cached_pack_plan(8192,8192,2,3,np.dtype('float32').str,False) is None


def test_cached_metadata_rechecks_mutated_input_and_seed_aliases(monkeypatch):
    monkeypatch.setattr(pooling,'_ALGORITHM','packed');monkeypatch.setattr(pooling,'_BACKWARD_ALGORITHM','packed')
    a=np.ones((1,1,17,23));x=torch.from_numpy(a).requires_grad_()
    y=torch.nn.functional.adaptive_avg_pool2d(x,(5,3));raw=np.ones(y.shape);seed=torch.from_numpy(raw)
    y.backward(seed,retain_graph=True)
    a[...,0,0]=np.inf;raw[...,0,0]=np.nan;rows=[]
    for cached in [True,False]:
        monkeypatch.setattr(pooling,'_CACHE_PLANS',cached)
        x.grad=None;y.backward(seed,retain_graph=True)
        rows.append((torch.nn.functional.adaptive_avg_pool2d(x,(5,3)).numpy().copy(),x.grad.numpy().copy()))
    for left,right in zip(*rows):np.testing.assert_array_equal(left,right)


def test_cached_coordinate_fallback_when_flat_indices_exceed_cap(monkeypatch):
    dtype=np.dtype('float64');plan=pooling._cached_pack_plan(96,96,23,25,dtype.str,False)
    assert plan is not None and plan.flat is None
    raw=np.random.default_rng(4005).normal(size=(1,1,96,96));rows=[]
    monkeypatch.setattr(pooling,'_ALGORITHM','packed');monkeypatch.setattr(pooling,'_BACKWARD_ALGORITHM','packed')
    for cached in [False,True]:
        monkeypatch.setattr(pooling,'_CACHE_PLANS',cached)
        x=torch.tensor(raw,requires_grad=True);y=torch.nn.functional.adaptive_avg_pool2d(x,(23,25));y.sum().backward()
        rows.append((y.numpy().copy(),x.grad.numpy().copy()))
    for left,right in zip(*rows):np.testing.assert_array_equal(left,right)
