"""Native contracts for true storage views, layout queries and forward tapes."""
import itertools
import math
import warnings
import numpy as np
import pytest
import torch as native
from stridetorch import torch
from stridetorch import UnsupportedOperation


def view_cases():
    cases=[]
    for shape in [(),(0,),(1,),(4,),(0,3),(1,4),(2,3),(2,3,4),(2,1,3),(0,2,3),(1,0,3)]:
        base=np.arange(math.prod(shape),dtype=np.float64).reshape(shape)
        layouts=[base,np.asfortranarray(base)]
        if base.ndim:
            layouts += [base.transpose(p) for p in itertools.permutations(range(base.ndim))]
            for axis in range(base.ndim):
                key=[slice(None)]*base.ndim;key[axis]=slice(None,None,2);layouts.append(base[tuple(key)])
            if base.size:layouts.append(np.broadcast_to(base.reshape(-1)[:1],(2,3)))
        for data in layouts:
            n=data.size;targets=[(),(n,),(-1,),(1,n),(n,1),(1,1,n),(n,1,1),data.shape,(0,-1),(-1,-1),(n+1,),(2,-1),(-1,2),(1,-1),(-1,1)]
            if n and n%2==0:targets += [(2,n//2),(1,2,n//2),(n//2,2,1)]
            cases.extend((data,target) for target in targets)
    return cases


@pytest.mark.parametrize('data,target',view_cases())
def test_native_view_shape_values_strides_and_alias(data,target):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');x=torch.from_numpy(data);reference=native.from_numpy(data)
    try:expected=reference.view(target)
    except (RuntimeError,TypeError) as error:
        with pytest.raises(type(error)):x.view(target)
        return
    y=x.view(target)
    np.testing.assert_array_equal(y.numpy(),expected.numpy())
    assert y.shape==tuple(expected.shape) and y.stride()==expected.stride()
    assert y.element_size()==expected.element_size()
    assert y._storage is x._storage
    assert y.data_ptr()==x.data_ptr()
    for axis in range(-y.ndim,y.ndim):assert y.stride(axis)==expected.stride(axis)


def layout_values(kind,dtype):
    a=np.arange(24,dtype=dtype).reshape(2,3,4)/17
    if kind=='step':return np.repeat(a,2,axis=-1)[...,::2]
    if kind=='fortran':return np.asfortranarray(a)
    return a.copy()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['contiguous','step','fortran'])
@pytest.mark.parametrize('method',['view','view_as'])
def test_native_reverse_view_and_retained_seeds(dtype,layout,method):
    data=layout_values(layout,dtype);target=(3,2,4) if layout!='fortran' else data.shape
    x=torch.from_numpy(data).requires_grad_();rx=native.from_numpy(data).requires_grad_()
    other=torch.empty(target);rother=native.empty(target)
    y=x.view(target) if method=='view' else x.view_as(other)
    ry=rx.view(target) if method=='view' else rx.view_as(rother)
    for scale in [1.,-.3]:
        seed=np.linspace(-.4,.8,y.numel(),dtype=dtype).reshape(target)*scale
        y.backward(torch.from_numpy(seed),retain_graph=True);ry.backward(native.from_numpy(seed),retain_graph=True)
        np.testing.assert_allclose(x.grad.numpy(),rx.grad.numpy(),rtol=2e-6,atol=1e-7)
    assert y._storage is x._storage


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('primal_layout',['contiguous','step','fortran'])
@pytest.mark.parametrize('tangent_layout',['contiguous','step','fortran'])
def test_native_jvp_layout_and_reverse(dtype,primal_layout,tangent_layout):
    a=layout_values(primal_layout,dtype)
    b=layout_values(tangent_layout,dtype);b[...]*=.7
    x=torch.from_numpy(a).requires_grad_();t=torch.from_numpy(b).requires_grad_()
    rx=native.from_numpy(a).requires_grad_();rt=native.from_numpy(b).requires_grad_()
    target=(-1,) if primal_layout!='fortran' else a.shape
    p,q=torch.func.jvp(lambda z:z.view(target),(x,),(t,))
    rp,rq=native.func.jvp(lambda z:z.view(target),(rx,),(rt,))
    np.testing.assert_array_equal(p.numpy(),rp.detach().numpy());np.testing.assert_array_equal(q.numpy(),rq.detach().numpy())
    assert p.stride()==rp.stride() and q.stride()==rq.stride()
    assert (q.data_ptr()==t.data_ptr())==(rq.data_ptr()==rt.data_ptr())
    g=torch.autograd.grad(p.square().sum()+q.square().sum(),(x,t))
    rg=native.autograd.grad(rp.square().sum()+rq.square().sum(),(rx,rt))
    for actual,expected in zip(g,rg):np.testing.assert_allclose(actual.numpy(),expected.numpy(),rtol=2e-6,atol=1e-7)


def test_view_alias_mutation_invalidates_upstream_saved_values():
    x=torch.arange(12,dtype=torch.float64).requires_grad_();saved=x.square().sum();y=x.view(3,4)
    version=x._version
    with torch.no_grad():y.detach()[1,2]=20.
    assert x[6].item()==20. and x._version==y._version==version+1
    with pytest.raises(UnsupportedOperation):saved.backward()


def test_view_does_not_guard_values_unneeded_for_constant_derivative():
    x=torch.arange(12,dtype=torch.float64).requires_grad_();y=x.view(3,4)
    with torch.no_grad():x[0]=100.;y.detach()[1,1]=200.
    y.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),np.ones(12))


def test_view_refuses_copy_but_reshape_copies():
    x=torch.arange(12,dtype=torch.float64).reshape(3,4).T
    with pytest.raises(RuntimeError):x.view(-1)
    flat=x.reshape(-1);assert flat.data_ptr()!=x.data_ptr()
    original=x.numpy().copy();flat[0]=-999.;np.testing.assert_array_equal(x.numpy(),original)


@pytest.mark.parametrize('shape',[(0,),(0,3),(2,0,4),(1,3),(2,3,4)])
@pytest.mark.parametrize('factory',['empty','zeros','ones','rand','randn'])
def test_native_factory_stride(factory,shape):
    x=getattr(torch,factory)(shape);rx=getattr(native,factory)(shape)
    assert x.stride()==rx.stride()


@pytest.mark.parametrize('dim',[False,True,1.2,'0',10,-10])
def test_native_stride_argument_rejections(dim):
    x=torch.ones(2,3);rx=native.ones(2,3)
    with pytest.raises((TypeError,IndexError)) as expected:rx.stride(dim)
    with pytest.raises(type(expected.value)):x.stride(dim)


def test_view_keywords_scalar_inference_and_dtype_boundary():
    x=torch.arange(6.);assert x.view(size=(2,3)).shape==(2,3)
    assert torch.tensor(2.).view(()).shape==()
    assert torch.empty(0,3).view(-1).shape==(0,)
    for args in [(),(True,6),(2.,3),(None,)]:
        with pytest.raises(TypeError):x.view(*args)
    with pytest.raises(TypeError):x.view(2,3,size=(2,3))
    with pytest.raises(TypeError):x.view_as([2,3])
    with pytest.raises(UnsupportedOperation):x.view(dtype=torch.int32)


@pytest.mark.parametrize('shape',[(0,),(0,3),(2,0,4)])
@pytest.mark.parametrize('factory',['full','randint'])
def test_other_empty_factories_keep_native_strides(shape,factory):
    args=(shape,2.) if factory=='full' else (0,5,shape)
    assert getattr(torch,factory)(*args).stride()==getattr(native,factory)(*args).stride()


@pytest.mark.parametrize('data',[[],[[],[]]])
def test_empty_sequence_factories_keep_native_strides(data):
    for name in ['tensor','as_tensor']:
        assert getattr(torch,name)(data).stride()==getattr(native,name)(data).stride()


def test_other_dense_empty_sources():
    for api in [torch,native]:
        assert api.arange(0).stride()==(1,)
        assert api.linspace(0.,1.,0).stride()==(1,)
        assert api.eye(0,3).stride()==(3,1)


def test_empty_randperm_stride_and_no_rng_consumption():
    for api in [torch,native]:
        with api.random.fork_rng():
            api.manual_seed(173)
            expected=api.randperm(7).tolist()
            api.manual_seed(173)
            empty=api.randperm(0)
            assert empty.shape==(0,) and empty.stride()==(1,)
            assert api.randperm(7).tolist()==expected
