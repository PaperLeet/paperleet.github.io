"""Native shape/axis/value/stride/alias and reverse/forward contracts."""
import itertools
import warnings
import numpy as np
import pytest
import torch as native
from paperleet_torch import torch


def boundary_cases():
    rows=[]
    for shape in [(),(1,),(2,3),(2,1,3),(0,1,3)]:
        a=np.arange(np.prod(shape,dtype=int),dtype=np.float64).reshape(shape);n=len(shape)
        dims=[-n-2,-n-1,-1,0,1,n-1,n,n+1,True,1.2,None]
        for lo,hi in itertools.product(dims[:8],dims[:8]):rows += [(a,'flatten',(lo,hi)),(a,'transpose',(lo,hi))]
        for d in dims:rows += [(a,'unsqueeze',(d,)),(a,'squeeze',(d,))]
        rows += [(a,'squeeze',((0,1),)),(a,'squeeze',((0,0),)),(a,'squeeze',((),)),(a,'permute',()),(a,'permute',(tuple(range(n)),)),(a,'permute',(tuple(reversed(range(n))),)),(a,'permute',((n,*range(1,n)),)),(a,'permute',((True,*range(1,n)),)),(a,'permute',((0,)*n,)),(a,'expand',((-1,)*(n+1),)),(a,'expand',((1,*shape),))]
    return rows


@pytest.mark.parametrize('data,method,args',boundary_cases())
def test_native_shape_boundary_matrix(data,method,args):
    x=torch.from_numpy(data);rx=native.from_numpy(data)
    try:expected=getattr(rx,method)(*args)
    except (TypeError,IndexError,RuntimeError) as error:
        with pytest.raises(type(error)):getattr(x,method)(*args)
        return
    actual=getattr(x,method)(*args)
    np.testing.assert_array_equal(actual.numpy(),expected.numpy())
    assert actual.shape==tuple(expected.shape) and actual.stride()==expected.stride()
    assert (actual is x)==(expected is rx)
    assert actual.data_ptr()==x.data_ptr()


def values(shape,layout,dtype):
    a=(np.arange(np.prod(shape,dtype=int),dtype=dtype)/13).reshape(shape)
    if layout=='fortran':return np.asfortranarray(a) if a.ndim else a.copy()
    if layout=='step' and a.ndim:return np.repeat(a,2,axis=-1)[...,::2]
    if layout=='singleton' and a.ndim:
        strides=list(a.strides)
        for k,n in enumerate(a.shape):
            if n==1:strides[k]=0
        return np.lib.stride_tricks.as_strided(a,shape=a.shape,strides=strides)
    return a


def transform(x,name):
    n=x.ndim
    if name=='flatten':return x.flatten()
    if name=='permute':return x.permute(tuple(reversed(range(n))))
    if name=='transpose':return x.transpose(0,-1)
    if name=='unsqueeze':return x.unsqueeze(0)
    if name=='squeeze':return x.squeeze()
    if name=='squeeze_tuple':return x.squeeze(tuple(range(n)))
    return x.expand(2,*x.shape)


@pytest.mark.parametrize('shape',[(),(1,),(2,3),(2,1,3),(0,1,3)])
@pytest.mark.parametrize('layout',['C','fortran','step','singleton'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('method',['flatten','permute','transpose','unsqueeze','squeeze','squeeze_tuple','expand'])
def test_native_shape_workflow_values_metadata_and_vjp(shape,layout,dtype,method):
    data=values(shape,layout,dtype);x=torch.from_numpy(data).requires_grad_();rx=native.from_numpy(data).requires_grad_()
    y=transform(x,method);ry=transform(rx,method)
    np.testing.assert_array_equal(y.numpy(),ry.detach().numpy())
    assert y.stride()==ry.stride() and (y is x)==(ry is rx)
    for scale in [1.,-.7]:
        g=((np.arange(y.numel(),dtype=dtype)+.1)*scale).reshape(y.shape)
        y.backward(torch.from_numpy(g),retain_graph=True);ry.backward(native.from_numpy(g),retain_graph=True)
        np.testing.assert_allclose(x.grad.numpy(),rx.grad.numpy(),rtol=3e-6,atol=2e-7)


@pytest.mark.parametrize('shape',[(2,3),(2,1,3),(0,1,3)])
@pytest.mark.parametrize('primal_layout',['C','fortran','step'])
@pytest.mark.parametrize('tangent_layout',['C','fortran','step'])
@pytest.mark.parametrize('method',['flatten','permute','transpose','unsqueeze','squeeze_tuple','expand'])
def test_native_shape_jvp_values_metadata_and_reverse(shape,primal_layout,tangent_layout,method):
    a=values(shape,primal_layout,np.float64);b=values(shape,tangent_layout,np.float64)
    x=torch.from_numpy(a).requires_grad_();t=torch.from_numpy(b).requires_grad_();rx=native.from_numpy(a).requires_grad_();rt=native.from_numpy(b).requires_grad_()
    p,q=torch.func.jvp(lambda z:transform(z,method),(x,),(t,));rp,rq=native.func.jvp(lambda z:transform(z,method),(rx,),(rt,))
    np.testing.assert_array_equal(p.numpy(),rp.detach().numpy());np.testing.assert_array_equal(q.numpy(),rq.detach().numpy())
    assert p.stride()==rp.stride() and q.stride()==rq.stride()
    g=torch.autograd.grad(q.square().sum(),(x,t),allow_unused=True);rg=native.autograd.grad(rq.square().sum(),(rx,rt),allow_unused=True)
    for actual,expected in zip(g,rg):
        assert (actual is None)==(expected is None)
        if actual is not None:np.testing.assert_allclose(actual.numpy(),expected.numpy(),rtol=1e-12,atol=1e-13)


@pytest.mark.parametrize('name,args',[('flatten',()),('transpose',(0,1)),('unsqueeze',(1,)),('squeeze',()),('squeeze',((0,1),)),('permute',((1,0),)),('broadcast_to',((2,3),))])
def test_native_functional_shape_entries(name,args):
    x=torch.arange(3,dtype=torch.float64).reshape(1,3).requires_grad_();rx=native.arange(3,dtype=native.float64).reshape(1,3).requires_grad_()
    y=getattr(torch,name)(x,*args);ry=getattr(native,name)(rx,*args)
    np.testing.assert_array_equal(y.numpy(),ry.detach().numpy());assert y.stride()==ry.stride()
    y.square().sum().backward();ry.square().sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(),rx.grad.numpy())
    with pytest.raises(TypeError):getattr(torch,name)([1,2,3],*args)


def test_keywords_duplicates_and_empty_permutations():
    x=torch.ones(2,3)
    assert x.permute(dims=(1,0)).shape==(3,2)
    assert torch.permute(x,dims=(1,0)).shape==(3,2)
    assert torch.ones(1,3).expand(size=(2,3),implicit=True).stride()==(0,1)
    with pytest.raises(TypeError):x.permute((1,0),dims=(1,0))
    with pytest.raises(TypeError):torch.permute(x,1,0)
    with pytest.raises(TypeError):torch.permute(torch.ones(2),0)
    with pytest.raises(TypeError):torch.ones(1,3).expand((2,3),size=(2,3))
    with pytest.raises(TypeError):torch.ones(1,3).expand(2,3,implicit=1)
    scalar=torch.tensor(2.);assert scalar.permute(()).shape==()


@pytest.mark.parametrize('shape',[(),(3,),(2,3),(2,1,3)])
@pytest.mark.parametrize('no_grad',[False,True])
def test_native_T_alias_object_and_grad_mode(shape,no_grad):
    data=np.arange(np.prod(shape,dtype=int),dtype=np.float64).reshape(shape)
    x=torch.from_numpy(data).requires_grad_();rx=native.from_numpy(data).requires_grad_()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if no_grad:
            with torch.no_grad():y=x.T
            with native.no_grad():ry=rx.T
        else:y=x.T;ry=rx.T
    assert (y is x)==(ry is rx) and y.requires_grad==ry.requires_grad and y.is_leaf==ry.is_leaf
    assert y.stride()==ry.stride();np.testing.assert_array_equal(y.numpy(),ry.detach().numpy())


@pytest.mark.parametrize('kind',['leaf','frozen_base','grad_base','detached','fortran'])
@pytest.mark.parametrize('method',['flatten','permute','transpose','unsqueeze','squeeze_tuple','expand'])
def test_native_no_grad_view_base_flags(kind,method):
    def make(api):
        if kind=='leaf':return api.from_numpy(np.arange(6.,dtype=np.float64).reshape(2,3)).requires_grad_()
        if kind=='fortran':return api.from_numpy(np.asfortranarray(np.arange(6.,dtype=np.float64).reshape(2,3))).requires_grad_()
        base=api.arange(6,dtype=api.float64)
        if kind=='grad_base':base.requires_grad_()
        view=base.reshape(2,3)
        if kind=='detached':view=view.detach()
        return view.requires_grad_()
    x=make(torch);rx=make(native)
    with torch.no_grad():y=transform(x,method)
    with native.no_grad():ry=transform(rx,method)
    assert y.requires_grad==ry.requires_grad and y.is_leaf==ry.is_leaf
    np.testing.assert_array_equal(y.numpy(),ry.detach().numpy())
    if y.requires_grad:
        (y*2).sum().backward();(ry*2).sum().backward()
        assert (y.grad is None)==(ry.grad is None)
        if y.grad is not None:np.testing.assert_array_equal(y.grad.numpy(),ry.grad.numpy())
        assert x.grad is None and rx.grad is None


@pytest.mark.parametrize('basegrad',[False,True])
@pytest.mark.parametrize('changes',[(True,),(False,),(True,False),(False,True)])
def test_native_no_grad_view_explicit_flag_changes(basegrad,changes):
    x=torch.arange(3,dtype=torch.float64).requires_grad_(basegrad);rx=native.arange(3,dtype=native.float64).requires_grad_(basegrad)
    with torch.no_grad():y=x.view(3)
    with native.no_grad():ry=rx.view(3)
    for flag in changes:
        y.requires_grad_(flag);ry.requires_grad_(flag)
        assert y.requires_grad==ry.requires_grad
    if y.requires_grad:
        y.square().sum().backward();ry.square().sum().backward()
        assert (y.grad is None)==(ry.grad is None)
        if y.grad is not None:np.testing.assert_array_equal(y.grad.numpy(),ry.grad.numpy())
        assert x.grad is None and rx.grad is None


@pytest.mark.parametrize('recording',[False,True])
@pytest.mark.parametrize('initial',[False,True])
@pytest.mark.parametrize('promote',[False,True])
@pytest.mark.parametrize('changes',[(True,),(False,),(True,False),(False,True)])
def test_native_graphless_view_tracks_base_flags_without_reconnecting(recording,initial,promote,changes):
    def exercise(api):
        base=api.from_numpy(np.arange(6.,dtype=np.float64)).requires_grad_(initial)
        with api.set_grad_enabled(recording):view=base.view(2,3)
        if promote:view.requires_grad_()
        steps=[]
        for flag in changes:
            base.requires_grad_(flag)
            steps.append((base.requires_grad,view.requires_grad,view.is_leaf))
        out=view.square().sum()
        if out.requires_grad:out.backward()
        return (steps,out.requires_grad,
                None if base.grad is None else base.grad.tolist(),
                None if view.grad is None else view.grad.tolist())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        assert exercise(torch)==exercise(native)


@pytest.mark.parametrize('initial',[False,True])
@pytest.mark.parametrize('recording',[False,True])
@pytest.mark.parametrize('flag',[True,False,1,None])
def test_native_view_direct_requires_grad_assignment(initial,recording,flag):
    def exercise(api):
        base=api.from_numpy(np.arange(3.)).requires_grad_(initial)
        with api.set_grad_enabled(recording):view=base.view(3)
        try:view.requires_grad=flag
        except RuntimeError:return 'error'
        base.requires_grad=not initial
        result=view.square().sum()
        if result.requires_grad:result.backward()
        return (view.requires_grad,view.is_leaf,
                None if base.grad is None else base.grad.tolist(),
                None if view.grad is None else view.grad.tolist())
    assert exercise(torch)==exercise(native)


@pytest.mark.parametrize('flag',[1,None,'True',np.bool_(True)])
def test_native_requires_grad_method_rejects_nonbool(flag):
    for api in (torch,native):
        with pytest.raises(TypeError):api.ones(2).requires_grad_(flag)


def test_released_nonleaf_gradient_flags_keep_native_leaf_rules():
    for api in (torch,native):
        x=api.ones(2,requires_grad=True);y=x.square();y.sum().backward()
        for flag in (False,True):
            with pytest.raises(RuntimeError):y.requires_grad=flag
        assert y.requires_grad_(True) is y
        with pytest.raises(RuntimeError):y.requires_grad_(False)
