"""Native matrix/axis transform schemas, storage and reverse/forward graphs."""
import gc
import warnings
import weakref
import numpy as np
import pytest
import torch as native
from torchwasm import torch

FORMS=['t','mT','swapaxes','swapdims','movedim','moveaxis','torch.t','torch.swapaxes','torch.swapdims','torch.movedim','torch.moveaxis']


def transform(api,x,form):
    name=form.removeprefix('torch.')
    if name=='mT':return x.mT
    args=() if name=='t' else ((0,-1) if name in ('swapaxes','swapdims') else (((0,-1),(-1,0)) if x.ndim>1 else (0,0)))
    return getattr(api,name)(x,*args) if form.startswith('torch.') else getattr(x,name)(*args)


def values(shape,dtype,layout):
    a=np.arange(int(np.prod(shape)),dtype='float64').reshape(shape)
    if np.dtype(dtype).kind=='c':a=a+1j*(a+.5)
    a=np.asarray(a,dtype=dtype).reshape(shape)
    if layout=='F' and a.ndim:return np.asfortranarray(a)
    if layout=='step' and a.ndim:return np.repeat(a,2,axis=-1)[...,::2]
    return a


@pytest.mark.parametrize('form',FORMS)
@pytest.mark.parametrize('shape',[(),(3,),(2,3),(2,1,3),(0,2,3)])
@pytest.mark.parametrize('dtype',['float16','float32','float64','int32','bool','complex64','complex128'])
@pytest.mark.parametrize('layout',['C','F','step'])
def test_native_transform_values_strides_and_object_alias(form,shape,dtype,layout):
    a=values(shape,dtype,layout);x=torch.from_numpy(a);rx=native.from_numpy(a)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:expected=transform(native,rx,form)
        except RuntimeError:
            with pytest.raises(RuntimeError):transform(torch,x,form)
            return
        actual=transform(torch,x,form)
    np.testing.assert_array_equal(actual.numpy(),expected.numpy())
    assert actual.dtype==a.dtype and actual.shape==tuple(expected.shape) and actual.stride()==expected.stride()
    assert (actual is x)==(expected is rx)
    assert actual.data_ptr()==x.data_ptr() and actual._storage is x._storage


@pytest.mark.parametrize('form',FORMS)
@pytest.mark.parametrize('shape',[(),(3,),(2,3),(2,1,3)])
@pytest.mark.parametrize('dtype',['float16','float32','float64'])
def test_native_linear_vjp_repeated_after_shared_storage_mutation(form,shape,dtype):
    data=values(shape,dtype,'F');results=[]
    for api in (native,torch):
        x=api.from_numpy(data.copy(order='K')).requires_grad_()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:y=transform(api,x,form)
            except RuntimeError:
                assert (form.removeprefix('torch.')=='t' and len(shape)>2) or (form=='mT' and len(shape)==1)
                return
        seed=api.tensor(np.arange(y.numel(),dtype='float64').reshape(y.shape).astype(dtype)/8+.5)
        y.backward(seed);first=x.grad
        with api.no_grad():y.detach().add_(1)
        y.backward(seed)
        assert x.grad is first
        results.append(x.grad.numpy())
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize('form',FORMS)
@pytest.mark.parametrize('shape',[(),(3,),(2,3),(2,1,3)])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('tangent_layout',['C','F','step'])
def test_native_jvp_layout_and_reverse_over_tangent(form,shape,dtype,tangent_layout):
    if (form.removeprefix('torch.')=='t' and len(shape)>2) or (form=='mT' and len(shape)==1):return
    primal=values(shape,dtype,'F');tangent=values(shape,dtype,tangent_layout);outputs=[]
    for api in (native,torch):
        p=api.from_numpy(primal.copy(order='K')).requires_grad_();t=api.from_numpy(tangent).requires_grad_()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');a,b=api.func.jvp(lambda z:transform(api,z,form),(p,),(t,))
        g=api.autograd.grad(b.square().sum(),(p,t),allow_unused=True)
        outputs.append((a,b,g))
    for expected,actual in zip(outputs[0][:2],outputs[1][:2]):
        np.testing.assert_array_equal(actual.detach().numpy(),expected.detach().numpy());assert actual.stride()==expected.stride()
    for expected,actual in zip(outputs[0][2],outputs[1][2]):
        assert (actual is None)==(expected is None)
        if actual is not None:np.testing.assert_array_equal(actual.numpy(),expected.numpy())


class Index:
    def __index__(self):return 0


def axis_cases(api):
    return [(0,0),(-1,-1),((),()),((0,),(-1,)),((0,1),(1,)),((0,1),(0,0)),
            ((0,-1),(0,-1)),(0,(0,)),((0,),0),(True,0),((True,),(0,)),(None,0),
            (1.2,0),(np.int64(0),np.int64(0)),(np.array(0),0),((np.array(0),),(0,)),
            (np.array([0]),(0,)),(api.tensor(0),0),(api.tensor(True),0),
            (api.tensor([0]),0),((api.tensor([0]),),(0,)),((api.tensor(True),),(0,)),
            (Index(),0),((Index(),),(0,)),((0,0),(0,0)),((0,-1),(0,1)),((0,99),(0,0)),
            ((0,1.5),(0,)),([],[]),([0,1],[1,0]),(2**63,0),((2**63,),(0,)),
            (-2**63-1,0),(np.uint64(2**63),0)]


@pytest.mark.parametrize('shape',[(),(3,),(2,1,3)])
@pytest.mark.parametrize('case',range(34))
@pytest.mark.parametrize('form',['movedim','moveaxis','torch.movedim','torch.moveaxis'])
def test_native_movement_overloads_and_error_priority(shape,case,form):
    x=torch.ones(shape);rx=native.ones(shape);a,b=axis_cases(torch)[case];ra,rb=axis_cases(native)[case]
    name=form.removeprefix('torch.');call=lambda api,x,a,b:getattr(api,name)(x,a,b) if form.startswith('torch.') else getattr(x,name)(a,b)
    try:expected=call(native,rx,ra,rb)
    except (TypeError,IndexError,RuntimeError,ValueError) as error:
        with pytest.raises(type(error)):call(torch,x,a,b)
        return
    actual=call(torch,x,a,b);assert actual.stride()==expected.stride() and actual.shape==tuple(expected.shape)


@pytest.mark.parametrize('form',FORMS)
@pytest.mark.parametrize('shape',[(),(3,),(2,3),(2,1,3)])
def test_native_no_grad_view_flags(form,shape):
    if (form.removeprefix('torch.')=='t' and len(shape)>2) or (form=='mT' and len(shape)==1):return
    results=[]
    for api in (native,torch):
        x=api.ones(shape,requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with api.no_grad():y=transform(api,x,form)
        results.append((y is x,y.requires_grad,y.is_leaf,y.stride()))
    assert results[0]==results[1]


def test_native_alias_keyword_names_and_invalid_functional_inputs():
    for api in (native,torch):
        x=api.ones(2,3,4)
        assert x.swapaxes(axis0=0,axis1=1).shape==(3,2,4)
        assert api.swapaxes(input=x,axis0=0,axis1=1).shape==(3,2,4)
        assert api.swapdims(input=x,dim0=0,dim1=1).shape==(3,2,4)
        assert api.movedim(input=x,source=[0,1],destination=[2,0]).shape==(3,4,2)
        with pytest.raises(TypeError):x.swapaxes(dim0=0,dim1=1)
        with pytest.raises(TypeError):x.swapdims(axis0=0,axis1=1)
        with pytest.raises(TypeError):x.t(0)
        for name in ('t','movedim','moveaxis','swapaxes','swapdims'):
            args=() if name=='t' else (0,0)
            with pytest.raises(TypeError):getattr(api,name)([1.,2.],*args)


def test_axis_lists_and_tensor_metadata_are_snapshotted_without_retaining_payloads():
    class Axes(list):pass
    source=Axes([0,1]);source.payload=np.zeros(1024);reference=weakref.ref(source.payload)
    target=[2,0];x=torch.ones(2,3,4,requires_grad=True);y=x.movedim(source,target)
    source[:]=[1,2];target[:]=[0,2];del source;gc.collect();assert reference() is None
    seed=torch.arange(y.numel(),dtype=torch.float32).reshape(y.shape);y.backward(seed)
    expected=np.moveaxis(seed.numpy(),[2,0],[0,1]);np.testing.assert_array_equal(x.grad.numpy(),expected)
    x.grad=None;axis=torch.tensor(0);ref=weakref.ref(axis._data);value=x.moveaxis(axis,-1)
    axis.fill_(1);del axis;gc.collect();assert ref() is None
    value.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),np.ones(x.shape))


@pytest.mark.parametrize('shape',[(2,3,4,2),(2,3,2,2,2)])
@pytest.mark.parametrize('form',['mT','movedim','moveaxis','torch.movedim','torch.moveaxis'])
def test_native_multiple_batch_axes_preserve_unmoved_order_and_dual_layout(shape,form):
    data=np.asfortranarray(np.arange(np.prod(shape),dtype='float64').reshape(shape)/8)
    tangent=np.ones(shape,dtype='float64')*.3;outputs=[]
    for api in (native,torch):
        x=api.from_numpy(data).requires_grad_();t=api.from_numpy(tangent).requires_grad_()
        def function(z):
            if form=='mT':return z.mT
            args=((0,2),(-1,1))
            return getattr(api,form.removeprefix('torch.'))(z,*args) if form.startswith('torch.') else getattr(z,form)(*args)
        p,q=api.func.jvp(function,(x,),(t,))
        seed=api.arange(p.numel(),dtype=api.float64).reshape(p.shape)/16
        g=api.autograd.grad(p,x,grad_outputs=seed)[0]
        outputs.append((p,q,g))
    for expected,actual in zip(*outputs):
        assert actual.stride()==expected.stride()
        np.testing.assert_array_equal(actual.detach().numpy(),expected.detach().numpy())


@pytest.mark.parametrize('shape',[(),(3,),(2,3),(2,3,4,2)])
@pytest.mark.parametrize('form',['transpose','swapaxes','swapdims','torch.transpose','torch.swapaxes','torch.swapdims'])
@pytest.mark.parametrize('case',range(12))
def test_native_transpose_scalar_axis_schema_and_exception_priority(shape,form,case):
    def arguments(api):
        return [(Index(),0),(np.array(0),1),(api.tensor([0]),1),(api.tensor(True),0),
                (np.int64(0),-1),(api.tensor(0),-1),(99,1.5),(99,api.tensor(True)),
                (99,2**63),(2**63,99),(-2**63-1,0),(True,0)][case]
    x=torch.ones(shape);rx=native.ones(shape);name=form.removeprefix('torch.')
    call=lambda api,x,args:getattr(api,name)(x,*args) if form.startswith('torch.') else getattr(x,name)(*args)
    try:expected=call(native,rx,arguments(native))
    except (TypeError,IndexError,RuntimeError,ValueError) as error:
        with pytest.raises(type(error)):call(torch,x,arguments(torch))
        return
    actual=call(torch,x,arguments(torch));assert actual.stride()==expected.stride()
    np.testing.assert_array_equal(actual.numpy(),expected.numpy())
