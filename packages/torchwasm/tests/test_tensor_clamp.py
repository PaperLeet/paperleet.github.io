"""Tensor-bound clamp contracts from native2.6 observations and finite differences."""
import gc
import weakref
import warnings
import numpy as np
import pytest
from torchwasm import torch


def exact(value, expected):
    actual=value.detach().numpy();expected=np.asarray(expected,dtype=actual.dtype)
    np.testing.assert_array_equal(actual,expected)
    if actual.dtype.kind=='f':np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


def layout(values,dtype,name):
    data=np.array(values,dtype=dtype)
    if name=='F':return np.asfortranarray(data)
    if name=='strided':
        storage=np.empty(tuple(size*2 for size in data.shape),dtype=dtype);view=storage[tuple(slice(None,None,2) for _ in data.shape)];view[...] = data;return view
    if name=='negative':return np.array(data[...,::-1],copy=True)[...,::-1]
    if name=='readonly':data.flags.writeable=False
    return data


def tensor(array,requires_grad=True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        return torch.from_numpy(array).requires_grad_(requires_grad)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('kind',['C','F','strided','readonly'])
@pytest.mark.parametrize('interface',['method','function'])
@pytest.mark.parametrize('nonfinite',[False,True])
def test_three_owner_broadcast_vjp_and_readonly_seed(dtype,kind,interface,nonfinite):
    arrays=[layout(v,dtype,kind) for v in [[[-2.,-.5,.5],[2.,0.,-2.]],[[-1.],[-1.]],[1.,1.,1.]]]
    saved=[a.copy() for a in arrays];x,lo,hi=map(tensor,arrays)
    seed=np.array([[np.nan,-0.,np.inf],[np.nan,2.,-np.inf]],dtype=dtype) if nonfinite else np.ones((2,3),dtype=dtype)
    seed.flags.writeable=False
    y=x.clamp(lo,hi) if interface=='method' else torch.clamp(input=x,min=lo,max=hi)
    gx,gl,gu=torch.autograd.grad(y,(x,lo,hi),tensor(seed,False))
    exact(y,[[-1.,-.5,.5],[1.,0.,-1.]])
    if nonfinite:
        exact(gx,[[0.,-0.,np.inf],[0.,2.,0.]])
        exact(gl,[[np.nan],[-np.inf]]);exact(gu,[np.nan,0.,0.])
    else:
        exact(gx,[[0.,1.,1.],[0.,1.,0.]])
        exact(gl,[[1.],[1.]]);exact(gu,[1.,0.,0.])
    for array,before in zip(arrays,saved):np.testing.assert_array_equal(array,before)
    assert all(owner.grad is None for owner in (x,lo,hi))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('kind',['C','strided','readonly'])
@pytest.mark.parametrize('side',['min','max'])
@pytest.mark.parametrize('interface',['method','function'])
def test_one_tensor_bound_methods_and_broadcast_reduction(dtype,kind,side,interface):
    x=tensor(layout([-2.,-1.,0.,1.,2.,np.nan],dtype,kind));bound=tensor(np.array([-1. if side=='min' else 1.],dtype=dtype))
    operation=getattr(x,'clamp_'+side) if interface=='method' else getattr(torch,'clamp_'+side)
    y=operation(bound) if interface=='method' else operation(x,bound)
    gx,gb=torch.autograd.grad(y,(x,bound),torch.ones_like(y))
    exact(gx,[0.,1.,1.,1.,1.,0.] if side=='min' else [1.,1.,1.,1.,0.,0.]);exact(gb,[1.])


@pytest.mark.parametrize('target',['lower','upper','both'])
@pytest.mark.parametrize('method',['grad','backward_inputs'])
def test_target_pruning_and_captured_gradient_activity(target,method):
    x=torch.tensor([-2.,0.,2.],requires_grad=True);lo=torch.tensor([-1.],requires_grad=True);hi=torch.tensor([1.],requires_grad=True);y=x.clamp(lo,hi)
    requested=(lo,hi) if target=='both' else (lo,) if target=='lower' else (hi,)
    if method=='grad':
        gradients=torch.autograd.grad(y,requested,torch.ones_like(y));assert all(v.grad is None for v in (x,lo,hi))
    else:
        y.backward(torch.ones_like(y),inputs=requested);gradients=tuple(v.grad for v in requested);assert x.grad is None
        assert (lo.grad is not None)==any(v is lo for v in requested) and (hi.grad is not None)==any(v is hi for v in requested)
    for value in gradients:exact(value,[1.])


def test_expanded_input_and_zero_stride_numpy_bounds():
    x=torch.tensor([-2.,0.,2.],dtype=torch.float64,requires_grad=True)
    lo=tensor(np.broadcast_to(np.array(-1.,dtype=np.float64),(2,1)))
    hi=tensor(np.broadcast_to(np.array(1.,dtype=np.float64),(2,1)))
    y=x.clamp(lo,hi);gx,gl,gu=torch.autograd.grad(y,(x,lo,hi),torch.ones_like(y))
    assert y.shape==(2,3);exact(gx,[0.,2.,0.]);exact(gl,[[1.],[1.]]);exact(gu,[[1.],[1.]])


@pytest.mark.parametrize('which',['source','shared_bounds'])
def test_duplicate_input_edges_accumulate_once_per_role(which):
    x=torch.tensor([-1.,0.,1.],requires_grad=True);bound=torch.tensor([0.],requires_grad=True)
    if which=='source':
        result=x.clamp(x,bound);gx,gb=torch.autograd.grad(result,(x,bound),torch.ones_like(result));exact(gx,[1.,1.,0.]);exact(gb,[1.])
    else:
        result=x.clamp(bound,bound);gx,gb=torch.autograd.grad(result,(x,bound),torch.ones_like(result));exact(gx,[0.,1.,0.]);exact(gb,[1.])


@pytest.mark.parametrize('upper_scalar',[True,False])
def test_zero_dimensional_promotion_and_backward_comparison_precision(upper_scalar):
    x=torch.tensor([1.],dtype=torch.float32,requires_grad=True);lo=torch.tensor(1.+2**-30,dtype=torch.float64,requires_grad=True);hi=torch.tensor(2. if upper_scalar else [2.],dtype=torch.float64,requires_grad=True)
    y=x.clamp(lo,hi);gx,gl,gu=torch.autograd.grad(y,(x,lo,hi),torch.ones_like(y))
    assert y.dtype==(torch.float32 if upper_scalar else torch.float64)
    exact(y,[1. if upper_scalar else 1.+2**-30]);exact(gx,[1.]);exact(gl,0.);exact(gu,0. if upper_scalar else [0.])


def test_integral_constant_input_with_learned_floating_bounds():
    x=torch.tensor([-2,0,2]);lo=torch.tensor(-1.,requires_grad=True);hi=torch.tensor([1.],requires_grad=True);y=x.clamp(lo,hi)
    assert y.dtype==torch.float32
    gl,gu=torch.autograd.grad(y,(lo,hi),torch.ones_like(y));exact(gl,1.);exact(gu,[1.])
    values,tangent=torch.func.jvp(lambda a,b:x.clamp(a,b),(lo,hi),(torch.tensor(3.),torch.tensor([4.])));exact(values,[-1.,0.,1.]);exact(tangent,[3.,0.,4.])


@pytest.mark.parametrize('target',['input','lower','upper','frozen_upper','output','downstream'])
def test_saved_storage_version_guards(target):
    x=torch.tensor([-.5,.5],requires_grad=True);lo=torch.tensor([-1.],requires_grad=True);hi=torch.tensor([1.],requires_grad=target!='frozen_upper')
    y=x.clamp(lo,hi);loss=y.square().sum() if target=='downstream' else y.sum()
    owner={'input':x,'lower':lo,'upper':hi,'frozen_upper':hi,'output':y,'downstream':y}[target]
    with torch.no_grad():owner.detach().add_(.25)
    if target=='output':loss.backward();exact(x.grad,[1.,1.]);exact(lo.grad,[0.]);exact(hi.grad,[0.])
    else:
        with pytest.raises(RuntimeError,match='modified'):loss.backward()
        assert x.grad is lo.grad is hi.grad is None


@pytest.mark.parametrize('retain',[False,True])
def test_raw_alias_edits_survive_dropped_wrappers_and_release_after_consumption(retain):
    x=torch.tensor([-2.125,-.625,.375,1.875],dtype=torch.float64,requires_grad=True);l=torch.tensor([-1.125],dtype=torch.float64,requires_grad=True);h=torch.tensor([.875],dtype=torch.float64,requires_grad=True)
    middle=x+.125;lo=l+.125;hi=h+.125;wrappers=[weakref.ref(v) for v in (middle,lo,hi)];arrays=[weakref.ref(v._data) for v in (middle,lo,hi)]
    y=middle.clamp(lo,hi);a=middle.detach().numpy();b=lo.detach().numpy();c=hi.detach().numpy();a[:]=[-3.,0.,3.,4.];b[:]=-2.;c[:]=2.;del middle,lo,hi,a,b,c;gc.collect()
    assert all(v() is None for v in wrappers) and all(v() is not None for v in arrays)
    for keep in ([True,False] if retain else [False]):
        gradients=torch.autograd.grad(y,(x,l,h),torch.tensor([2.,3.,5.,7.],dtype=torch.float64),retain_graph=keep)
        for value,expected in zip(gradients,[[0.,3.,0.,0.],[2.],[12.]]):exact(value,expected)
        gc.collect();assert all((v() is not None)==keep for v in arrays)


def test_inference_has_owned_output_without_saved_operand_lifetimes():
    x=torch.tensor([0.],requires_grad=True);lo=torch.tensor([-1.],requires_grad=True);hi=torch.tensor([1.],requires_grad=True);refs=[weakref.ref(v._data) for v in (x,lo,hi)]
    with torch.no_grad():y=x.clamp(lo,hi)
    del x,lo,hi;gc.collect();assert not y.requires_grad and all(v() is None for v in refs);exact(y,[0.])


def test_empty_broadcast_preserves_owner_shapes_and_contiguous_empty_metadata():
    x=torch.empty(0,requires_grad=True);lo=torch.tensor([[-1.],[0.]],requires_grad=True);hi=torch.tensor([[1.],[2.]],requires_grad=True);y=x.clamp(lo,hi)
    assert y.shape==(2,0) and y.stride()==(1,1)
    gx,gl,gu=torch.autograd.grad(y,(x,lo,hi),torch.ones_like(y));assert gx.shape==(0,);exact(gl,[[0.],[0.]]);exact(gu,[[0.],[0.]])


@pytest.mark.parametrize('invalid',['mixed_lower','mixed_upper','shape','complex'])
def test_argument_failures_preserve_inputs_and_original_scalar_route(invalid):
    x=torch.tensor([-.5,.5],requires_grad=True);lo=torch.tensor([-1.],requires_grad=True);hi=torch.tensor([1.],requires_grad=True);before=[v.detach().numpy().copy() for v in (x,lo,hi)]
    if invalid=='mixed_lower':args=(-1.,hi)
    elif invalid=='mixed_upper':args=(lo,1.)
    elif invalid=='shape':args=(torch.ones(3),hi)
    else:args=(torch.tensor([1j],dtype=torch.complex64),hi)
    with pytest.raises((TypeError,RuntimeError)):x.clamp(*args)
    for value,expected in zip((x,lo,hi),before):exact(value,expected);assert value.grad is None
    exact(x.clamp(-1.,1.),[-.5,.5])


@pytest.mark.parametrize('mode',['direct','functional','hvp'])
def test_quadratic_derivatives_with_broadcast_tensor_bounds(mode):
    x=torch.tensor([-2.,-.5,.5,2.],dtype=torch.float64);lo=torch.tensor([-1.],dtype=torch.float64);hi=torch.tensor([1.],dtype=torch.float64);v=(torch.full_like(x,2.),torch.full_like(lo,3.),torch.full_like(hi,4.))
    if mode=='hvp':
        value,products=torch.autograd.functional.hvp(lambda a,b,c:a.clamp(b,c).square().sum(),(x,lo,hi),v);exact(value,2.5)
        for actual,expected in zip(products,[[0.,4.,4.,0.],[6.],[8.]]):exact(actual,expected)
    else:
        fn=lambda a,b,c:a.clamp(b,c)
        y,t=torch.func.jvp(fn,(x,lo,hi),v) if mode=='direct' else torch.autograd.functional.jvp(fn,(x,lo,hi),v)
        exact(y,[-1.,-.5,.5,1.]);exact(t,[3.,2.,2.,4.])


def test_gradients_against_independent_quadratic_finite_differences():
    values=[np.array([[-2.,-.25,.5],[2.,.25,-3.]]),np.array([[-1.],[-.75]]),np.array([1.25,1.5,1.75])]
    inputs=tuple(torch.tensor(a,dtype=torch.float64,requires_grad=True) for a in values);y=inputs[0].clamp(inputs[1],inputs[2]).square().sum();actual=torch.autograd.grad(y,inputs)
    def value(args):return np.square(np.clip(args[0],args[1],args[2])).sum()
    for which,array in enumerate(values):
        expected=np.empty_like(array)
        for index in np.ndindex(array.shape):
            plus=[a.copy() for a in values];minus=[a.copy() for a in values];plus[which][index]+=1e-6;minus[which][index]-=1e-6;expected[index]=(value(plus)-value(minus))/(2e-6)
        np.testing.assert_allclose(actual[which].numpy(),expected,rtol=1e-8,atol=1e-8)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_negative_stride_import_is_rejected_before_clamp(dtype):
    with pytest.raises(ValueError,match='strides'):
        torch.from_numpy(np.arange(3,dtype=dtype)[::-1])


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('subset',[(0,),(1,),(2,),(1,2),(0,1,2)])
@pytest.mark.parametrize('mode',['direct','functional','hvp'])
@pytest.mark.parametrize('loss',['norm','nonlinear'])
def test_composed_second_derivatives_against_independent_hessian(dtype,subset,mode,loss):
    raw=[np.array([[-2.,-.5,.5],[2.,.25,-3.]],dtype=dtype),np.array([[-1.],[-.75]],dtype=dtype),np.array([1.25,1.5,1.75],dtype=dtype)]
    base=tuple(torch.tensor(a) for a in raw);values=tuple(base[i].requires_grad_() for i in subset)
    direction=[np.full_like(a,(i+1)*.125) if i in subset else np.zeros_like(a) for i,a in enumerate(raw)]
    masks=[(raw[0]>=raw[1])&(raw[0]<=raw[2]),(raw[0]<raw[1])&(raw[1]<raw[2]),(raw[0]>raw[2])|(raw[2]<raw[1])]
    z=np.clip(*raw);dz=sum(np.where(m,d,0.) for m,d in zip(masks,direction))
    if loss=='norm':
        norm=np.sqrt(np.square(z.astype(np.float64)).sum());expected_value=norm
        expected_tangent=(z*dz).sum()/norm;hz=dz/norm-z*(z*dz).sum()/norm**3
    else:
        expected_value=(np.sin(z)+np.square(z)).sum();expected_tangent=((np.cos(z)+2*z)*dz).sum();hz=(2-np.sin(z))*dz
    def function(*parts):
        args=list(base)
        for i,p in zip(subset,parts):args[i]=p
        clipped=args[0].clamp(args[1],args[2])
        return torch.linalg.vector_norm(clipped) if loss=='norm' else (clipped.sin()+clipped.square()).sum()
    v=tuple(torch.tensor(direction[i]) for i in subset)
    if mode=='hvp':value,products=torch.autograd.functional.hvp(function,values,v)
    else:
        value,tangent=torch.func.jvp(function,values,v) if mode=='direct' else torch.autograd.functional.jvp(function,values,v,create_graph=True)
        np.testing.assert_allclose(tangent.numpy(),expected_tangent,rtol=4e-6 if dtype==np.float32 else 2e-12,atol=2e-7 if dtype==np.float32 else 2e-13)
        products=torch.autograd.grad(tangent,values)
    tol=dict(rtol=4e-6 if dtype==np.float32 else 2e-12,atol=2e-7 if dtype==np.float32 else 2e-13)
    np.testing.assert_allclose(value.numpy(),expected_value,**tol)
    for i,actual in zip(subset,products):
        expected=np.where(masks[i],hz,0.)
        if i==1:expected=expected.sum(axis=1,keepdims=True)
        if i==2:expected=expected.sum(axis=0)
        np.testing.assert_allclose(actual.numpy(),expected,**tol)
