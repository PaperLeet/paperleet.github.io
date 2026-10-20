import gc
import itertools
import warnings
import weakref
import numpy as np
import pytest
from torchwasm import torch,UnsupportedOperation


def tensor(data,requires_grad=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        return torch.from_numpy(data).requires_grad_(requires_grad)


def exact(value,expected):
    actual=value.detach().numpy();expected=np.asarray(expected,dtype=actual.dtype)
    np.testing.assert_array_equal(actual,expected)
    if actual.dtype.kind=='f':np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape,axis',[((),0),((5,),0),((2,5),0),((2,5),1),((2,0,3),1),((2,0,3),2)])
@pytest.mark.parametrize('kind',['C','F','strided','readonly','broadcast'])
@pytest.mark.parametrize('interface',['method','function'])
def test_values_vjp_layout_and_readonly_seed(dtype,shape,axis,kind,interface):
    data=np.asarray((np.arange(np.prod(shape) if shape else 1,dtype=dtype).reshape(shape)+1)/8,dtype=dtype)
    if kind=='F':data=np.asfortranarray(data) if shape else data
    if kind=='strided' and shape:
        storage=np.empty(tuple(n*2 for n in shape),dtype=dtype);view=storage[tuple(slice(None,None,2) for _ in shape)];view[...]=data;data=view
    if kind=='readonly':data.flags.writeable=False
    if kind=='broadcast' and shape and data.size:
        data=np.broadcast_to(np.take(data,[0],axis=axis),shape)
    before=data.copy();x=tensor(data,True)
    y=x.cumsum(axis) if interface=='method' else torch.cumsum(x,axis)
    seed=np.arange(y.numel(),dtype=dtype).reshape(shape)/4 if shape else np.asarray(.75,dtype=dtype);seed.flags.writeable=False
    gradient,=torch.autograd.grad(y,x,tensor(seed))
    if shape:
        triangle=np.tril(np.ones((shape[axis],shape[axis]),dtype=dtype))
        expected=np.moveaxis(np.moveaxis(data,axis,-1)@triangle.T,-1,axis)
        expected_g=np.moveaxis(np.moveaxis(seed,axis,-1)@triangle,-1,axis)
    else:expected=data;expected_g=seed
    exact(y,expected);exact(gradient,expected_g)
    assert y.shape==shape and y.is_contiguous() and gradient.shape==shape
    np.testing.assert_array_equal(data,before);assert x.grad is None


@pytest.mark.parametrize('dtype',[np.bool_,np.uint8,np.int8,np.int16,np.int32,np.int64,np.uint64])
@pytest.mark.parametrize('shape',[(),(3,),(2,3),(2,0)])
def test_integral_default_promotion(dtype,shape):
    data=np.ones(shape,dtype=dtype);result=tensor(data).cumsum(-1)
    assert result.dtype==torch.int64 and result.shape==shape
    expected=np.broadcast_to(np.arange(1,shape[-1]+1,dtype=np.int64),shape) if shape else np.array(1,dtype=np.int64)
    exact(result,expected)


def test_unsigned64_cast_and_signed_overflow():
    data=np.array([2**63,2**64-1,1],dtype=np.uint64)
    exact(tensor(data).cumsum(0),[-2**63,2**63-1,-2**63])


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64])
def test_native_accumulation_precision_and_precast(dtype):
    scale=1e4 if dtype==torch.float16 else 1e8
    y=torch.tensor([scale,1.,-scale],dtype=dtype).cumsum(0)
    exact(y,[scale,np.asarray(scale+1,dtype=dtype),1.])
    exact(torch.tensor([1e8+1.,-1e8],dtype=torch.float64).cumsum(0,dtype=torch.float32),[1e8,0.])


@pytest.mark.parametrize('source',[torch.float32,torch.float64])
@pytest.mark.parametrize('target',[torch.float16,torch.float32,torch.float64])
def test_dtype_override_casts_reverse_into_input_dtype(source,target):
    x=torch.tensor([.25,-.5,.75],dtype=source,requires_grad=True);y=torch.cumsum(input=x,dim=0,dtype=target)
    g,=torch.autograd.grad(y,x,torch.tensor([1.,2.,3.],dtype=target))
    assert y.dtype==target and g.dtype==source;exact(g,[6.,5.,3.])


def test_scalar_signed_zero_copy_and_vector_positive_zero():
    x=torch.tensor(-0.,requires_grad=True);y=x.cumsum(-1);assert y.shape==() and y is not x and not np.shares_memory(x._data,y._data);exact(y,-0.)
    exact(torch.tensor([-0.,-0.,-0.]).cumsum(0),[0.,0.,0.])
    y.backward(torch.tensor(-0.));exact(x.grad,-0.)


def test_nonfinite_values_do_not_change_linear_pullback():
    x=torch.tensor([float('nan'),float('inf'),-float('inf')],requires_grad=True);y=x.cumsum(0)
    g,=torch.autograd.grad(y,x,torch.tensor([1.,2.,3.]));exact(g,[6.,5.,3.]);assert np.isnan(y.numpy()).all()


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('loss',['quadratic','norm','sin'])
@pytest.mark.parametrize('mode',['direct','functional','hvp'])
@pytest.mark.parametrize('interface',['method','function'])
def test_composed_hessians_against_independent_linear_operator(axis,dtype,loss,mode,interface):
    raw=np.array([[.25,-.5,.75],[1.,-.25,.5]],dtype=dtype);direction=np.array([[.5,-.25,1.],[.25,.75,-.5]],dtype=dtype)
    coords=list(np.ndindex(raw.shape));matrix=np.array([[int(o[1-axis]==i[1-axis] and o[axis]>=i[axis]) for i in coords]for o in coords],dtype=np.float64)
    point=matrix@raw.reshape(-1);dv=matrix@direction.reshape(-1)
    if loss=='quadratic':expected_value=point@point;grad=2*point;h=2*np.eye(point.size)
    elif loss=='norm':
        radius=np.sqrt(point@point);expected_value=radius;grad=point/radius;h=np.eye(point.size)/radius-np.outer(point,point)/radius**3
    else:expected_value=np.sin(point).sum();grad=np.cos(point);h=np.diag(-np.sin(point))
    x=tensor(raw.copy(),True);v=tensor(direction.copy(),True)
    def fn(a):
        z=a.cumsum(axis) if interface=='method' else torch.cumsum(a,axis)
        return z.square().sum() if loss=='quadratic' else torch.linalg.vector_norm(z) if loss=='norm' else z.sin().sum()
    if mode=='hvp':value,hv=torch.autograd.functional.hvp(fn,x,v)
    else:
        value,tangent=torch.func.jvp(fn,(x,),(v,)) if mode=='direct' else torch.autograd.functional.jvp(fn,x,v,create_graph=True)
        np.testing.assert_allclose(tangent.numpy(),grad@dv,rtol=3e-5,atol=3e-6)
        hv,=torch.autograd.grad(tangent,x)
    np.testing.assert_allclose(value.numpy(),expected_value,rtol=3e-5,atol=3e-6)
    np.testing.assert_allclose(hv.numpy(),(matrix.T@h@dv).reshape(raw.shape),rtol=3e-5,atol=3e-6)


def test_linear_hvp_strictness_and_no_grad_forward_ad():
    x=torch.tensor([1.,2.,3.]);v=torch.ones_like(x)
    _,h=torch.autograd.functional.hvp(lambda a:torch.cumsum(a,0).sum(),x,v);exact(h,[0.,0.,0.])
    with pytest.raises(RuntimeError):torch.autograd.functional.hvp(lambda a:torch.cumsum(a,0).sum(),x,v,strict=True)
    with torch.no_grad():p,t=torch.func.jvp(lambda a:torch.cumsum(a,0),(x,),(v,))
    exact(p,[1.,3.,6.]);exact(t,[1.,2.,3.])


def test_unsaved_input_lifetimes_and_mutation_guards():
    x=torch.tensor([.25,-.5,.75],requires_grad=True);middle=x+.125;wrapper=weakref.ref(middle);array=weakref.ref(middle._data);y=middle.cumsum(0)
    del middle;gc.collect();assert wrapper() is None and array() is None
    with torch.no_grad():x.add_(1.)
    y.sum().backward();exact(x.grad,[3.,2.,1.])
    x=torch.tensor([1.,2.],requires_grad=True);y=x.cumsum(0);loss=y.square().sum()
    with torch.no_grad():y.detach().add_(1.)
    with pytest.raises(RuntimeError,match='modified'):loss.backward()


@pytest.mark.parametrize('name',['clamp','clamp_min','clamp_max'])
def test_top_level_clamp_forward_rule_and_capability(name):
    x=torch.tensor([-2.,-.5,.5,2.]);v=torch.ones_like(x);args=(torch.tensor([-1.]),torch.tensor([1.])) if name=='clamp' else (torch.tensor([-1. if name=='clamp_min' else 1.]),)
    _,t=torch.func.jvp(lambda a:getattr(torch,name)(a,*args),(x,),(v,))
    exact(t,[0.,1.,1.,0.] if name=='clamp' else [0.,1.,1.,1.] if name=='clamp_min' else [1.,1.,1.,0.]);assert torch._browser_capabilities['torch.'+name].forward
    assert torch._browser_capabilities['torch.cumsum'].forward


def test_existing_transform_dtype_boundaries_stay_explicit():
    x=torch.tensor([1.,2.],dtype=torch.float32)
    with pytest.raises(UnsupportedOperation):torch.func.jvp(lambda a:a.cumsum(0),(x,),(torch.ones(2,dtype=torch.float64),))
    with pytest.raises(UnsupportedOperation):torch.func.jvp(lambda a:a.cumsum(0,dtype=torch.float16),(x,),(torch.ones_like(x),))


@pytest.mark.parametrize('dtype,length',[(np.float32,32767),(np.float32,32768),(np.float32,32769),(np.float32,65537),(np.float16,65535),(np.float16,65536),(np.float16,65537)])
def test_block_edges_and_half_reverse_accumulation(dtype,length):
    raw=np.resize(np.array([.125,-.25,.5,-.375],dtype=dtype),length);x=tensor(raw,True);y=x.cumsum(0)
    exact(y,np.resize(np.array([.125,-.125,.375,0.],dtype=dtype),length))
    seed=np.full(length,.25,dtype=dtype);seed.flags.writeable=False
    gradient,=torch.autograd.grad(y,x,tensor(seed));exact(gradient,np.arange(length,0,-1)*.25)


@pytest.mark.parametrize('dtype,length,scale',[(np.float32,32770,1e8),(np.float16,65538,1e4)])
def test_unrounded_opmath_carry_in_forward_and_reverse(dtype,length,scale):
    values=np.zeros(length,dtype=dtype);values[0]=scale;values[-3]=1.;values[-2]=-scale;values[-1]=1.
    expected=np.full(length,scale,dtype=dtype);expected[-2:]=[1.,2.]
    exact(tensor(values).cumsum(0),expected)
    x=tensor(np.zeros(length,dtype=dtype),True);g,=torch.autograd.grad(x.cumsum(0),x,tensor(values[::-1].copy()))
    exact(g,expected[::-1])


@pytest.mark.parametrize('dtype,scale,last',[(np.float16,1e4,[1.,2.]),(np.float32,1e8,[1.,2.]),(np.float64,1e16,[0.,1.])])
@pytest.mark.parametrize('kind',['C','F','strided'])
def test_short_axis_batch_keeps_the_native_accumulator(dtype,scale,last,kind):
    values=np.broadcast_to(np.array([scale,1.,-scale,1.],dtype=dtype)[:,None],(4,1024)).copy()
    if kind=='F':values=np.asfortranarray(values)
    if kind=='strided':
        storage=np.empty((4,2048),dtype=dtype);storage[:,::2]=values;values=storage[:,::2]
    x=tensor(values);exact(x.cumsum(0),np.broadcast_to(np.array([scale,scale,*last],dtype=dtype)[:,None],values.shape))
    assert x.cumsum(0).is_contiguous()


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64,np.complex64,np.complex128])
def test_native_positive_zero_over_long_prefixes(dtype):
    value=complex(-0.,-0.) if np.dtype(dtype).kind=='c' else -0.
    x=tensor(np.full(65537,value,dtype=dtype));y=x.cumsum(0).numpy()
    assert not np.signbit(y.real).any()
    if y.dtype.kind=='c':assert not np.signbit(y.imag).any()


@pytest.mark.parametrize('shape,axis',[((3,300,50),2),((3,300,50),0),((2,5,4000),2)])
def test_multidimensional_partial_work_buffers(shape,axis):
    x=tensor(np.ones(shape,dtype=np.float32),True);y=x.cumsum(axis)
    pattern=np.arange(1,shape[axis]+1,dtype=np.float32);view=[1]*len(shape);view[axis]=shape[axis]
    exact(y,np.broadcast_to(pattern.reshape(view),shape))
    gradient,=torch.autograd.grad(y.square().sum(),x)
    i=np.arange(shape[axis],dtype=np.float64);expected=shape[axis]*(shape[axis]+1)-i*(i+1)
    exact(gradient,np.broadcast_to(expected.reshape(view),shape))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('length',[32769,65536])
def test_norm_hvp_through_long_scans(dtype,length):
    raw=np.zeros(length,dtype=dtype);raw[0]=1.;direction=np.zeros(length,dtype=dtype);direction[-1]=1.
    x=tensor(raw);v=tensor(direction)
    value,product=torch.autograd.functional.hvp(lambda a:torch.linalg.vector_norm(torch.cumsum(a,0)),x,v)
    np.testing.assert_allclose(value.numpy(),np.sqrt(length),rtol=3e-6)
    expected=np.arange(length,dtype=np.float64)/(length*np.sqrt(length))
    np.testing.assert_allclose(product.numpy(),expected,rtol=3e-5,atol=3e-7)


def test_forced_small_work_blocks_keep_exact_prefix_and_reverse(monkeypatch):
    from torchwasm import _reductions
    monkeypatch.setattr(_reductions,'_SCAN_WORK_BYTES',16)
    x=torch.tensor([[1e8,1.,-1e8,1.,1.],[1.,2.,3.,4.,5.]],dtype=torch.float32,requires_grad=True)
    y=x.cumsum(1);exact(y,[[1e8,1e8,1.,2.,3.],[1.,3.,6.,10.,15.]])
    g,=torch.autograd.grad(y,x,torch.tensor([[1.,2.,3.,4.,5.],[1e8,-1e8,1.,1.,1.]]))
    exact(g,[[15.,14.,12.,9.,5.],[3.,-1e8,3.,2.,1.]])


def test_reverse_cast_precedes_scan_in_original_input_dtype():
    x=torch.ones(32770,dtype=torch.float32,requires_grad=True);y=x.cumsum(0,dtype=torch.float16)
    g,=torch.autograd.grad(y,x,torch.full_like(y,.25));exact(g,np.arange(32770,0,-1)*.25)
    x=torch.ones(4,dtype=torch.float16,requires_grad=True);y=x.cumsum(0,dtype=torch.float64)
    g,=torch.autograd.grad(y,x,torch.full_like(y,1e-8));exact(g,[0.,0.,0.,0.])
    x=torch.ones(3,dtype=torch.float32,requires_grad=True);y=x.cumsum(0,dtype=torch.float64)
    g,=torch.autograd.grad(y,x,torch.tensor([1e8+1.,-1e8,0.],dtype=torch.float64));exact(g,[0.,-1e8,0.])


@pytest.mark.parametrize('mode',['direct','functional'])
def test_dtype_override_preserves_transform_specific_rounding(mode):
    x=torch.zeros(3,dtype=torch.float32);v=torch.tensor([1e8,1.,-1e8],dtype=torch.float32);fn=lambda a:torch.cumsum(a,0,dtype=torch.float64)
    _,t=torch.func.jvp(fn,(x,),(v,)) if mode=='direct' else torch.autograd.functional.jvp(fn,x,v)
    exact(t,[1e8,1e8+1. if mode=='direct' else 1e8,1.])
    x=torch.zeros(2,dtype=torch.float64);v=torch.tensor([1e8+1.,-1e8],dtype=torch.float64);fn=lambda a:torch.cumsum(a,0,dtype=torch.float32)
    _,t=torch.func.jvp(fn,(x,),(v,)) if mode=='direct' else torch.autograd.functional.jvp(fn,x,v)
    exact(t,[1e8,0. if mode=='direct' else 1.])


@pytest.mark.parametrize('source,target',[(torch.float32,torch.float64),(torch.float64,torch.float32)])
@pytest.mark.parametrize('loss',['quadratic','norm'])
def test_dtype_changing_composed_hvp(source,target,loss):
    x=torch.tensor([.25,-.5,.75],dtype=source);v=torch.tensor([.125,.25,-.5],dtype=source)
    def fn(a):
        y=torch.cumsum(a,0,dtype=target)
        return y.square().sum() if loss=='quadratic' else torch.linalg.vector_norm(y)
    value,product=torch.autograd.functional.hvp(fn,x,v)
    matrix=np.tril(np.ones((3,3)));point=matrix@np.array([.25,-.5,.75]);direction=matrix@np.array([.125,.25,-.5])
    radius=np.sqrt(point@point);h=2*np.eye(3) if loss=='quadratic' else np.eye(3)/radius-np.outer(point,point)/radius**3
    np.testing.assert_allclose(product.numpy(),matrix.T@h@direction,rtol=3e-5,atol=3e-6)
