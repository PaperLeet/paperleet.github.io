"""Native CPU reference for NaN reductions, casts, graph lifetime and JVP."""
import numpy as np
import pytest
import torch as native
from stridetorch import UnsupportedOperation, torch


def reduce(api, x, name, functional=True, **options):
    return getattr(api,name)(x,**options) if functional else getattr(x,name)(**options)


def equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    tolerance = (2e-3,0.) if expected.dtype == np.float16 else (3e-6,2e-7) if expected.dtype == np.float32 else (2e-11,1e-12)
    np.testing.assert_allclose(actual,expected,rtol=tolerance[0],atol=tolerance[1],equal_nan=True)


@pytest.mark.parametrize('name',['nansum','nanmean'])
@pytest.mark.parametrize('source',['float16','float32','float64'])
@pytest.mark.parametrize('target',['float16','float32','float64'])
@pytest.mark.parametrize('shape,dim,keepdim',[((),None,False),((3,4),0,False),((3,4),-1,True),((3,4),[],True),((0,3),0,False),((3,0),(0,1),True)])
@pytest.mark.parametrize('functional',[True,False])
def test_native_nan_values_and_gradients(name,source,target,shape,dim,keepdim,functional):
    data=np.linspace(-.9,1.1,int(np.prod(shape))).reshape(shape).astype(source)
    if data.size:data.flat[::3]=np.nan
    outputs=[]
    for api in (native,torch):
        x=api.tensor(data,requires_grad=True)
        y=reduce(api,x,name,functional,dim=dim,keepdim=keepdim,dtype=getattr(api,target))
        y.backward(api.full(y.shape,.37,dtype=getattr(api,target)))
        outputs.append((y.detach().numpy(),x.grad.numpy()))
    for expected,actual in zip(*outputs):equal(actual,expected)


@pytest.mark.parametrize('name',['nansum','nanmean'])
@pytest.mark.parametrize('source',['float32','float64'])
@pytest.mark.parametrize('target',['float32','float64'])
@pytest.mark.parametrize('data,tangent',[
    ([1.,np.nan,3.],[1.,np.nan,3.]),([1.,np.nan,3.],[np.nan,2.,3.]),
    ([np.nan,np.nan],[np.inf,np.nan]),([np.inf,1.],[1.,2.]),
    ([np.inf,-np.inf],[1.,2.]),([],[]),
])
@pytest.mark.parametrize('seed',[.37,np.inf,np.nan])
def test_nan_jvp_and_reverse_ieee(name,source,target,data,tangent,seed):
    outputs=[]
    for api in (native,torch):
        p=api.tensor(data,dtype=getattr(api,source),requires_grad=True)
        t=api.tensor(tangent,dtype=getattr(api,source),requires_grad=True)
        value,jvp=api.func.jvp(lambda x:reduce(api,x,name,dtype=getattr(api,target)),(p,),(t,))
        jvp.backward(api.tensor(seed,dtype=getattr(api,target)))
        primal_grad = None if p.grad is None else p.grad.numpy(force=True) if api is native else p.grad.numpy()
        outputs.append((value.detach().numpy(),jvp.detach().numpy(),primal_grad,t.grad.numpy()))
    for expected,actual in zip(*outputs):
        if expected is None:assert actual is None
        else:equal(actual,expected)


@pytest.mark.parametrize('name',['nansum','nanmean'])
@pytest.mark.parametrize('target',['float16','float32','float64'])
@pytest.mark.parametrize('seed',[0.,np.inf,np.nan])
def test_nan_primitive_backward_uses_multiply_mask(name,target,seed):
    results=[]
    for api in (native,torch):
        x=api.tensor([[1.,np.nan,3.],[np.nan,np.nan,np.nan]],requires_grad=True)
        y=reduce(api,x,name,dim=1,dtype=getattr(api,target))
        y.backward(api.full(y.shape,seed,dtype=getattr(api,target)))
        results.append(x.grad.numpy())
    equal(results[1],results[0])


@pytest.mark.parametrize('name',['nansum','nanmean'])
@pytest.mark.parametrize('source,destination',[('float32','float64'),('float64','float32')])
def test_nan_original_input_cast_survives_model_conversion(name,source,destination):
    results=[]
    for api in (native,torch):
        layer=api.nn.Linear(3,1,bias=False).to(dtype=getattr(api,source))
        with api.no_grad():layer.weight.copy_(api.tensor([[1.,np.nan,3.]],dtype=getattr(api,source)))
        y=reduce(api,layer.weight,name,dtype=api.float64)
        layer.to(dtype=getattr(api,destination))
        y.backward(api.tensor(1e40,dtype=api.float64));results.append(layer.weight.grad.numpy())
    equal(results[1],results[0])


@pytest.mark.parametrize('name',['nansum','nanmean'])
@pytest.mark.parametrize('mutate',['input','output','external','rebind'])
def test_nan_mask_storage_and_output_lifetime(name,mutate):
    results=[]
    for api in (native,torch):
        data=np.array([1.,np.nan,3.],dtype=np.float64)
        x=api.from_numpy(data).requires_grad_();y=reduce(api,x,name)
        if mutate=='input':
            with api.no_grad():x.detach().fill_(2)
            with pytest.raises((RuntimeError,UnsupportedOperation)):y.backward()
            continue
        if mutate=='output':
            with api.no_grad():y.detach().fill_(99)
        elif mutate=='external':data[1]=2
        else:
            replacement=api.tensor([1.,2.,3.],dtype=api.float64)
            if api is native:x.data=replacement
            else:x._data=replacement._data  # Exercise internal rebinding; no public .data setter.
        y.backward();results.append(x.grad.numpy())
        with pytest.raises(RuntimeError):y.backward()
    if results:equal(results[1],results[0])


@pytest.mark.parametrize('target',['primal','tangent'])
def test_nanmean_jvp_target_prunes_primal_version_guard(target):
    for api in (native,torch):
        p=api.tensor([1.,np.nan,3.],requires_grad=True);t=api.ones(3,requires_grad=True)
        _,jvp=api.func.jvp(lambda x:api.nanmean(x),(p,),(t,))
        with api.no_grad():p.fill_(2)
        if target=='primal':
            with pytest.raises((RuntimeError,UnsupportedOperation)):api.autograd.grad(jvp,p)
        else:np.testing.assert_array_equal(api.autograd.grad(jvp,t)[0].numpy(),[.5,0.,.5])


@pytest.mark.parametrize('name',['sum','mean','nansum','nanmean'])
@pytest.mark.parametrize('options',[{'dim':True},{'dim':[False]},{'dim':1.},{'dim':(0,0)},{'dim':3},{'dim':0,'keepdim':1},{'dim':None,'keepdim':None}])
def test_reduction_argument_exception_types(name,options):
    errors=[]
    for api in (native,torch):
        try:reduce(api,api.ones(2,3),name,**options)
        except Exception as error:errors.append(type(error))
        else:raise AssertionError('Invalid arguments accepted')
    assert errors[0] is errors[1]


@pytest.mark.parametrize('name',['nansum','nanmean'])
def test_nan_half_composite_overflow_and_dtype_guards(name):
    for api in (native,torch):
        x=api.ones(65536,dtype=api.float16,requires_grad=True)
        y=reduce(api,x,name);assert np.isposinf(y.item())
        y.backward();expected=1. if name=='nansum' else np.float16(1/65536)
        np.testing.assert_array_equal(x.grad.numpy(),np.full(65536,expected,dtype=np.float16))
        assert reduce(api,api.ones(2,3),name,keepdim=True).shape==(1,1)
        with pytest.raises(RuntimeError):reduce(api,api.ones(3,requires_grad=True),name,dtype=api.int64)
        if name=='nanmean':
            with pytest.raises(RuntimeError):api.nanmean(api.tensor([1,2]),dtype=api.float64)
        else:assert api.nansum(api.tensor([True,False])).dtype==api.int64


@pytest.mark.parametrize('name',['nansum','nanmean'])
@pytest.mark.parametrize('source',['float16','float32','float64','int64','bool','complex64'])
@pytest.mark.parametrize('target',[None,'float16','float64','int64','bool','complex64'])
def test_nan_dtype_matrix_and_exception_types(name,source,target):
    outcomes=[]
    for api in (native,torch):
        x=api.tensor([1,0,3],dtype=getattr(api,source))
        options={} if target is None else dict(dtype=getattr(api,target))
        try:outcomes.append(reduce(api,x,name,**options).numpy())
        except Exception as error:outcomes.append(type(error))
    if isinstance(outcomes[0],type):assert outcomes[0] is outcomes[1]
    else:equal(outcomes[1],outcomes[0])


@pytest.mark.parametrize('name',['nansum','nanmean'])
@pytest.mark.parametrize('primal_grad,tangent_grad',[(False,True),(True,False),(True,True)])
@pytest.mark.parametrize('mutate',[True,False])
def test_nan_jvp_graph_flags_lifetime_and_input_mutation(name,primal_grad,tangent_grad,mutate):
    outcomes=[]
    for api in (native,torch):
        p=api.tensor([1.,np.nan,3.],requires_grad=primal_grad)
        t=api.ones(3,requires_grad=tangent_grad)
        _,jvp=api.func.jvp(lambda x:reduce(api,x,name),(p,),(t,))
        tracked=tangent_grad or name=='nanmean' and primal_grad
        assert jvp.requires_grad==tracked
        if not tracked:continue
        if mutate:
            with api.no_grad():p.fill_(2)
        if mutate and name=='nanmean' and primal_grad:
            with pytest.raises((RuntimeError,UnsupportedOperation)):jvp.backward()
            continue
        jvp.backward(retain_graph=True)
        # Native's lazy ZeroTensor .grad is immutable; avoid its unrelated
        # accumulation failure while checking replay of the retained graph.
        p.grad=None;t.grad=None;jvp.backward()
        if tangent_grad:outcomes.append(t.grad.numpy())
        with pytest.raises(RuntimeError):jvp.backward()
    if outcomes:equal(outcomes[1],outcomes[0])


@pytest.mark.parametrize('name',['nansum','nanmean'])
@pytest.mark.parametrize('target',['int64','bool','complex64'])
@pytest.mark.parametrize('dim',[5,(0,0),(0,1)])
def test_nan_axis_errors_precede_kernel_and_autograd_guards(name,target,dim):
    outcomes=[]
    for api in (native,torch):
        try:reduce(api,api.tensor(1.,requires_grad=True),name,dim=dim,dtype=getattr(api,target))
        except Exception as error:outcomes.append(type(error))
        else:raise AssertionError('Invalid reduction accepted')
    assert outcomes[0] is outcomes[1]
