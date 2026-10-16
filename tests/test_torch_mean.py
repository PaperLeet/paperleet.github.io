"""Native-reference mean casts, accumulation, overloads and AD contracts."""
import numpy as np
import pytest
import torch as native
from stridetorch import UnsupportedOperation, torch


def call(api, x, functional, **kwargs):
    return api.mean(x, **kwargs) if functional else x.mean(**kwargs)


@pytest.mark.parametrize('source', ['float16','float32','float64','int64','bool'])
@pytest.mark.parametrize('target', [None,'float16','float32','float64'])
@pytest.mark.parametrize('shape,dim,keepdim', [((),None,False),((3,5),0,True),((3,5),(0,1),False),((3,5),[],True),((0,3),0,False),((3,0),-1,True)])
@pytest.mark.parametrize('functional', [False,True])
def test_mean_cast_values_and_seeded_gradients(source,target,shape,dim,keepdim,functional):
    count=int(np.prod(shape));data=np.linspace(-.9,1.1,max(1,count))[:count].reshape(shape).astype(source)
    results=[]
    for api in (native,torch):
        x=api.tensor(data,dtype=getattr(api,source),requires_grad=data.dtype.kind=='f')
        options=dict(dim=dim,keepdim=keepdim)
        if target is not None:options['dtype']=getattr(api,target)
        if source in ('int64','bool') and target is None:
            with pytest.raises(RuntimeError):call(api,x,functional,**options)
            continue
        y=call(api,x,functional,**options);gradient=None
        if x.requires_grad:
            y.backward(api.full(y.shape,.37,dtype=getattr(api,target or source)))
            gradient=x.grad.numpy()
        results.append((y.detach().numpy(),gradient))
    if not results:return
    expected,actual=results
    for a,b in zip(expected,actual):
        if a is None:assert b is None;continue
        assert a.shape==b.shape and a.dtype==b.dtype
        tolerance=(2e-3,0.) if a.dtype==np.float16 else (2e-6,2e-7) if a.dtype==np.float32 else (2e-11,1e-12)
        np.testing.assert_allclose(b,a,rtol=tolerance[0],atol=tolerance[1],equal_nan=True)


@pytest.mark.parametrize('source', ['float16','float32','float64'])
@pytest.mark.parametrize('functional', [False,True])
def test_half_mean_large_count_preserves_finite_value_and_subnormal_gradients(source,functional):
    results=[]
    for api in (native,torch):
        x=api.ones(65536,dtype=getattr(api,source),requires_grad=True)
        y=call(api,x,functional,dtype=api.float16)
        assert y.item()==1
        y.backward(api.tensor(.37,dtype=api.float16));results.append(x.grad.numpy())
    assert np.all(results[0]!=0)
    np.testing.assert_array_equal(results[1],results[0])


@pytest.mark.parametrize('source', ['float32','float64'])
@pytest.mark.parametrize('target', ['float32','float64'])
@pytest.mark.parametrize('dim', [None,0,-1,[]])
def test_mean_jvp_and_reverse_over_tangent_across_casts(source,target,dim):
    outputs=[]
    for api in (native,torch):
        p=api.tensor(np.linspace(-1,1,12).reshape(3,4),dtype=getattr(api,source),requires_grad=True)
        t=api.tensor(np.linspace(.1,.9,12).reshape(3,4),dtype=getattr(api,source),requires_grad=True)
        primal,tangent=api.func.jvp(lambda x:api.mean(x,dim=dim,dtype=getattr(api,target)),(p,),(t,))
        tangent.square().sum().backward()
        assert p.grad is None
        outputs.append((primal.detach().numpy(),tangent.detach().numpy(),t.grad.numpy()))
    for expected,actual in zip(*outputs):
        assert actual.dtype==expected.dtype
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-7)


@pytest.mark.parametrize('options', [{'keepdim':True},{'keepdim':False},{'dim':True},{'dim':[False]},{'dim':1.},{'dim':(0,0)},{'dim':3},{'dim':0,'keepdim':1},{'dim':None,'keepdim':None}])
@pytest.mark.parametrize('functional', [False,True])
def test_mean_reduction_argument_errors_match_native(options,functional):
    errors=[]
    for api in (native,torch):
        try:call(api,api.ones(2,3),functional,**options)
        except Exception as error:errors.append(type(error))
        else:raise AssertionError('Invalid mean arguments were accepted')
    assert errors[0] is errors[1]


def test_mean_dtype_and_output_guards_remain_explicit():
    x=torch.ones(2)
    assert 'dtype' not in torch._browser_capabilities['torch.mean'].unsupported_keywords
    with pytest.raises(UnsupportedOperation,match='out'):torch.mean(x,dtype=torch.float64,out=torch.zeros(1))
    with pytest.raises(UnsupportedOperation,match='Complex autograd'):torch.mean(x.requires_grad_(),dtype=torch.complex64)
    with pytest.raises(UnsupportedOperation,match='Forward mode'):
        torch.func.jvp(lambda value:torch.mean(value,dtype=torch.float16),(torch.ones(2),),(torch.ones(2),))


@pytest.mark.parametrize('source,destination', [('float32','float64'),('float64','float32')])
@pytest.mark.parametrize('target', ['float16','float32','float64'])
@pytest.mark.parametrize('count', [6,65536])
def test_mean_pullback_retains_native_cast_rounding_after_model_conversion(source,destination,target,count):
    outputs=[]
    for api in (native,torch):
        layer=api.nn.Linear(count,1,bias=False).to(dtype=getattr(api,source))
        with api.no_grad():layer.weight.fill_(.5)
        identity=id(layer.weight);out=api.mean(layer.weight,dtype=getattr(api,target))
        layer.to(dtype=getattr(api,destination));assert id(layer.weight)==identity
        out.backward(api.tensor(.37,dtype=getattr(api,target)))
        outputs.append(layer.weight.grad.numpy())
    assert outputs[0].dtype==outputs[1].dtype
    np.testing.assert_array_equal(outputs[1],outputs[0])


@pytest.mark.parametrize('source', ['complex64','complex128'])
@pytest.mark.parametrize('target', [None,'float16','float32','float64','complex64','complex128'])
def test_mean_preserves_existing_complex_forward_casts_without_enabling_complex_ad(source,target):
    values=[]
    for api in (native,torch):
        x=api.tensor([[1+2j,3-4j],[-2+1j,4+3j]],dtype=getattr(api,source))
        options=dict(dim=0)
        if target is not None:options['dtype']=getattr(api,target)
        values.append(api.mean(x,**options).numpy())
    assert values[0].dtype==values[1].dtype
    np.testing.assert_allclose(values[1],values[0],rtol=2e-6,atol=2e-7)
