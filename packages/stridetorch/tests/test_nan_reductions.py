"""NumPy-only mathematical and graph contracts, also executed on WASM."""
import numpy as np
import pytest
from stridetorch import UnsupportedOperation, torch


@pytest.mark.parametrize('name',['nansum','nanmean'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_nan_reduction_jvp_duality_and_directional_derivative(name,dtype):
    data=np.array([[1.,np.nan,3.],[4.,5.,np.nan]],dtype=dtype)
    direction=np.array([[.2,.4,.6],[.8,1.,1.2]],dtype=dtype)
    p=torch.tensor(data,requires_grad=True);t=torch.tensor(direction,requires_grad=True)
    f=lambda x:getattr(torch,name)(x,dim=1)
    value,jvp=torch.func.jvp(f,(p,),(t,))
    count=2 if name=='nanmean' else 1
    np.testing.assert_allclose(value.numpy(),[4/count,9/count])
    np.testing.assert_allclose(jvp.numpy(),[.8/count,1.8/count],rtol=1e-6)
    seed=torch.tensor([.7,-.3],dtype=dtype)
    grad=torch.autograd.grad(value,p,grad_outputs=seed)[0]
    np.testing.assert_allclose((grad.numpy()*direction).sum(),(seed.numpy()*jvp.numpy()).sum(),rtol=1e-6)
    eps=1e-3 if dtype==torch.float32 else 1e-5
    numeric=(f(torch.tensor(data+eps*direction)).numpy()-f(torch.tensor(data-eps*direction)).numpy())/(2*eps)
    np.testing.assert_allclose(jvp.numpy(),numeric,rtol=5e-4,atol=3e-5)
    jvp.backward(seed)
    expected=np.where(np.isnan(data),0,np.array([[.7],[-.3]],dtype=dtype)/count)
    np.testing.assert_allclose(t.grad.numpy(),expected,rtol=1e-6)
    if name=='nanmean':np.testing.assert_array_equal(p.grad.numpy(),np.zeros_like(data))
    else:assert p.grad is None


@pytest.mark.parametrize('name',['nansum','nanmean'])
def test_nan_all_missing_empty_and_numpy_policy(name):
    f=getattr(torch,name)
    with np.errstate(all='raise'):
        x=torch.tensor([np.nan,np.nan],requires_grad=True)
        y=f(x);y.backward()
        if name=='nansum':
            assert y.item()==0;np.testing.assert_array_equal(x.grad.numpy(),[0.,0.])
        else:assert np.isnan(y.item()) and np.isnan(x.grad.numpy()).all()
        empty=f(torch.empty(0,3),dim=0).numpy()
        assert np.all(empty==0) if name=='nansum' else np.isnan(empty).all()
        assert np.geterr()==dict(divide='raise',over='raise',under='raise',invalid='raise')


def test_nan_jvp_zero_primal_edge_and_ignored_tangent_ieee():
    p=torch.tensor([np.inf,np.nan],requires_grad=True)
    t=torch.tensor([2.,np.nan],requires_grad=True)
    _,jvp=torch.func.jvp(lambda x:torch.nanmean(x),(p,),(t,))
    assert jvp.item()==2.
    jvp.backward(torch.tensor(np.inf))
    np.testing.assert_array_equal(p.grad.numpy(),[0.,0.])
    assert np.isposinf(t.grad.numpy()[0]) and t.grad.numpy()[1]==0
    with pytest.raises(RuntimeError):jvp.backward()


def test_nan_output_and_ad_bounds_are_explicit():
    for name in ('nansum','nanmean'):
        f=getattr(torch,name)
        with pytest.raises(UnsupportedOperation):f(torch.ones(2),out=torch.zeros(1))
        with pytest.raises(RuntimeError):f(torch.tensor([1+2j]))
        with pytest.raises(UnsupportedOperation):
            torch.func.jvp(lambda x:f(x,dtype=torch.float16),(torch.ones(2),),(torch.ones(2),))


@pytest.mark.parametrize('name',['nansum','nanmean'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_nan_saved_mask_uses_current_values_and_cached_count(name,dtype):
    data=np.array([1.,np.nan,3.],dtype=dtype)
    x=torch.from_numpy(data).requires_grad_()
    y=getattr(torch,name)(x,dtype=torch.float64)
    data[1]=2.;y.detach().fill_(99)
    y.backward()
    expected=1. if name=='nansum' else .5
    np.testing.assert_array_equal(x.grad.numpy(),np.full(3,expected,dtype=dtype))
    with pytest.raises(RuntimeError):y.backward()


@pytest.mark.parametrize('name',['nansum','nanmean'])
def test_nan_jvp_input_guard_and_selective_pruning(name):
    p=torch.tensor([1.,np.nan,3.],requires_grad=True);t=torch.ones(3,requires_grad=True)
    _,jvp=torch.func.jvp(lambda x:getattr(torch,name)(x),(p,),(t,))
    with torch.no_grad():p.fill_(2.)
    tangent_grad=torch.autograd.grad(jvp,t,retain_graph=True)[0]
    np.testing.assert_array_equal(tangent_grad.numpy(),[1.,0.,1.] if name=='nansum' else [.5,0.,.5])
    if name=='nanmean':
        with pytest.raises(UnsupportedOperation):jvp.backward()
    else:
        jvp.backward();np.testing.assert_array_equal(t.grad.numpy(),[1.,0.,1.])
