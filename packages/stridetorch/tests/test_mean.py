"""Independent mean oracles, real AD, numerical policy and storage lifetime."""
import gc
import weakref
import numpy as np
import pytest
from stridetorch import UnsupportedOperation, torch


@pytest.mark.parametrize('dtype', [torch.float16,torch.float32,torch.float64])
def test_half_opmath_accumulation_and_rounded_pullback(dtype):
    x=torch.ones(65536,dtype=dtype,requires_grad=True)
    y=torch.mean(x,dtype=torch.float16)
    assert y.dtype==torch.float16 and y.item()==1
    y.backward(torch.tensor(.37,dtype=torch.float16))
    expected=np.float16(np.float32(np.float16(.37))/np.float32(65536))
    assert expected!=0
    np.testing.assert_array_equal(x.grad.numpy(),np.full(65536,expected,dtype=dtype))


def test_half_target_uses_float_opmath_and_nonfinite_values_propagate():
    data=np.linspace(-.9,1.1,16,dtype=np.float64)
    expected=np.float16(sum(map(float,data))/16)
    assert torch.mean(torch.from_numpy(data),dtype=torch.float16).item()==expected
    cancellation=np.tile([65504.,-65504.,1.,1.],2048)
    assert torch.mean(torch.tensor(cancellation,dtype=torch.float16)).item()==.5
    with np.errstate(all='raise'):
        assert torch.mean(torch.tensor([70000.,-70000.]),dtype=torch.float16).item()==0
        assert np.isnan(torch.mean(torch.tensor([np.inf,-np.inf])).item())
        assert np.isnan(torch.mean(torch.empty(0,3),dim=0).numpy()).all()


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_mean_casted_integer_bool_and_noncontiguous_inputs(dtype):
    assert torch.mean(torch.tensor([True,False,True]),dtype=dtype).item()==pytest.approx(2/3)
    np.testing.assert_array_equal(torch.mean(torch.tensor([[1,3],[5,7]]),dim=0,dtype=dtype).numpy(),[3.,5.])
    data=np.arange(12,dtype=np.float64).reshape(3,4).T
    x=torch.from_numpy(data).requires_grad_()
    y=torch.mean(x,dim=np.int64(1),keepdim=True,dtype=dtype)
    np.testing.assert_array_equal(y.numpy(),data.mean(axis=1,keepdims=True))
    y.sum().backward();np.testing.assert_allclose(x.grad.numpy(),np.full(data.shape,np.asarray(1/3,dtype=dtype)),rtol=1e-7)


@pytest.mark.parametrize('source,target', [(torch.float32,torch.float64),(torch.float64,torch.float32)])
def test_mean_jvp_duality_and_repeated_reverse_of_tangent(source,target):
    p=torch.tensor([[1.,2.],[3.,4.]],dtype=source,requires_grad=True)
    t=torch.tensor([[.2,.4],[.6,.8]],dtype=source,requires_grad=True)
    primal,tangent=torch.func.jvp(lambda x:torch.mean(x,dim=0,dtype=target),(p,),(t,))
    np.testing.assert_allclose(primal.numpy(),[2.,3.]);np.testing.assert_allclose(tangent.numpy(),[.4,.6])
    tangent.sum().backward();np.testing.assert_array_equal(t.grad.numpy(),np.full((2,2),.5,dtype=source))
    tangent.sum().backward();np.testing.assert_array_equal(t.grad.numpy(),np.ones((2,2),dtype=source))
    assert p.grad is None


def test_mean_saved_metadata_permits_mutation_and_releases_input_values():
    x=torch.tensor([2.,4.],requires_grad=True);out=torch.mean(x,dtype=torch.float64)
    with torch.no_grad():x.fill_(100.);out.detach().fill_(999.)
    out.backward();np.testing.assert_array_equal(x.grad.numpy(),[.5,.5])
    leaf=torch.ones(3,4,dtype=torch.float64,requires_grad=True)
    middle=leaf+1;weak=weakref.ref(middle._data)
    result=torch.mean(middle,dtype=torch.float32)
    del middle;gc.collect();assert weak() is None
    result.backward()
    np.testing.assert_array_equal(leaf.grad.numpy(),np.full((3,4),np.float32(1/12),dtype=np.float64))


def test_mean_strict_overloads_and_bounded_ad_leave_numpy_policy_intact():
    x=torch.ones(2,3)
    for kwargs in ({'keepdim':False},{'dim':True},{'dim':[False]},{'dim':1.},{'dim':0,'keepdim':1}):
        with pytest.raises(TypeError):torch.mean(x,**kwargs)
    with pytest.raises(RuntimeError):torch.mean(x,dim=(0,0))
    with pytest.raises(IndexError):torch.mean(x,dim=4)
    with pytest.raises(UnsupportedOperation):torch.mean(x,out=torch.zeros(1))
    with np.errstate(all='raise'):
        torch.mean(torch.ones(65536,dtype=torch.float16)).item()
        assert np.geterr()==dict(divide='raise',over='raise',under='raise',invalid='raise')
        # Pinned WASM NumPy can record this policy without trapping FPU flags.
        try:direct=np.divide(1.,0.)
        except FloatingPointError:pass
        else:assert np.isposinf(direct)
