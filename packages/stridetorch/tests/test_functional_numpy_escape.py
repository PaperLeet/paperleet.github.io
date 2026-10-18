"""Native-compatible NumPy boundaries inside a functional reverse transform."""
import numpy as np
import pytest
from stridetorch import torch

@pytest.mark.parametrize('conversion',['numpy','asarray'])
@pytest.mark.parametrize('requires_grad',[False,True])
@pytest.mark.parametrize('enabled',[False,True])
def test_recorded_functional_numpy_escape_rejected(conversion,requires_grad,enabled):
    x=torch.tensor([2.,3.],requires_grad=requires_grad)
    def function(a):return torch.tensor(np.sin(a.numpy() if conversion=='numpy' else np.asarray(a)))
    with torch.set_grad_enabled(enabled):
        with pytest.raises(RuntimeError):torch.autograd.functional.vjp(function,x,torch.ones_like(x))
        assert torch.is_grad_enabled()==enabled

@pytest.mark.parametrize('conversion',['detach','no_grad','force'])
@pytest.mark.parametrize('requires_grad',[False,True])
@pytest.mark.parametrize('enabled',[False,True])
def test_explicit_numpy_disconnection_matches_native_zero_vjp(conversion,requires_grad,enabled):
    x=torch.tensor([2.,3.],requires_grad=requires_grad)
    def function(a):
        if conversion=='detach':data=a.detach().numpy()
        elif conversion=='force':data=a.numpy(force=True)
        else:
            with torch.no_grad():data=a.numpy()
        return torch.tensor(np.sin(data))
    with torch.set_grad_enabled(enabled):
        out,g=torch.autograd.functional.vjp(function,x,torch.ones_like(x))
        assert torch.is_grad_enabled()==enabled
    np.testing.assert_allclose(out.numpy(),np.sin([2.,3.]),rtol=1e-6,atol=1e-7);np.testing.assert_array_equal(g.numpy(),[0,0]);assert x.grad is None


def test_nested_transform_cannot_disable_outer_numpy_guard():
    def function(a):
        torch.autograd.functional.vjp(lambda b:b.square(),torch.ones_like(a),torch.ones_like(a))
        return torch.tensor(np.sin(a.numpy()))
    with pytest.raises(RuntimeError):torch.autograd.functional.vjp(function,torch.ones(2),torch.ones(2))
    # Existing direct-adapter exports outside a functional transform keep their contract.
    np.testing.assert_array_equal(torch.ones(2,requires_grad=True).numpy(),[1,1])

@pytest.mark.parametrize('force',[1,0,None,'x',np.bool_(True)])
def test_numpy_force_requires_a_python_bool(force):
    with pytest.raises(TypeError):torch.ones(2).numpy(force=force)


def test_numpy_force_is_keyword_only():
    with pytest.raises(TypeError):torch.ones(2).numpy(True)

@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('force',[False,True])
def test_numpy_export_dtype_and_storage(dtype,force):
    x=torch.tensor([[2.,3.],[4.,5.]],dtype=getattr(torch,dtype)).T
    array=x.numpy(force=force);assert array.dtype==np.dtype(dtype) and array.shape==(2,2);array[0,1]=17
    assert x[0,1].item()==17
