"""Public precision regressions, with native-observed fixed expectations."""
import numpy as np
import pytest
from stridetorch import torch

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_negative_half_power_keeps_signed_zero_and_infinity(dtype):
    x=torch.tensor([-0.,0.,-float('inf'),float('inf')],dtype=dtype)
    y=x.pow(-.5).numpy()
    assert np.isneginf(y[0]) and np.isposinf(y[1])
    assert np.isnan(y[2]) and y[3]==0 and not np.signbit(y[3])

@pytest.mark.parametrize('exponent',[np.float32(2.9),np.float64(2.9)])
def test_numpy_scalar_power_preserves_half_tensor_storage(exponent):
    x=torch.tensor([.25,1.,2.],dtype=torch.float16)
    y=x.pow(exponent)
    assert y.dtype==torch.float16
    np.testing.assert_array_equal(y.numpy(),[.0179443359375,1.,7.46484375])

@pytest.mark.parametrize('exponent',[1.99999999,2.00000001])
def test_public_power_and_functional_hvp_keep_existing_forward_values(exponent):
    x=torch.tensor([-.7]*8,dtype=torch.float32,requires_grad=True)
    v=torch.ones_like(x)*.25
    expected=np.full(8,np.float32(.49),dtype=np.float32)
    np.testing.assert_allclose(x.pow(exponent).detach().numpy(),expected,rtol=2e-7,atol=0)
    primal,product=torch.autograd.functional.hvp(lambda t:t.pow(exponent).sum(),x,v)
    np.testing.assert_allclose(primal.detach().numpy(),3.919999837875366,rtol=2e-7,atol=0)
    assert np.isnan(product.detach().numpy()).all()
