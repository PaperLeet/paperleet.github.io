"""The same callback may alternate between legacy and norm HVP paths."""
import numpy as np
import pytest
from stridetorch import torch


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_same_callback_follows_each_actual_branch_once(dtype_name):
    dtype=getattr(torch,dtype_name)
    direction=np.array([.25,-.5,.75],dtype=dtype_name)
    calls=[]
    def function(value):
        calls.append(1)
        if value.sum().item()<0:
            return value.pow(3).sum()
        return torch.linalg.vector_norm(value).square()
    for count,sign in enumerate([-1.,1.,-1.,1.],1):
        point=np.array([1.,2.,3.],dtype=dtype_name)*sign
        x=torch.tensor(point,dtype=dtype,requires_grad=True)
        previous_grad=torch.full_like(x,9.);x.grad=previous_grad
        value,product=torch.autograd.functional.hvp(function,x,torch.tensor(direction,dtype=dtype))
        expected=6*point*direction if sign<0 else 2*direction
        np.testing.assert_allclose(product.detach().numpy(),expected,rtol=1e-5,atol=1e-6)
        assert len(calls)==count and type(value) is torch.Tensor and type(product) is torch.Tensor
        assert x.requires_grad and x.grad is previous_grad
        np.testing.assert_array_equal(x.detach().numpy(),point)
        np.testing.assert_array_equal(x.grad.numpy(),np.full(3,9.))
