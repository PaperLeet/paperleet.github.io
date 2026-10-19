"""Public numerical backprop remains usable after HVP success and reverse errors."""
import numpy as np
import pytest
from stridetorch import torch


@pytest.mark.parametrize('dtype_name',['float32','float64'])
@pytest.mark.parametrize('outcome',['success','callback_failure','strict_independent'])
def test_numerical_backward_after_hvp(dtype_name,outcome):
    dtype=getattr(torch,dtype_name)
    x=torch.tensor([.5,1.,1.5],dtype=dtype,requires_grad=True)
    direction=torch.tensor([.25,-.5,.75],dtype=dtype)
    coefficient=torch.tensor([1.,1.5,2.],dtype=dtype)
    previous_grad=torch.full_like(x,9.);x.grad=previous_grad;calls=[]
    def function(value):
        calls.append(1)
        if outcome=='strict_independent':return torch.linalg.vector_norm(value,ord=0)
        output=torch.linalg.vector_norm(value*coefficient).square()
        if outcome=='callback_failure':
            raise RuntimeError('user callback failure')
        return output
    if outcome=='success':
        value,product=torch.autograd.functional.hvp(function,x,direction)
        assert not value.requires_grad and not product.requires_grad
        np.testing.assert_allclose(product.detach().numpy(),np.asarray([.5,-2.25,6.],dtype=dtype_name),rtol=1e-5,atol=1e-5)
    elif outcome=='callback_failure':
        with pytest.raises(RuntimeError,match='user callback failure'):
            torch.autograd.functional.hvp(function,x,direction)
    else:
        with pytest.raises(RuntimeError,match='independent'):
            torch.autograd.functional.hvp(function,x,direction,strict=True)
    assert calls==[1] and x.grad is previous_grad
    np.testing.assert_array_equal(x.grad.numpy(),np.full(3,9.,dtype=dtype_name))
    probe=torch.tensor([.5,-1.,2.],dtype=dtype,requires_grad=True)
    (probe*probe+.25).sum().backward()
    np.testing.assert_array_equal(probe.grad.numpy(),np.asarray([1.,-2.,4.],dtype=dtype_name))
