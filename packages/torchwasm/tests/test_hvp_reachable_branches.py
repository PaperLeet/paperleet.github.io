"""Public HVP checks for discarded branches, shared values and numeric fallback."""
import numpy as np
import pytest
from torchwasm import torch


@pytest.mark.parametrize('dtype_name',['float32','float64'])
@pytest.mark.parametrize('kind',['polynomial','weighted_norm','shared_exp','matmul_fallback'])
@pytest.mark.parametrize('discarded_nonfinite',[False,True])
def test_discarded_norm_branches_preserve_hvp(dtype_name,kind,discarded_nonfinite):
    dtype=getattr(torch,dtype_name)
    point=np.arange(1,7,dtype=dtype_name).reshape(2,3)/4
    direction=np.arange(-2,4,dtype=dtype_name).reshape(2,3)/4
    coefficients=np.asarray([.5,1.,1.5],dtype=dtype_name)
    weights=np.asarray([[.5,-.25],[.25,1.],[1.,.5]],dtype=dtype_name)
    x=torch.tensor(point,dtype=dtype,requires_grad=True)
    old_grad=torch.full_like(x,9.);x.grad=old_grad
    coefficient=torch.tensor(coefficients,dtype=dtype)
    weight=torch.tensor(weights,dtype=dtype)
    calls=[]
    def function(value):
        calls.append(1)
        torch.linalg.vector_norm(value*(float('nan') if discarded_nonfinite else 2.))
        shared=value.exp()
        shared.sin().sum()
        if kind=='polynomial':return value.pow(3).sum()
        if kind=='weighted_norm':return torch.linalg.vector_norm(value*coefficient).square()
        if kind=='shared_exp':return torch.linalg.vector_norm(shared).square()
        return torch.linalg.vector_norm(value@weight).square()
    value,product=torch.autograd.functional.hvp(function,x,torch.tensor(direction,dtype=dtype))
    if kind=='polynomial':
        expected_value=np.sum(point**3);expected=6*point*direction
    elif kind=='weighted_norm':
        expected_value=np.sum((point*coefficients)**2);expected=2*coefficients**2*direction
    elif kind=='shared_exp':
        expected_value=np.sum(np.exp(2*point));expected=4*np.exp(2*point)*direction
    else:
        expected_value=np.sum((point@weights)**2);expected=2*(direction@weights)@weights.T
    np.testing.assert_allclose(value.detach().numpy(),expected_value,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(product.detach().numpy(),expected,rtol=1e-5,atol=1e-5)
    assert calls==[1] and type(value) is torch.Tensor and type(product) is torch.Tensor
    assert x.requires_grad and x.grad is old_grad
    np.testing.assert_array_equal(x.detach().numpy(),point)
    np.testing.assert_array_equal(x.grad.numpy(),np.full((2,3),9.,dtype=dtype_name))
