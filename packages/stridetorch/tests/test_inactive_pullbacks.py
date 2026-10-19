"""Preserve finite public HVPs while removing unused private pullbacks."""
import math
import numpy as np
import pytest
from stridetorch import torch


@pytest.mark.parametrize('dtype_name,norm_target,rtol',[
    ('float32',87.5,5e-5),('float64',709.,2e-10)])
@pytest.mark.parametrize('coefficient_kind',['number','numpy','tensor'])
def test_large_broadcast_offset_hvp_stays_finite_and_preserves_callers(
        dtype_name,norm_target,rtol,coefficient_kind):
    dtype=getattr(torch,dtype_name)
    x=torch.full((32,),norm_target/math.sqrt(32),dtype=dtype,requires_grad=True)
    x.grad=torch.full_like(x,7.)
    coefficient=(torch.tensor(0.,dtype=dtype) if coefficient_kind=='tensor' else
                 np.float64(0.) if coefficient_kind=='numpy' else 0.)
    original_type=type(coefficient)
    calls=[]
    def function(value):
        calls.append(1)
        return torch.linalg.vector_norm(value+coefficient).exp()
    # The requested gradient components and this scaled Hessian direction are
    # finite. The unrequested scalar contraction would overflow. NumericalScope
    # masks private kernel warnings, so constructor/work counts are observed
    # separately; this test protects values and the caller error policy.
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        value,product=torch.autograd.functional.hvp(function,x,torch.full_like(x,1/32))
        assert np.geterr()==dict(over='raise',invalid='raise',divide='raise',under='ignore')
    assert calls==[1]
    assert type(coefficient) is original_type
    assert np.isfinite(value.detach().numpy()).all()
    assert np.isfinite(product.detach().numpy()).all()
    expected=math.exp(norm_target)/32
    np.testing.assert_allclose(product.detach().numpy(),expected,rtol=rtol,atol=0)
    np.testing.assert_array_equal(x.grad.numpy(),np.full(32,7.))
    if coefficient_kind=='tensor':
        assert not coefficient.requires_grad and coefficient.grad is None


@pytest.mark.parametrize('dtype_name,large,small',[
    ('float32',1e37,1e-37),('float64',1e307,1e-307)])
def test_large_fixed_multiplier_hvp_stays_finite(
        dtype_name,large,small):
    dtype=getattr(torch,dtype_name)
    x=torch.full((64,),large,dtype=dtype,requires_grad=True)
    coefficient=torch.tensor(small,dtype=dtype)
    original_type=type(coefficient)
    calls=[]
    def function(value):
        calls.append(1)
        return torch.linalg.vector_norm(value*coefficient,ord=.5)
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        value,product=torch.autograd.functional.hvp(function,x,torch.ones_like(x))
        assert np.geterr()==dict(over='raise',invalid='raise',divide='raise',under='ignore')
    assert calls==[1] and type(coefficient) is original_type
    assert np.isfinite(value.detach().numpy()).all()
    assert np.isfinite(product.detach().numpy()).all()
    # The direction is radial; a degree-one homogeneous norm has zero radial
    # curvature, including after multiplication by a fixed scalar.
    np.testing.assert_allclose(product.detach().numpy(),0,rtol=0,
                               atol=np.finfo(dtype_name).tiny)
    assert not coefficient.requires_grad and coefficient.grad is None
