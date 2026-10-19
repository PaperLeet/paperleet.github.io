"""Public HVP graph barriers: nonlinear and shared-storage frozen terms."""
import numpy as np
import pytest
from stridetorch import torch

@pytest.mark.parametrize('dtype_name',['float32','float64'])
@pytest.mark.parametrize('barrier',['square_no_grad','view_no_grad','square_detach','view_detach'])
@pytest.mark.parametrize('output_kind',['square_sum','norm_square','norm'])
def test_frozen_upstream_term_does_not_contribute_to_input_hessian(dtype_name,barrier,output_kind):
    point=np.array([.75,1.25,2.],dtype=dtype_name)
    direction=np.array([.2,-.3,.4],dtype=dtype_name)
    x=torch.tensor(point,dtype=getattr(torch,dtype_name),requires_grad=True)
    v=torch.tensor(direction,dtype=getattr(torch,dtype_name))
    old_grad=torch.full_like(x,7.);x.grad=old_grad;calls=[]
    def function(value):
        calls.append(1)
        if barrier.endswith('no_grad'):
            with torch.no_grad():
                frozen=value.square() if barrier.startswith('square') else value.reshape(1,3).reshape(3)
        else:
            frozen=(value.square() if barrier.startswith('square') else value.reshape(1,3).reshape(3)).detach()
        source=value+frozen
        if output_kind=='square_sum':return source.square().sum()
        norm=torch.linalg.vector_norm(source)
        return norm.square() if output_kind=='norm_square' else norm
    value,product=torch.autograd.functional.hvp(function,x,v)
    if output_kind=='norm':
        source=point+point**2 if barrier.startswith('square') else 2*point
        source=source.astype('float64');vector=direction.astype('float64');length=np.linalg.norm(source)
        expected=vector/length-source*np.dot(source,vector)/length**3
    else:expected=2*direction
    tolerance=2e-5 if dtype_name=='float32' else 2e-12
    np.testing.assert_allclose(product.detach().numpy(),expected,rtol=tolerance,atol=tolerance)
    assert len(calls)==1 and x.grad is old_grad and x.requires_grad
    np.testing.assert_array_equal(x.detach().numpy(),point)
    np.testing.assert_array_equal(x.grad.detach().numpy(),np.full_like(point,7.))
    assert type(value) is torch.Tensor and type(product) is torch.Tensor
    assert not value.requires_grad and not product.requires_grad


@pytest.mark.parametrize('dtype_name',['float32','float64'])
@pytest.mark.parametrize('barrier',['square_no_grad','view_no_grad','square_detach','view_detach'])
@pytest.mark.parametrize('output_kind',['square_sum','norm'])
@pytest.mark.parametrize('entry',['direct','functional'])
def test_jvp_preserves_native_forward_and_reverse_based_barrier_semantics(dtype_name,barrier,output_kind,entry):
    point=np.array([.75,1.25,2.],dtype=dtype_name)
    direction=np.array([.2,-.3,.4],dtype=dtype_name)
    x=torch.tensor(point,dtype=getattr(torch,dtype_name),requires_grad=True)
    v=torch.tensor(direction,dtype=getattr(torch,dtype_name))
    old_grad=torch.full_like(x,7.);x.grad=old_grad;calls=[]
    def function(value):
        calls.append(1)
        if barrier.endswith('no_grad'):
            with torch.no_grad():
                frozen=value.square() if barrier.startswith('square') else value.reshape(1,3).reshape(3)
        else:
            frozen=(value.square() if barrier.startswith('square') else value.reshape(1,3).reshape(3)).detach()
        source=value+frozen
        return source.square().sum() if output_kind=='square_sum' else torch.linalg.vector_norm(source)
    if entry=='direct':value,tangent=torch.func.jvp(function,(x,),(v,))
    else:value,tangent=torch.autograd.functional.jvp(function,x,v)
    source=(point+point**2 if barrier.startswith('square') else 2*point).astype('float64')
    delta=direction.astype('float64')
    if entry=='direct' and barrier.endswith('no_grad'):
        delta=delta*(1+2*point.astype('float64') if barrier.startswith('square') else 2)
    if output_kind=='square_sum':expected_value=np.dot(source,source);expected_tangent=2*np.dot(source,delta)
    else:expected_value=np.linalg.norm(source);expected_tangent=np.dot(source,delta)/expected_value
    tolerance=2e-5 if dtype_name=='float32' else 2e-12
    np.testing.assert_allclose(value.detach().numpy(),expected_value,rtol=tolerance,atol=tolerance)
    np.testing.assert_allclose(tangent.detach().numpy(),expected_tangent,rtol=tolerance,atol=tolerance)
    assert len(calls)==1 and x.grad is old_grad and x.requires_grad
    np.testing.assert_array_equal(x.detach().numpy(),point)
    np.testing.assert_array_equal(x.grad.detach().numpy(),np.full_like(point,7.))
