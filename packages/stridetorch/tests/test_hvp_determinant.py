"""Cofactor-minor references independent of the SVD implementation."""
import numpy as np
import pytest
from stridetorch import torch

def minor(a,i,j):return np.delete(np.delete(a,i,axis=0),j,axis=1)

def cofactor_reference(a,e):
    a=a.astype(np.float64);e=e.astype(np.float64);n=len(a)
    cofactor=np.empty_like(a);curvature=np.empty_like(a)
    for i in range(n):
        for j in range(n):
            block=minor(a,i,j);direction=minor(e,i,j)
            cofactor[i,j]=(-1.)**(i+j)*np.linalg.det(block)
            value=0.
            for k in range(n-1):
                for l in range(n-1):
                    value+=(-1.)**(i+j+k+l)*np.linalg.det(minor(block,k,l))*direction[k,l]
            curvature[i,j]=value
    return cofactor,curvature

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('n',[2,3,4])
@pytest.mark.parametrize('strided',[False,True])
@pytest.mark.parametrize('squared',[False,True])
def test_rank_deficient_det_hvp_matches_polynomial_minors(dtype,n,strided,squared):
    rng=np.random.default_rng(7103+n)
    left=np.linalg.qr(rng.normal(size=(n,n)))[0];right=np.linalg.qr(rng.normal(size=(n,n)))[0]
    data=(left@np.diag([0.,*range(2,n+1)])@right.T).astype(dtype)
    direction=rng.normal(size=data.shape).astype(dtype)
    if strided:
        storage=np.empty(data.shape+(2,),dtype=dtype);storage[...,0]=data;storage[...,1]=-7;data=storage[...,0]
    c,h=cofactor_reference(data,direction)
    determinant=np.linalg.det(data.astype(np.float64));first=np.sum(c*direction)
    expected=2*first*c+2*determinant*h if squared else h
    x=torch.from_numpy(data).requires_grad_(True);v=torch.tensor(direction,requires_grad=True)
    x.grad=torch.ones_like(x)*3;v.grad=torch.ones_like(v)*4;buffers=(x.grad,v.grad)
    calls=[];hooks=[]
    x.register_hook(lambda g:hooks.append(g));v.register_hook(lambda g:hooks.append(g))
    def function(a):
        calls.append(torch.is_grad_enabled());value=torch.det(a)
        return value.square() if squared else value
    with torch.no_grad():
        output,actual=torch.autograd.functional.hvp(function,x,v,strict=True)
        assert not torch.is_grad_enabled()
    np.testing.assert_allclose(actual.numpy(),expected,rtol=3e-5 if dtype==np.float32 else 2e-10,atol=5e-6 if dtype==np.float32 else 2e-11)
    assert actual.dtype==np.dtype(dtype) and not actual.requires_grad and not output.requires_grad
    assert x.grad is buffers[0] and v.grad is buffers[1] and not hooks and calls==[True]
    np.testing.assert_array_equal(x.detach().numpy(),data)
    np.testing.assert_array_equal(v.detach().numpy(),direction)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('squared',[False,True])
def test_batched_mixed_det_routes_have_cross_batch_curvature(dtype,squared):
    data=np.array([[[0.,0.,0.],[0.,2.,0.],[0.,0.,3.]],
        [[2.,.2,0.],[.1,3.,.3],[0.,.4,4.]],
        [[1.,2.,3.],[2.,4.,6.],[0.,1.,1.]]],dtype=dtype)
    direction=np.cos(np.arange(data.size)*.13).reshape(data.shape).astype(dtype)
    weights=np.array([.7,-.2,.4],dtype=dtype)
    pairs=[cofactor_reference(a,e) for a,e in zip(data,direction)]
    value=np.sum(weights*np.linalg.det(data.astype(np.float64)))
    first=sum(float(w*np.sum(c*e)) for w,(c,h),e in zip(weights,pairs,direction))
    expected=np.stack([w*(2*first*c+2*value*h if squared else h) for w,(c,h) in zip(weights,pairs)])
    def function(a):
        result=(torch.det(a)*torch.tensor(weights)).sum()
        return result.square() if squared else result
    _,actual=torch.autograd.functional.hvp(function,torch.tensor(data),torch.tensor(direction),strict=True)
    np.testing.assert_allclose(actual.numpy(),expected,rtol=4e-5 if dtype==np.float32 else 1e-9,atol=2e-5 if dtype==np.float32 else 2e-10)

@pytest.mark.parametrize('shape',[(0,0),(0,2,2),(2,0,0),(0,1,1)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('squared',[False,True])
def test_empty_det_backward_stays_independent_in_strict_hvp(shape,dtype,squared):
    x=torch.empty(shape,dtype=dtype);v=torch.ones_like(x);calls=[]
    def function(a):
        calls.append(1);value=torch.det(a).sum()
        return value.square() if squared else value
    _,value=torch.autograd.functional.hvp(function,x,v)
    assert value.shape==shape and value.numel()==0 and not value.requires_grad
    with pytest.raises(RuntimeError):torch.autograd.functional.hvp(function,x,v,strict=True)
    assert calls==[1,1]

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('kind',['two_zero','rank_one_three','tiny_identity'])
@pytest.mark.parametrize('squared',[False,True])
def test_degenerate_singular_values_keep_polynomial_curvature_and_numpy_policy(dtype,kind,squared):
    if kind=='two_zero':data=np.zeros((2,2),dtype=dtype)
    elif kind=='rank_one_three':data=np.diag(np.array([0.,0.,3.],dtype=dtype))
    else:data=np.eye(2,dtype=dtype)*np.sqrt(np.finfo(dtype).eps)
    x=torch.tensor(data);v=torch.ones_like(x)
    cofactor,curvature=cofactor_reference(data,np.ones_like(data))
    expected=2*np.sum(cofactor)*cofactor+2*np.linalg.det(data.astype(np.float64))*curvature if squared else curvature
    with np.errstate(all='raise'):
        _,value=torch.autograd.functional.hvp(lambda a:torch.det(a).square() if squared else torch.det(a),x,v,strict=True)
        assert all(v=='raise' for v in np.geterr().values())
    assert np.isfinite(value.numpy()).all()
    np.testing.assert_allclose(value.numpy(),expected,rtol=1e-5,atol=np.finfo(dtype).tiny)
