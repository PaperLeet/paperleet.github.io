"""Cosine API values and both differentiation channels against independent native PyTorch."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


def pair(data, dtype=np.float64):
    data=np.asarray(data,dtype=dtype)
    return wasm.Tensor(data.copy(),requires_grad=True),torch.tensor(data.copy(),requires_grad=True)


def close(actual,expected,dtype=np.float64):
    tolerance=8e-6 if dtype==np.float32 else 3e-12
    np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),atol=tolerance,rtol=tolerance,equal_nan=True)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape1,shape2,dim',[
    ((2,3),(2,3),1),((2,1,4),(1,3,4),-1),((4,),(2,3,4),2),
    ((3,2,4),(3,1,4),0),((2,0),(1,0),1),((0,3),(1,3),-1),((),(),0),
])
def test_cosine_broadcast_values_and_both_input_gradients(dtype,shape1,shape2,dim):
    rng=np.random.RandomState(410)
    x,nx=pair(np.asarray(rng.randn(*shape1) if shape1 else -.7),dtype)
    y,ny=pair(np.asarray(rng.randn(*shape2) if shape2 else .4),dtype)
    actual=wasm.cosine_similarity(x,y,dim);expected=torch.cosine_similarity(nx,ny,dim)
    weights=np.asarray(rng.randn(*actual.shape) if actual.shape else -.8,dtype=dtype)
    (actual*wasm.Tensor(weights)).sum().backward();(expected*torch.tensor(weights)).sum().backward()
    close(actual,expected,dtype);close(x.grad,nx.grad,dtype);close(y.grad,ny.grad,dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('eps',[0.,1.,float('inf')])
def test_cosine_zero_sub_epsilon_boundary_and_nonfinite_semantics(dtype,eps):
    data=[[0.,0.],[.3,.4],[.6,.8],[1.2,1.6]]
    x,nx=pair(data,dtype);y,ny=pair([[1.,2.]]*4,dtype)
    with np.errstate(all='ignore'):
        actual=wasm.cosine_similarity(x,y,1,eps);expected=torch.cosine_similarity(nx,ny,1,eps)
        actual.sum().backward();expected.sum().backward()
    close(actual,expected,dtype);close(x.grad,nx.grad,dtype);close(y.grad,ny.grad,dtype)
    if eps==1.:
        naive=torch.nn.functional.normalize(nx,dim=1,eps=eps)*torch.nn.functional.normalize(ny,dim=1,eps=eps)
        naive_grad=torch.autograd.grad(naive.sum(),nx)[0]
        assert not torch.allclose(nx.grad[1],naive_grad[1]),'The epsilon-region reverse rule must not be replaced by normalize/clamp derivatives'


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('row',[[0.,0.],[.3,.4],[.6,.8],[1.2,1.6]])
def test_cosine_jvp_and_reverse_gradients_of_both_channels_at_epsilon(dtype,row):
    a,na=pair(row,dtype);b,nb=pair([1.,2.],dtype)
    ta,nta=pair([.7,-.2],dtype);tb,ntb=pair([-.7,.2],dtype)
    with np.errstate(all='ignore'):
        primal,tangent=wasm.jvp(lambda x,y:wasm.cosine_similarity(x,y,0,1.),(a,b),(ta,tb))
        expected,jvp=torch.func.jvp(lambda x,y:torch.cosine_similarity(x,y,0,1.),(na,nb),(nta,ntb))
        actual_grad=wasm.autograd_grad(primal,(a,b));expected_grad=torch.autograd.grad(expected,(na,nb),retain_graph=True)
        actual_second=wasm.autograd_grad(tangent,(a,b,ta,tb));expected_second=torch.autograd.grad(jvp,(na,nb,nta,ntb))
    close(primal,expected,dtype);close(tangent,jvp,dtype)
    for got,want in zip(actual_grad,expected_grad):close(got,want,dtype)
    for got,want in zip(actual_second,expected_second):close(got,want,dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_cosine_jvp_broadcast_noncontiguous_inputs_and_frozen_argument(dtype):
    rng=np.random.RandomState(411)
    a,na=pair(rng.randn(2,1,3),dtype);b,nb=pair(rng.randn(1,4,3),dtype)
    ta,nta=pair(rng.randn(2,1,3),dtype);tb,ntb=pair(rng.randn(1,4,3),dtype)
    p,t=wasm.jvp(lambda x,y:wasm.cosine_similarity(x,y,-1),(a,b),(ta,tb))
    np_,nt=torch.func.jvp(lambda x,y:torch.cosine_similarity(x,y,-1),(na,nb),(nta,ntb))
    close(p,np_,dtype);close(t,nt,dtype)
    got=wasm.autograd_grad((p.square()+t.square()).sum(),(a,b,ta,tb));want=torch.autograd.grad((np_.square()+nt.square()).sum(),(na,nb,nta,ntb))
    for x,y in zip(got,want):close(x,y,dtype)
    base=rng.randn(3,8).astype(dtype);other=rng.randn(4).astype(dtype)
    a=wasm.Tensor(base[:,::2],requires_grad=True);na=torch.tensor(base)[:,::2].requires_grad_()
    b=wasm.Tensor(other);nb=torch.tensor(other)
    actual=wasm.cosine_similarity(a,b,-1);expected=torch.cosine_similarity(na,nb,-1)
    actual.sum().backward();expected.sum().backward();close(actual,expected,dtype);close(a.grad,na.grad,dtype)
    direction=wasm.Tensor(np.ones(a.shape,dtype=dtype));native_direction=torch.ones_like(na)
    _,t=wasm.jvp(lambda x:wasm.cosine_similarity(x,b,-1),(a,),(direction,))
    _,nt=torch.func.jvp(lambda x:torch.cosine_similarity(x,nb,-1),(na,),(native_direction,));close(t,nt,dtype)
    assert a._data.strides==base[:,::2].strides


@pytest.mark.parametrize('left,right',[(np.float32,np.float64),(np.float64,np.float32),(np.int64,np.float32),(np.bool_,np.float64)])
def test_cosine_mixed_floating_and_integral_type_promotion(left,right):
    x=np.asarray([[1,0,2],[3,1,0]],dtype=left);y=np.asarray([.4,-.5,.9],dtype=right)
    a=wasm.Tensor(x,requires_grad=x.dtype.kind=='f');na=torch.tensor(x,requires_grad=x.dtype.kind=='f')
    b=wasm.Tensor(y,requires_grad=True);nb=torch.tensor(y,requires_grad=True)
    actual=wasm.cosine_similarity(a,b);expected=torch.cosine_similarity(na,nb)
    actual.sum().backward();expected.sum().backward()
    close(actual,expected,np.float32 if right==np.float32 else np.float64);assert actual.dtype==expected.detach().numpy().dtype
    close(b.grad,nb.grad,np.float32 if right==np.float32 else np.float64)
    if x.dtype.kind=='f':close(a.grad,na.grad,np.float32 if left==np.float32 else np.float64)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_cosine_aliased_inputs_and_saved_tensor_mutation(dtype):
    a,na=pair([[.3,.4],[1.2,1.6]],dtype)
    actual=wasm.cosine_similarity(a,a,1,1.);expected=torch.cosine_similarity(na,na,1,1.)
    actual.sum().backward();expected.sum().backward();close(actual,expected,dtype);close(a.grad,na.grad,dtype)
    a,_=pair([[.3,.4],[1.2,1.6]],dtype);b,_=pair([[1.,2.],[2.,1.]],dtype)
    output=wasm.cosine_similarity(a,b)
    with wasm.no_grad():a[0,0]=5.
    with pytest.raises(wasm.BrowserUnsupported,match='modified in-place'):output.sum().backward()


@pytest.mark.parametrize('eps',[-1.,float('nan')])
def test_cosine_invalid_epsilon_is_a_normal_native_error(eps):
    with pytest.raises(RuntimeError):wasm.cosine_similarity(wasm.Tensor([1.]),wasm.Tensor([1.]),0,eps)
    with pytest.raises(RuntimeError):torch.cosine_similarity(torch.tensor([1.]),torch.tensor([1.]),0,eps)


def test_cosine_dimension_shape_dtype_and_unvalidated_modes():
    a=wasm.Tensor([[1.,2.]],dtype=np.float32)
    for dim in (-3,2):
        with pytest.raises(IndexError):wasm.cosine_similarity(a,a,dim)
    with pytest.raises(TypeError):wasm.cosine_similarity(a,a,True)
    with pytest.raises(RuntimeError):wasm.cosine_similarity(a,wasm.Tensor(np.ones((2,3),np.float32)))
    with pytest.raises(RuntimeError):wasm.cosine_similarity(wasm.Tensor([1,2]),wasm.Tensor([2,3]),0)
    with pytest.raises(RuntimeError):wasm.cosine_similarity(wasm.Tensor([1j]),wasm.Tensor([1.]),0)
    with pytest.raises(wasm.BrowserUnsupported,match='float32/float64'):wasm.cosine_similarity(wasm.Tensor(np.ones(2,np.float16)),a,0)
    with pytest.raises(wasm.BrowserUnsupported,match='Nested JVP'):wasm.jvp(lambda x:wasm.jvp(lambda y:wasm.cosine_similarity(y,y,0),(x,),(x,))[0],(a,),(a,))


def test_installed_cosine_aliases_module_and_jvp_do_not_silently_return_zero_tangents():
    code='''
import numpy as np
from paperleet.browser.numpy_autograd import install
import sys
sys.modules["torch"]=install()
import torch
import torch.nn.functional as F
x=torch.tensor([[.3,.4],[1.2,1.6]],dtype=torch.float64,requires_grad=True)
y=torch.tensor([[1.,2.],[2.,1.]],dtype=torch.float64,requires_grad=True)
expected=F.cosine_similarity(x,y,dim=1,eps=1.)
np.testing.assert_allclose(torch.cosine_similarity(x,y,dim=1,eps=1.).numpy(),expected.numpy())
module=torch.nn.CosineSimilarity(dim=1,eps=1.)
assert list(module.parameters())==[]
np.testing.assert_allclose(module(x,y).numpy(),expected.numpy())
for function in (torch.cosine_similarity,F.cosine_similarity,module):
 call=(lambda a,b:function(a,b)) if function is module else (lambda a,b:function(a,b,dim=1,eps=1.))
 primal,tangent=torch.func.jvp(call,(x,y),(torch.ones_like(x),-torch.ones_like(y)))
 assert np.any(tangent.numpy()!=0)
 np.testing.assert_allclose(primal.numpy(),expected.numpy())
 tangent.sum().backward()
 assert x.grad is not None and y.grad is not None
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env={**os.environ},timeout=20)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('data',[
    [[0.,0.],[3e-9,4e-9],[6e-9,8e-9]],
    [[1e20,-1e20],[float('inf'),1.],[float('nan'),1.]],
])
def test_cosine_default_epsilon_tiny_vectors_extreme_values_and_nonfinites(dtype,data):
    x,nx=pair(data,dtype);y,ny=pair([[1.,1.]],dtype)
    with np.errstate(all='ignore'):
        actual=wasm.cosine_similarity(x,y);expected=torch.cosine_similarity(nx,ny)
        actual.sum().backward();expected.sum().backward()
    close(actual,expected,dtype);close(x.grad,nx.grad,dtype);close(y.grad,ny.grad,dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_cosine_scalar_jvp_and_reverse_over_forward(dtype):
    x,nx=pair(.3,dtype);y,ny=pair(2.,dtype);t,nt=pair(.7,dtype);u,nu=pair(-.2,dtype)
    p,j=wasm.jvp(lambda a,b:wasm.cosine_similarity(a,b,0,1.),(x,y),(t,u))
    expected,tangent=torch.func.jvp(lambda a,b:torch.cosine_similarity(a,b,0,1.),(nx,ny),(nt,nu))
    close(p,expected,dtype);close(j,tangent,dtype)
    got=wasm.autograd_grad(j,(x,y,t,u));want=torch.autograd.grad(tangent,(nx,ny,nt,nu))
    for a,b in zip(got,want):close(a,b,dtype)
