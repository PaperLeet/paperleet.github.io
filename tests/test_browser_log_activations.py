"""Independent native parity for Softplus/LogSigmoid values and both AD channels."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from paperleet.browser import numpy_autograd as wasm


def pair(data,dtype):
    data=np.asarray(data,dtype=dtype)
    return wasm.Tensor(data.copy(),requires_grad=True),torch.tensor(data.copy(),requires_grad=True)


def close(actual,expected,dtype):
    tolerance=4e-6 if dtype==np.float32 else 2e-12
    np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),rtol=tolerance,atol=tolerance,equal_nan=True)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('beta,threshold',[
    (1.,20.),(.5,2.),(3.,10.),(2.,1.),(-2.,1.),(0.,20.),(-0.,20.),
    (1.,1000.),(1.,float('nan')),(float('nan'),20.),
    (float('inf'),20.),(-float('inf'),20.),(1.,float('inf')),(1.,-float('inf')),
])
def test_softplus_native_values_vjp_jvp_and_reverse_of_both_channels(dtype,beta,threshold):
    values=[-float('inf'),-1000.,-100.,-40.,-20.,-1.,-.5,0.,.5,1.,20.,40.,100.,1000.,float('inf'),float('nan')]
    x,nx=pair(values,dtype);direction,nd=pair(np.linspace(-.7,1.1,len(values)),dtype)
    with np.errstate(all='ignore'):
        actual=wasm.softplus(x,beta,threshold);expected=F.softplus(nx,beta,threshold)
        gradient=wasm.autograd_grad(actual.sum(),(x,))[0];native_gradient=torch.autograd.grad(expected.sum(),nx)[0]
        primal,tangent=wasm.jvp(lambda a:wasm.softplus(a,beta,threshold),(x,),(direction,))
        native_primal,native_tangent=torch.func.jvp(lambda a:F.softplus(a,beta,threshold),(nx,),(nd,))
        actual_grad=wasm.autograd_grad(tangent.sum(),(x,direction));native_grad=torch.autograd.grad(native_tangent.sum(),(nx,nd))
    close(actual,expected,dtype);close(gradient,native_gradient,dtype)
    close(primal,native_primal,dtype);close(tangent,native_tangent,dtype)
    for a,b in zip(actual_grad,native_grad):close(a,b,dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('activation',['softplus','logsigmoid'])
@pytest.mark.parametrize('shape',[(),(0,),(2,0,3),(2,3)])
def test_log_activations_shape_precision_and_empty_or_scalar_ad(dtype,activation,shape):
    data=np.full(shape,.25,dtype=dtype)
    x,nx=pair(data,dtype);direction,nd=pair(np.ones(shape,dtype=dtype),dtype)
    operation=(lambda a:wasm.softplus(a)) if activation=='softplus' else (lambda a:a.logsigmoid())
    native=F.softplus if activation=='softplus' else F.logsigmoid
    primal,tangent=wasm.jvp(operation,(x,),(direction,));np_,nt=torch.func.jvp(native,(nx,),(nd,))
    close(primal,np_,dtype);close(tangent,nt,dtype);assert primal.shape==shape and tangent.dtype==np.dtype(dtype)
    got=wasm.autograd_grad((primal+tangent).sum(),(x,direction));want=torch.autograd.grad((np_+nt).sum(),(nx,nd))
    for a,b in zip(got,want):close(a,b,dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_logsigmoid_extreme_values_jvp_and_native_tail_curvature(dtype):
    data=[-float('inf'),-1000.,-100.,-40.,-20.,-1.,0.,1.,20.,40.,100.,1000.,float('inf'),float('nan')]
    x,nx=pair(data,dtype);v,nv=pair(np.ones(len(data)),dtype)
    with np.errstate(all='ignore'):
        actual=x.logsigmoid();expected=F.logsigmoid(nx)
        gx=wasm.autograd_grad(actual.sum(),(x,))[0];ng=torch.autograd.grad(expected.sum(),nx)[0]
        p,j=wasm.jvp(lambda a:a.logsigmoid(),(x,),(v,));np_,nj=torch.func.jvp(F.logsigmoid,(nx,),(nv,))
        got=wasm.autograd_grad(j.sum(),(x,v));want=torch.autograd.grad(nj.sum(),(nx,nv))
    for a,b in [(actual,expected),(gx,ng),(p,np_),(j,nj),*zip(got,want)]:close(a,b,dtype)
    # Absolute tolerances must not conceal saturation on the wrong tail.
    indices=[3,4,8,9]
    np.testing.assert_allclose(got[0].numpy()[indices],want[0].detach().numpy()[indices],atol=0.,rtol=4e-6 if dtype==np.float32 else 2e-12)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_softplus_threshold_first_and_second_rules_are_distinct(dtype):
    x,nx=pair([.4999,.5,.5001],dtype);v,nv=pair([1.,1.,1.],dtype)
    p,j=wasm.jvp(lambda a:wasm.softplus(a,beta=2.,threshold=1.),(x,),(v,))
    np_,nj=torch.func.jvp(lambda a:F.softplus(a,beta=2.,threshold=1.),(nx,),(nv,))
    got=wasm.autograd_grad(j.sum(),(x,))[0];want=torch.autograd.grad(nj.sum(),nx)[0]
    close(p,np_,dtype);close(j,nj,dtype);close(got,want,dtype)
    assert got.numpy()[0]>0 and got.numpy()[1]==got.numpy()[2]==0
    assert j.numpy()[1]<1 and j.numpy()[2]==1


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('activation',['softplus','logsigmoid'])
def test_log_activations_noncontiguous_storage_and_saved_source_mutation(dtype,activation):
    base=np.linspace(-3,3,24,dtype=dtype).reshape(3,8)
    x=wasm.Tensor(base[:,::2],requires_grad=True);nx=torch.tensor(base)[:,::2].requires_grad_()
    op=(lambda a:wasm.softplus(a,.7,2.)) if activation=='softplus' else (lambda a:a.logsigmoid())
    native=(lambda a:F.softplus(a,.7,2.)) if activation=='softplus' else F.logsigmoid
    actual=op(x);expected=native(nx);actual.square().sum().backward();expected.square().sum().backward()
    close(actual,expected,dtype);close(x.grad,nx.grad,dtype);assert x._data.strides==base[:,::2].strides
    fresh=wasm.Tensor(base.copy(),requires_grad=True);output=op(fresh)
    with wasm.no_grad():fresh[0,0]=.9
    with pytest.raises(wasm.BrowserUnsupported,match='modified in-place'):output.sum().backward()


def test_softplus_overflow_is_not_replaced_by_a_finite_approximation():
    x,nx=pair([1000.],np.float64)
    with np.errstate(all='ignore'):
        value=wasm.softplus(x,threshold=1000.);native=F.softplus(nx,threshold=1000.)
        value.sum().backward();native.sum().backward()
    assert np.isposinf(value.numpy()).all() and torch.isposinf(native).all()
    assert np.isnan(x.grad.numpy()).all() and torch.isnan(nx.grad).all()


def test_log_activation_invalid_types_and_explicit_unvalidated_modes():
    for data in ([1,2],[True,False],[1j,2j]):
        for operation in (lambda a:wasm.softplus(a),lambda a:a.logsigmoid()):
            with pytest.raises(RuntimeError):operation(wasm.Tensor(data))
    for operation in (lambda a:wasm.softplus(a),lambda a:a.logsigmoid()):
        with pytest.raises(wasm.BrowserUnsupported,match='float32/float64'):operation(wasm.Tensor(np.ones(2,np.float16)))
    with pytest.raises(wasm.BrowserUnsupported,match='Tensor-valued'):wasm.softplus(wasm.Tensor([1.]),beta=wasm.Tensor(2.))
    with pytest.raises(TypeError):wasm.softplus(wasm.Tensor([1.]),beta='bad')
    with pytest.raises(TypeError):wasm.softplus([1.])


def test_log_activation_installed_modules_functional_jvp_and_sequential():
    script='''
import sys,numpy as np
from paperleet.browser.numpy_autograd import install
sys.modules['torch']=install()
import torch
from torch.nn import functional as F
x=torch.tensor([[-2.,-.5,0.,.5,2.]],dtype=torch.float64,requires_grad=True)
v=torch.ones_like(x).requires_grad_()
for functional,module in [(F.softplus,torch.nn.Softplus()),(F.logsigmoid,torch.nn.LogSigmoid())]:
 np.testing.assert_allclose(functional(x).numpy(),module(x).numpy())
 p,j=torch.func.jvp(module,(x,),(v,))
 assert np.any(j.numpy()!=0)
 g=torch.autograd.grad((p+j).sum(),(x,v))
 assert all(value is not None for value in g)
 assert list(module.parameters())==[]
layer=torch.nn.Sequential(torch.nn.Softplus(),torch.nn.LogSigmoid())
np.testing.assert_allclose(layer(x).numpy(),F.logsigmoid(F.softplus(x)).numpy())
p,j=torch.func.jvp(layer,(x,),(v,)); assert j.shape==x.shape
with torch.no_grad():
 p,j=torch.func.jvp(F.softplus,(x,),(v,)); assert not p.requires_grad and not j.requires_grad
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,env={**os.environ},timeout=20)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('activation',['softplus','logsigmoid'])
@pytest.mark.parametrize('infinite',[False,True])
def test_log_activations_large_incoming_gradients_preserve_native_operation_order(dtype,activation,infinite):
    values=[-100.,-20.,0.,10.,20.,21.,1000.]
    magnitude=float('inf') if infinite else 1e38 if dtype==np.float32 else 1e300
    x,nx=pair(values,dtype);v,nv=pair(np.full(len(values),magnitude),dtype)
    op=(lambda a:wasm.softplus(a)) if activation=='softplus' else (lambda a:a.logsigmoid())
    native=F.softplus if activation=='softplus' else F.logsigmoid
    with np.errstate(all='ignore'):
        actual=op(x);expected=native(nx)
        gx=wasm.autograd_grad(actual,(x,),v)[0];ng=torch.autograd.grad(expected,nx,nv)[0]
        p,j=wasm.jvp(op,(x,),(v,));np_,nj=torch.func.jvp(native,(nx,),(nv,))
        got=wasm.autograd_grad(j.sum(),(x,v));want=torch.autograd.grad(nj.sum(),(nx,nv))
    for a,b in [(actual,expected),(gx,ng),(p,np_),(j,nj),*zip(got,want)]:close(a,b,dtype)
