"""JVP determinant polynomial derivatives and strict dependency regression cases."""
import numpy as np
import pytest
from torchwasm import torch
from torchwasm._errors import BrowserUnsupported


# Closed-form cofactor matrices and their directional derivatives; no runtime
# SVD/solve rule is used to produce these expectations.
_FIXTURES = [
    ([[0.]], [[.4]], [[1.]], [[0.]], False),
    ([[1.,2.],[2.,4.]], [[.2,.3],[-.5,.7]], [[4.,-2.],[-2.,1.]], [[.7,.5],[-.3,.2]], False),
    ([[1.,2.],[2.,4.]], [[.2,.3],[-.5,.7]], [[4.,-2.],[-2.,1.]], [[.7,.5],[-.3,.2]], True),
    ([[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]], [[.2,.3,-.4],[.5,.6,.7],[-.8,.9,1.1]],
     [[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]], [[1.1,0.,.8],[0.,1.1,-.9],[.4,-.7,.8]], False),
    ([[1.,0.,0.],[0.,0.,0.],[0.,0.,0.]], [[.2,.3,-.4],[.5,.6,.7],[-.8,.9,1.1]],
     [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,1.1,-.9],[0.,-.7,.6]], False),
]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('gx,gv', [(False,False),(False,True),(True,False),(True,True)])
@pytest.mark.parametrize('graph,strict', [(False,False),(False,True),(True,False),(True,True)])
@pytest.mark.parametrize('data,vector,cofactor,curvature,squared', _FIXTURES)
def test_singular_determinant_jvp_full_polynomial(dtype,gx,gv,graph,strict,data,vector,cofactor,curvature,squared):
    x=torch.tensor(data,dtype=dtype,requires_grad=gx)
    v=torch.tensor(vector,dtype=dtype,requires_grad=gv)
    x.grad=torch.full_like(x,17.)
    v.grad=torch.full_like(v,19.)
    buffers=(x.grad,v.grad)
    hooks=[]
    handles=[item.register_hook(lambda gradient:hooks.append(True)) for item in (x,v) if item.requires_grad]
    def function(a):
        d=torch.linalg.det(a)
        return d.square() if squared else d
    output,tangent=torch.autograd.functional.jvp(function,x,v,create_graph=graph,strict=strict)
    c=np.asarray(cofactor,dtype=np.float64)
    h=np.asarray(curvature,dtype=np.float64)
    derivative=float(np.sum(c*np.asarray(vector)))
    if squared:
        h=2*derivative*c
        c=np.zeros_like(c)
        derivative=0.
    np.testing.assert_allclose(output.detach().numpy(),0.,atol=2e-6)
    np.testing.assert_allclose(tangent.detach().numpy(),derivative,rtol=1e-5,atol=2e-6)
    assert output.requires_grad is graph and tangent.requires_grad is graph
    assert hooks==[] and x.grad is buffers[0] and v.grad is buffers[1]
    np.testing.assert_array_equal(x.grad.numpy(),np.full(x.shape,17.))
    np.testing.assert_array_equal(v.grad.numpy(),np.full(v.shape,19.))
    if graph:
        if gx:
            np.testing.assert_allclose(torch.autograd.grad(output,x,retain_graph=True)[0].numpy(),c,rtol=1e-5,atol=2e-6)
            np.testing.assert_allclose(torch.autograd.grad(tangent,x,retain_graph=True)[0].numpy(),h,rtol=1e-5,atol=2e-6)
        if gv:
            np.testing.assert_allclose(torch.autograd.grad(tangent,v,retain_graph=True)[0].numpy(),c,rtol=1e-5,atol=2e-6)
    for handle in handles:handle.remove()


@pytest.mark.parametrize('shape', [(0,0),(0,2,2)])
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
@pytest.mark.parametrize('graph,strict', [(False,False),(False,True),(True,False),(True,True)])
def test_empty_determinant_jvp_graph_and_strictness(shape,dtype,graph,strict):
    x=torch.empty(shape,dtype=dtype,requires_grad=True)
    v=torch.empty_like(x,requires_grad=True)
    if strict:
        with pytest.raises(RuntimeError):
            torch.autograd.functional.jvp(torch.linalg.det,x,v,create_graph=graph,strict=True)
        return
    value,tangent=torch.autograd.functional.jvp(torch.linalg.det,x,v,create_graph=graph)
    assert value.requires_grad is graph and not tangent.requires_grad
    np.testing.assert_array_equal(tangent.numpy(),np.zeros(shape[:-2]))
    if graph:
        assert torch.autograd.grad(value.sum(),(x,v),allow_unused=True)==(None,None)


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
@pytest.mark.parametrize('graph', [False,True])
def test_strict_output_capture_and_empty_sum(dtype,graph):
    x=torch.tensor([.3,.7],dtype=dtype)
    v=torch.ones_like(x)
    capture=torch.tensor([1.,2.],dtype=dtype,requires_grad=True)
    with pytest.raises(RuntimeError):
        torch.autograd.functional.jvp(lambda a:(a.sin(),capture.square()),x,v,strict=True,create_graph=graph)
    value,tangent=torch.autograd.functional.jvp(lambda a:a*capture,x,v,strict=True,create_graph=graph)
    np.testing.assert_array_equal(tangent.detach().numpy(),[1.,2.])
    assert tangent.requires_grad is graph
    empty=torch.empty((0,),dtype=dtype)
    if graph:
        with pytest.raises(RuntimeError):
            torch.autograd.functional.jvp(lambda a:a.sum(),empty,empty,strict=True,create_graph=True)
    else:
        _,result=torch.autograd.functional.jvp(lambda a:a.sum(),empty,empty,strict=True)
        assert result.item()==0.


def test_functional_jvp_callback_hook_rejection_restores_ad_mode():
    x=torch.tensor([.3,.7],requires_grad=True)
    v=torch.ones_like(x)
    def function(a):
        a.register_hook(lambda gradient:gradient*2.)
        return a.square()
    with pytest.raises(BrowserUnsupported,match='gradient hooks inside forward AD'):
        torch.autograd.functional.jvp(function,x,v,strict=True,create_graph=True)
    with torch.no_grad():
        _,tangent=torch.func.jvp(lambda a:a.square(),(x,),(v,))
    np.testing.assert_allclose(tangent.numpy(),[.6,1.4])
