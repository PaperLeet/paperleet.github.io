import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close


def finite_difference(fn, value):
    expected = torch.zeros_like(value)
    for i in range(value.numel()):
        plus, minus = value.detach().clone(), value.detach().clone()
        plus.reshape(-1)[i] += 1e-6
        minus.reshape(-1)[i] -= 1e-6
        expected.reshape(-1)[i] = (fn(plus) - fn(minus)) / 2e-6
    return expected


def scalar_softmax(values):
    z = max(values)
    exps = [math.exp(x-z) for x in values]
    return [x/sum(exps) for x in exps]

from problems.multimodal.flamingo.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """The residual has text shape and malformed media IDs are rejected."""
    x=torch.zeros(3,2,dtype=torch.float64);k=torch.ones(2,2,dtype=torch.float64);tm=torch.tensor([0,1,2]);vm=torch.tensor([1,2])
    assert impl.gated_media_attention(x,k,k,tm,vm,x.new_tensor(.2)).shape==x.shape
    with pytest.raises(ValueError):impl.gated_media_attention(x,k,k,tm,vm-1,x.new_tensor(.2))

@pytest.mark.t1
def test_manual_media_attention(impl):
    """A text token sees only its most recent medium, not all previous images."""
    x=torch.tensor([[0.,0.],[1.,0.],[0.,1.]],dtype=torch.float64);k=x.new_tensor([[1.,0.],[0.,1.],[1.,1.]]);v=x.new_tensor([[2.,0.],[0.,4.],[9.,7.]])
    got=impl.gated_media_attention(x,k,v,torch.tensor([0,1,2]),torch.tensor([1,1,2]),x.new_tensor(.7))
    w=scalar_softmax([1/math.sqrt(2),0.]);expected=x.clone();expected[1]+=math.tanh(.7)*x.new_tensor([2*w[0],4*w[1]]);expected[2]+=math.tanh(.7)*v[2]
    close(got,expected,atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_zero_gate_identity_and_gate_gradient(impl):
    """A zero gate preserves pretrained text states but receives a learning signal."""
    x=torch.ones(2,2,dtype=torch.float64);v=x*2;g=torch.tensor(0.,dtype=torch.float64,requires_grad=True);ids=torch.tensor([1,1])
    got=impl.gated_media_attention(x,x,v,ids,ids,g);close(got,x)
    dg,=torch.autograd.grad(got.sum(),g);close(dg,g.new_tensor(8.))

@pytest.mark.t2
def test_no_media_has_no_visual_gradient(impl):
    """Rows before the first image neither read nor differentiate visual values."""
    x=torch.ones(2,2,dtype=torch.float64);v=torch.ones(3,2,dtype=torch.float64,requires_grad=True)
    got=impl.gated_media_attention(x,v.detach(),v,torch.tensor([0,0]),torch.tensor([1,1,2]),x.new_tensor(1.));close(got,x)
    dv,=torch.autograd.grad(got.sum(),v);close(dv,torch.zeros_like(v));assert torch.isfinite(got).all()

@pytest.mark.t2
def test_input_gradient(impl):
    """Finite differences include residual and attention-logit derivatives."""
    x=torch.tensor([[.4,1.],[1.,-.3]],dtype=torch.float64,requires_grad=True);k=x.detach().flip(0);v=k*2;ids=torch.tensor([1,1])
    fn=lambda z:impl.gated_media_attention(z,k,v,ids,ids,z.new_tensor(.8)).square().sum()
    grad,=torch.autograd.grad(fn(x),x);close(grad,finite_difference(fn,x),atol=1e-8,rtol=1e-5)

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.gated_media_attention);artifact("toy_experiment",result);m=result["metrics"]
    assert m["final"]<.01*m["initial"] and m["prefix_change"]==0.
