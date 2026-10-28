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

from problems.multimodal.chameleon.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Affine tensors match heads and features; epsilon is positive."""
    x=torch.ones(3,2,4,dtype=torch.float64);w=torch.ones(2,4,dtype=torch.float64);b=w*0
    assert impl.qk_layer_norm(x,w,b,1e-5).shape==x.shape
    with pytest.raises(ValueError):impl.qk_layer_norm(x,w[:1],b,1e-5)
    with pytest.raises(ValueError):impl.qk_layer_norm(x,w,b,0.)

@pytest.mark.t1
def test_scalar_head_oracle(impl):
    """Each head gets its own population statistics and affine transformation."""
    x=torch.tensor([[[1.,2.,4.],[8.,-2.,3.]],[[2.,5.,0.],[1.,3.,9.]]],dtype=torch.float64);w=x.new_tensor([[1.,2.,.5],[.3,1.,2.]]);b=x.new_tensor([[.2,0.,-.4],[1.,2.,3.]])
    expected=[]
    for token in x.tolist():
        row=[]
        for h,values in enumerate(token):
            mean=sum(values)/3;var=sum((z-mean)**2 for z in values)/3
            row.append([(z-mean)/math.sqrt(var+.02)*w[h,d].item()+b[h,d].item() for d,z in enumerate(values)])
        expected.append(row)
    close(impl.qk_layer_norm(x,w,b,.02),x.new_tensor(expected),atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_gradient(impl):
    """Central differences validate the mean and variance contribution to gradients."""
    x=torch.tensor([[[.3,1.,-2.]]],dtype=torch.float64,requires_grad=True);w=x.new_tensor([[.7,1.2,.4]]);b=x.new_tensor([[.1,.3,-.2]])
    fn=lambda z:impl.qk_layer_norm(z,w,b,.01).square().sum();g,=torch.autograd.grad(fn(x),x)
    close(g,finite_difference(fn,x),atol=1e-8,rtol=1e-5)

@pytest.mark.t2
def test_head_shift_invariance(impl):
    """Adding a separate constant to every head leaves normalized values unchanged."""
    x=torch.arange(12,dtype=torch.float64).reshape(2,2,3);w=torch.ones(2,3,dtype=torch.float64);b=w*0
    close(impl.qk_layer_norm(x,w,b,.01),impl.qk_layer_norm(x+x.new_tensor([[[100.],[-40.]]]),w,b,.01),atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_constant_and_single_feature(impl):
    """Constant heads normalize to zero before bias, including D=1."""
    x=torch.ones(2,2,1,dtype=torch.float64);w=x.new_tensor([[2.],[3.]]);b=x.new_tensor([[.4],[-.7]])
    close(impl.qk_layer_norm(x,w,b,1e-5),b[None].repeat(2,1,1));assert torch.isfinite(impl.qk_layer_norm(x,w,b,1e-5)).all()

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.qk_layer_norm);artifact("toy_experiment",result);m=result["metrics"]
    assert m["entropy_spread"]<1e-3 and m["normalized_entropy"]>m["raw_entropy"]+.2
