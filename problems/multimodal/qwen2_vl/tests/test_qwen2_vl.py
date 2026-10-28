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

from problems.multimodal.qwen2_vl.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Three axes split a positive multiple of six feature dimensions."""
    x=torch.ones(2,12,dtype=torch.float64);p=torch.zeros(3,2,dtype=torch.float64);f=torch.ones(6,dtype=torch.float64)
    assert impl.chunked_mrope(x,p,f).shape==x.shape
    with pytest.raises(ValueError):impl.chunked_mrope(x[:,:10],p,f)

@pytest.mark.t1
def test_scalar_rotation_oracle(impl):
    """Scalar rotations fix axis allocation, frequency indices, pairing and rotation signs."""
    x=torch.arange(1,25,dtype=torch.float64).reshape(2,12)/7;p=x.new_tensor([[.2,.4],[1.,1.3],[2.,2.7]]);f=x.new_tensor([1.,.7,.4,.2,.1,.03]);expected=[]
    for n in range(2):
        first=[];second=[]
        for r in range(6):
            half=6;axis=r // (half//3);angle=p[axis,n].item()*f[r].item();a=x[n,r].item();b=x[n,r+6].item()
            first.append(a*math.cos(angle)-b*math.sin(angle));second.append(b*math.cos(angle)+a*math.sin(angle))
        expected.append(first+second)
    close(impl.chunked_mrope(x,p,f),x.new_tensor(expected),atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_position_gradient(impl):
    """Position gradients match scalar finite differences through sine and cosine."""
    x=torch.arange(1,13,dtype=torch.float64).reshape(1,12)/5;p=torch.tensor([[.3],[1.2],[2.1]],dtype=torch.float64,requires_grad=True);f=x.new_tensor([1.,.7,.4,.2,.1,.03])
    fn=lambda z:impl.chunked_mrope(x,z,f).sum();g,=torch.autograd.grad(fn(p),p)
    close(g,finite_difference(fn,p),atol=1e-8,rtol=1e-5)

@pytest.mark.t2
def test_norm_and_inverse(impl):
    """Independent rotations preserve vector norms and are inverted by negative positions."""
    x=torch.arange(1,25,dtype=torch.float64).reshape(2,12)/4;p=x.new_tensor([[.2,.5],[1.,2.],[3.,4.]]);f=x.new_tensor([1.,.7,.4,.2,.1,.03]);out=impl.chunked_mrope(x,p,f)
    close(out.square().sum(-1),x.square().sum(-1),atol=1e-10,rtol=1e-12)
    close(impl.chunked_mrope(out,-p,f),x,atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_zero_and_common_shift(impl):
    """Zero position is identity; a shared shift preserves query-key dot products."""
    x=torch.arange(1,13,dtype=torch.float64).reshape(1,12);y=x.flip(-1);f=x.new_tensor([1.,.7,.4,.2,.1,.03]);p=x.new_tensor([[.3],[1.],[2.]]);shift=x.new_tensor([[2.],[.4],[1.]])
    close(impl.chunked_mrope(x,p*0,f),x)
    a=(impl.chunked_mrope(x,p,f)*impl.chunked_mrope(y,p*2,f)).sum();b=(impl.chunked_mrope(x,p+shift,f)*impl.chunked_mrope(y,p*2+shift,f)).sum();close(a,b,atol=1e-10,rtol=1e-12)

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.chunked_mrope);artifact("toy_experiment",result);m=result["metrics"]
    assert m["retrieval_accuracy"]==1. and m["norm_error"]<1e-10
