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

from problems.multimodal.llava.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Replacing one placeholder with P patches changes length by P-1."""
    t=torch.zeros(4,3,dtype=torch.float64);v=torch.ones(2,3,dtype=torch.float64);labels=torch.tensor([1,2,3,4])
    a,b=impl.pack_visual_tokens(t,v,labels,1);assert a.shape==(5,3) and b.shape==(5,) and a.dtype==t.dtype
    with pytest.raises(ValueError):impl.pack_visual_tokens(t,v,labels,4)

@pytest.mark.t1
def test_hand_packed_sequence(impl):
    """Image tokens replace the sentinel and are excluded from language targets."""
    t=torch.tensor([[1.,2.],[99.,99.],[3.,4.]],dtype=torch.float64);v=t.new_tensor([[5.,6.],[7.,8.]])
    a,b=impl.pack_visual_tokens(t,v,torch.tensor([-100,42,9]),1)
    close(a,t.new_tensor([[1.,2.],[5.,6.],[7.,8.],[3.,4.]]));close(b,torch.tensor([-100,-100,-100,9]))

@pytest.mark.t2
def test_gradient_routing(impl):
    """The deleted placeholder is absent from the graph, while each retained token receives its aligned gradient."""
    t=torch.ones(3,2,dtype=torch.float64,requires_grad=True);v=torch.ones(2,2,dtype=torch.float64,requires_grad=True)
    out,_=impl.pack_visual_tokens(t,v,torch.tensor([1,2,3]),1);weights=torch.arange(8,dtype=torch.float64).reshape(4,2)
    gt,gv=torch.autograd.grad((out*weights).sum(),(t,v))
    close(gt,torch.cat([weights[:1],torch.zeros_like(t[:1]),weights[3:]],0));close(gv,weights[1:3])

@pytest.mark.t2
def test_first_last_and_single(impl):
    """Boundary sentinels and a one-token sequence do not leave phantom tokens."""
    t=torch.tensor([[1.],[2.]],dtype=torch.float64);v=t.new_tensor([[8.],[9.]]);l=torch.tensor([5,6])
    a,b=impl.pack_visual_tokens(t,v,l,0);close(a,t.new_tensor([[8.],[9.],[2.]]));close(b,torch.tensor([-100,-100,6]))
    a,b=impl.pack_visual_tokens(t,v,l,1);close(a,t.new_tensor([[1.],[8.],[9.]]))
    a,b=impl.pack_visual_tokens(t[:1],v,l[:1],0);close(a,v);close(b,torch.tensor([-100,-100]))

@pytest.mark.t2
def test_input_immutability(impl):
    """Packing preserves caller-owned features and labels."""
    t=torch.arange(6,dtype=torch.float64).reshape(3,2);v=t[:2].clone();l=torch.tensor([1,2,3]);saved=[x.clone() for x in (t,v,l)]
    impl.pack_visual_tokens(t,v,l,1)
    for a,b in zip((t,v,l),saved):close(a,b)

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.pack_visual_tokens);artifact("toy_experiment",result);m=result["metrics"]
    assert m["final"]<.001*m["initial"] and m["supervised_tokens"]==1
