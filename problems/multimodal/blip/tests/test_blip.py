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

from problems.multimodal.blip.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Rectangular queues are supported, while positive indices must be valid."""
    x=torch.zeros(2,4,dtype=torch.float64);p=torch.tensor([3,1]);old=x.clone()
    assert impl.momentum_target_loss(x,x,p,.3).shape==();close(x,old)
    with pytest.raises(ValueError): impl.momentum_target_loss(x,x,p+4,.3)
    with pytest.raises(ValueError): impl.momentum_target_loss(x,x,p,1.1)

@pytest.mark.t1
def test_manual_soft_targets(impl):
    """A scalar mixture of teacher probabilities and arbitrary positive columns fixes alpha's direction."""
    x=torch.tensor([[1.,2.,-.5],[.4,-.7,1.]],dtype=torch.float64);t=x.new_tensor([[.2,1.,-.1],[1.,0.,2.]]);p=torch.tensor([0,2])
    expected=0.
    for i in range(2):
        a=scalar_softmax(x[i].tolist());b=scalar_softmax(t[i].tolist())
        expected-=sum((.3*b[j]+.7*int(j==p[i].item()))*math.log(a[j]) for j in range(3))/2
    close(impl.momentum_target_loss(x,t,p,.3),x.new_tensor(expected),atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_teacher_stop_gradient(impl):
    """The momentum teacher produces targets only; the student learns softmax-target."""
    x=torch.tensor([[.2,1.,-.3]],dtype=torch.float64,requires_grad=True);t=torch.tensor([[1.,-.2,.5]],dtype=torch.float64,requires_grad=True);p=torch.tensor([2])
    g,tg=torch.autograd.grad(impl.momentum_target_loss(x,t,p,.4),(x,t),allow_unused=True)
    assert tg is None
    target=.4*torch.softmax(t.detach(),-1)+x.new_tensor([[0.,0.,.6]])
    close(g,torch.softmax(x.detach(),-1)-target,atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_alpha_endpoints(impl):
    """Alpha zero ignores the teacher; alpha one ignores hard positive labels."""
    x=torch.tensor([[1.,0.,-1.]],dtype=torch.float64);t=x.flip(1);p=torch.tensor([0])
    close(impl.momentum_target_loss(x,t,p,0.),impl.momentum_target_loss(x,t*7,p,0.))
    close(impl.momentum_target_loss(x,t,p,1.),impl.momentum_target_loss(x,t,torch.tensor([2]),1.))

@pytest.mark.t2
def test_offset_and_batch_duplication(impl):
    """Softmax is invariant to row offsets and the batch mean is invariant to duplication."""
    x=torch.tensor([[1.,2.,3.]],dtype=torch.float64);p=torch.tensor([1]);t=-x
    base=impl.momentum_target_loss(x,t,p,.25)
    close(base,impl.momentum_target_loss(x+1000,t-300,p,.25))
    close(base,impl.momentum_target_loss(x.repeat(2,1),t.repeat(2,1),p.repeat(2),.25))

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.momentum_target_loss);artifact("toy_experiment",result);m=result["metrics"]
    assert m["final"]<.15*m["initial"] and m["accuracy"]==1.
