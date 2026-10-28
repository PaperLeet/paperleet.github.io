import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.diffusion_kto.experiment import run

@pytest.mark.t0
def test_binary_contract(impl):
    """Feedback must be binary signs and the output a scalar of the input dtype."""
    x=torch.zeros(2,dtype=torch.float64);y=x.new_tensor([1.,-1.])
    assert impl.kto_loss(x,x,y,x,.5).dtype==x.dtype
    with pytest.raises(ValueError):impl.kto_loss(x,x,x,x,1.)
    with pytest.raises(ValueError):impl.kto_loss(x,x,y,x,0.)

@pytest.mark.t1
def test_scalar_utility_oracle(impl):
    """Independent scalar sigmoid verifies signed utility and beta-scaled baseline."""
    c=torch.tensor([.9,-.4,.1],dtype=torch.float64);r=c.new_tensor([.1,.2,-.2]);y=c.new_tensor([1.,-1.,1.]);u=c.new_tensor([.1,.7])
    expected=-sum(1/(1+math.exp(-s*.7*(a-b-.4)))-.5 for a,b,s in zip(c.tolist(),r.tolist(),y.tolist()))/3
    close(impl.kto_loss(c,r,y,u,.7),c.new_tensor(expected))

@pytest.mark.t1
def test_negative_unrelated_mean_clamped(impl):
    """A negative empirical KL estimate is replaced by zero before utility evaluation."""
    c=torch.tensor([.3,-.8]);y=c.new_tensor([1.,-1.]);r=torch.zeros_like(c)
    close(impl.kto_loss(c,r,y,c.new_tensor([-3.,1.]),1.),impl.kto_loss(c,r,y,torch.zeros_like(c),1.))

@pytest.mark.t2
def test_analytic_gradient_and_detach(impl):
    """Only current policy log probabilities receive signed sigmoid derivatives."""
    c=torch.tensor([.4,-.2],dtype=torch.float64,requires_grad=True);r=torch.zeros_like(c,requires_grad=True);y=c.new_tensor([1.,-1.],requires_grad=True);u=c.new_tensor([.2,.4],requires_grad=True)
    grads=torch.autograd.grad(impl.kto_loss(c,r,y,u,2.),(c,r,y,u),allow_unused=True)
    z=(c.detach()-.3)*y.detach()*2;v=torch.sigmoid(z)
    close(grads[0],-y.detach()*v*(1-v));assert all(g is None or not g.any() for g in grads[1:])

@pytest.mark.t2
def test_sign_flip_and_saturation(impl):
    """Feedback sign reversal reverses centered utility; extreme finite logits stay finite."""
    c=torch.tensor([1000.,-1000.],dtype=torch.float64);y=c.new_tensor([1.,-1.]);z=torch.zeros_like(c)
    close(impl.kto_loss(c,z,y,z,2.),c.new_tensor(-.5))
    close(impl.kto_loss(c,z,-y,z,2.),c.new_tensor(.5))

@pytest.mark.t2
def test_permutation_and_batch_duplication(impl):
    """The utility is a per-example mean and unrelated batch size is independent."""
    c=torch.tensor([.2,-.9]);r=c*.2;y=c.new_tensor([1.,-1.]);u=c.new_tensor([.1,.7,.2])
    close(impl.kto_loss(c,r,y,u,.4),impl.kto_loss(c.repeat(3),r.repeat(3),y.repeat(3),u,.4))

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.kto_loss)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['final'] < -.45
    assert m['margin'] > 1.
