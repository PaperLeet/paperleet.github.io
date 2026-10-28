import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.ddpo.experiment import run

@pytest.mark.t0
def test_density_and_loss_shapes(impl):
    """Gaussian coordinates reduce to (B,T), then PPO reduces to a scalar."""
    m=torch.zeros(2,3,4,dtype=torch.float64);s=m.new_tensor([.5,1.,2.]);l=impl.gaussian_log_prob(m,m,s)
    assert l.shape==(2,3) and l.dtype==m.dtype
    assert impl.ddpo_loss(l,l,m.new_tensor([1.,-1.]),.2).shape==()
    with pytest.raises(ValueError):impl.gaussian_log_prob(m,m,-s)
    with pytest.raises(ValueError):impl.ddpo_loss(l,l,m.new_tensor([1.]),.2)

@pytest.mark.t1
def test_scalar_gaussian_oracle(impl):
    """Joint density is an independently evaluated sum of coordinate normal densities."""
    m=torch.tensor([[[.2,-.5],[.3,.8]]],dtype=torch.float64);a=m.new_tensor([[[1.,.1],[-.2,.4]]]);s=m.new_tensor([.4,1.3])
    expected=[sum(-.5*((v-u)/scale)**2-math.log(scale)-.5*math.log(2*math.pi) for u,v in zip(m[0,t].tolist(),a[0,t].tolist())) for t,scale in enumerate(s.tolist())]
    close(impl.gaussian_log_prob(m,a,s),m.new_tensor([expected]))

@pytest.mark.t1
def test_both_clipping_signs(impl):
    """Positive and negative terminal advantages choose opposite clipped tails."""
    ratio=torch.tensor([[1.6,.5],[1.6,.5]],dtype=torch.float64);old=torch.zeros_like(ratio);adv=ratio.new_tensor([2.,-3.])
    expected=-(2.4+1.-4.8-2.4)/4
    close(impl.ddpo_loss(ratio.log(),old,adv,.2),ratio.new_tensor(expected))

@pytest.mark.t2
def test_gaussian_score_and_frozen_sample(impl):
    """The exact score derivative reaches means, not sampled actions or scales."""
    m=torch.randn(2,2,3,dtype=torch.float64,requires_grad=True);a=torch.randn_like(m,requires_grad=True);s=m.new_tensor([.4,1.2],requires_grad=True)
    gs=torch.autograd.grad(impl.gaussian_log_prob(m,a,s).sum(),(m,a,s),allow_unused=True)
    close(gs[0],(a.detach()-m.detach())/s.detach()[None,:,None].square());assert all(g is None or not g.any() for g in gs[1:])

@pytest.mark.t2
def test_clipped_zero_gradient_and_freeze(impl):
    """Saturated favorable updates stop; old policy and rewards remain frozen."""
    c=torch.tensor([[math.log(2.)],[math.log(.5)]],dtype=torch.float64,requires_grad=True);o=torch.zeros_like(c,requires_grad=True);a=c.new_tensor([1.,-1.],requires_grad=True)
    gs=torch.autograd.grad(impl.ddpo_loss(c,o,a,.2),(c,o,a),allow_unused=True)
    close(gs[0],torch.zeros_like(c));assert all(g is None or not g.any() for g in gs[1:])

@pytest.mark.t2
def test_time_mean_and_log_domain(impl):
    """Repeating sampled transitions preserves the objective; dangerous ratios are rejected."""
    c=torch.tensor([[.2,-.1],[-.3,.1]]);o=torch.zeros_like(c);a=c.new_tensor([1.,-2.])
    close(impl.ddpo_loss(c,o,a,.2),impl.ddpo_loss(c.repeat(1,3),o.repeat(1,3),a,.2))
    with pytest.raises(ValueError):impl.ddpo_loss(c+100,o,a,.2)

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.gaussian_log_prob, impl.ddpo_loss)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['final_error'] < .05*m['initial_error']
    assert m['updates']==60
