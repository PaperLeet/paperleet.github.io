import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.dpok.experiment import run

@pytest.mark.t0
def test_shapes_and_parameters(impl):
    """Fixed-covariance Gaussian parameters and surrogate shapes are explicit."""
    c=torch.zeros(2,3,4,dtype=torch.float64);s=c.new_tensor([1.,2.,3.]);k=impl.gaussian_kl(c,c,s)
    assert k.shape==(2,3) and k.dtype==c.dtype
    with pytest.raises(ValueError):impl.gaussian_kl(c,c,s*0)
    with pytest.raises(ValueError):impl.dpok_surrogate(k,c.new_tensor([1.,2.]),k,1.,-1.)

@pytest.mark.t1
def test_scalar_kl(impl):
    """Independent coordinate sums verify one-half and variance scaling."""
    c=torch.tensor([[[1.,2.],[3.,-1.]]],dtype=torch.float64);r=torch.zeros_like(c);s=c.new_tensor([2.,.5])
    close(impl.gaussian_kl(c,r,s),c.new_tensor([[5/8,20.]]))

@pytest.mark.t1
def test_surrogate_sum_time(impl):
    """The trajectory score and KL bound sum time before averaging trajectories."""
    p=torch.tensor([[-1.,-2.],[-3.,-1.]],dtype=torch.float64);r=p.new_tensor([2.,-1.]);k=p.new_tensor([[.2,.3],[.1,.4]])
    close(impl.dpok_surrogate(p,r,k,2.,.4),p.new_tensor(2.2))

@pytest.mark.t2
def test_mean_gradient_and_frozen_reference(impl):
    """Gaussian KL differentiates only the current policy mean."""
    c=torch.ones(2,2,3,dtype=torch.float64,requires_grad=True);r=torch.zeros_like(c,requires_grad=True);s=c.new_tensor([.5,2.],requires_grad=True)
    gs=torch.autograd.grad(impl.gaussian_kl(c,r,s).sum(),(c,r,s),allow_unused=True)
    close(gs[0],torch.ones_like(c)/s.detach()[None,:,None].square());assert all(g is None or not g.any() for g in gs[1:])

@pytest.mark.t2
def test_surrogate_gradient_ownership(impl):
    """Rewards freeze while both logp and KL retain their analytic coefficients."""
    p=torch.zeros(2,3,dtype=torch.float64,requires_grad=True);r=p.new_tensor([2.,-1.],requires_grad=True);k=torch.ones_like(p,requires_grad=True)
    gp,gr,gk=torch.autograd.grad(impl.dpok_surrogate(p,r,k,.7,.2),(p,r,k),allow_unused=True)
    close(gp,(-.7*r.detach()/2)[:,None].expand(2,3));close(gk,p.new_full((2,3),.1));assert gr is None or not gr.any()

@pytest.mark.t2
def test_zero_kl_and_time_replication(impl):
    """Identical means give zero KL; repeating time multiplies a trajectory objective."""
    c=torch.randn(2,3,4);s=torch.ones(3);close(impl.gaussian_kl(c,c,s),torch.zeros(2,3))
    p=torch.randn(2,3);k=p.square();r=torch.tensor([1.,-2.]);loss=impl.dpok_surrogate(p,r,k,1.,.2)
    close(impl.dpok_surrogate(p.repeat(1,2),r,k.repeat(1,2),1.,.2),2*loss)

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.gaussian_kl, impl.dpok_surrogate)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['final_error'] < .03*m['initial_error']
    assert .5<m['mean']<1.1
