import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.flow_dppo.experiment import run

@pytest.mark.t0
def test_contract_and_invalid_divergence(impl):
    """Gaussian scales are positive and divergence masks require nonnegative distances."""
    c=torch.ones(2,2,3,dtype=torch.float64);k=impl.policy_kl(c,c,c.new_tensor([1.,2.]));assert k.shape==(2,2)
    with pytest.raises(ValueError):impl.policy_kl(c,c,c.new_tensor([0.,1.]))
    with pytest.raises(ValueError):impl.flow_dppo_loss(k,c.new_tensor([1.,-1.]),k-1,k,.2,0.)

@pytest.mark.t1
def test_gaussian_kl_oracle(impl):
    """The exact distribution divergence sums feature displacements over twice the variance."""
    c=torch.tensor([[[1.,2.],[2.,-1.]]],dtype=torch.float64);o=torch.zeros_like(c);s=c.new_tensor([1.,2.])
    close(impl.policy_kl(c,o,s),c.new_tensor([[2.5,.625]]))

@pytest.mark.t1
def test_directional_truth_table(impl):
    """Above threshold only outward positive/increasing and negative/decreasing updates stop."""
    ratio=torch.tensor([[2.,.5],[2.,.5]],dtype=torch.float64);a=ratio.new_tensor([1.,-1.]);kl=torch.ones_like(ratio);z=torch.zeros_like(ratio)
    close(impl.flow_dppo_loss(ratio.log(),a,kl,z,.1,0.),ratio.new_tensor(.375))

@pytest.mark.t2
def test_strict_boundary_and_unit_ratio(impl):
    """D==delta and ratio==1 are both retained, even with signed advantages."""
    ratio=torch.tensor([[2.,1.],[.5,1.]],dtype=torch.float64);a=ratio.new_tensor([2.,-1.]);kl=ratio.new_tensor([[.2,9.],[.2,9.]])
    close(impl.flow_dppo_loss(ratio.log(),a,kl,torch.zeros_like(kl),.2,0.),ratio.new_tensor(-1.125))

@pytest.mark.t2
def test_blocked_ratio_stability(impl):
    """Blocked exponentials are removed before exp and reference KL remains active."""
    l=torch.tensor([[1000.],[-1000.]],dtype=torch.float64,requires_grad=True);a=l.new_tensor([1.,-1.]);k=torch.ones_like(l);r=torch.ones_like(l,requires_grad=True)
    loss=impl.flow_dppo_loss(l,a,k,r,.1,.4);gl,gr=torch.autograd.grad(loss,(l,r))
    close(loss,l.new_tensor(.4));close(gl,torch.zeros_like(l));close(gr,torch.full_like(r,.2))

@pytest.mark.t2
def test_corrective_gradients_and_detach(impl):
    """Corrective policy gradients survive high divergence; only reference KL gets KL gradients."""
    l=torch.tensor([[-.2],[.3]],dtype=torch.float64,requires_grad=True);a=l.new_tensor([1.,-1.],requires_grad=True);k=torch.ones_like(l,requires_grad=True);r=torch.ones_like(l,requires_grad=True)
    gl,ga,gk,gr=torch.autograd.grad(impl.flow_dppo_loss(l,a,k,r,.1,.2),(l,a,k,r),allow_unused=True)
    close(gl,-l.detach().exp()*a.detach()[:,None]/2);close(gr,torch.full_like(r,.1));assert all(g is None or not g.any() for g in (ga,gk))
    current=torch.tensor([[[.4,-.2],[.8,.3]]],dtype=torch.float64,requires_grad=True)
    old=torch.zeros_like(current,requires_grad=True)
    std=current.new_tensor([.5,2.],requires_grad=True)
    gc,go,gs=torch.autograd.grad(impl.policy_kl(current,old,std).sum(),(current,old,std),allow_unused=True)
    close(gc,current.detach()/std.detach()[None,:,None].square())
    assert all(g is None or not g.any() for g in (go,gs))

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.policy_kl, impl.flow_dppo_loss)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['final_error'] < .05*m['initial_error']
    assert m['updates']==65
