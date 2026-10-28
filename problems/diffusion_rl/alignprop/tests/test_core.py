import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.alignprop.experiment import run

@pytest.mark.t0
def test_schedule_contract(impl):
    """The sampler validates amplitude arrays and gradient start index."""
    x=torch.ones(2,2,dtype=torch.float64);a=x.new_tensor([.6,.8,1.]);s=x.new_tensor([.8,.6,0.])
    out=impl.reward_rollout(x,a,s,lambda state,i:state*0,0);assert out.shape==x.shape and out.dtype==x.dtype
    with pytest.raises(ValueError):impl.reward_rollout(x,a,s,lambda state,i:state,2)
    with pytest.raises(ValueError):impl.reward_rollout(x,a*0,s,lambda state,i:state,0)

@pytest.mark.t1
def test_constant_epsilon_closed_form(impl):
    """A constant epsilon telescopes to alpha_T/alpha_0*(x-sigma_0*e)+sigma_T*e."""
    x=torch.tensor([[1.,-2.]],dtype=torch.float64);a=x.new_tensor([.4,.7,.9,1.]);s=(1-a.square()).sqrt();e=x.new_tensor([[.3,-.8]])
    close(impl.reward_rollout(x,a,s,lambda state,i:e,0),(x-s[0]*e)/a[0])

@pytest.mark.t1
def test_truncation_forward_invariance(impl):
    """Changing only the gradient boundary never changes a forward trajectory."""
    x=torch.tensor([[.7]],dtype=torch.float64);a=x.new_tensor([.4,.7,.9,1.]);s=(1-a.square()).sqrt()
    values=[impl.reward_rollout(x,a,s,lambda state,i:.2*state+.1*i,k) for k in range(3)]
    for v in values[1:]:close(v,values[0])

@pytest.mark.t2
def test_independent_suffix_derivative(impl):
    """Scalar affine recurrences independently compute gradients of suffix parameters."""
    x=torch.tensor([[.7]],dtype=torch.float64,requires_grad=True);a=x.new_tensor([.4,.7,.9,1.],requires_grad=True);s=x.new_tensor([.9,.6,.3,0.],requires_grad=True);w=x.new_tensor([.2,.3,.4],requires_grad=True)
    out=impl.reward_rollout(x,a,s,lambda state,i:w[i]*state,1)
    gx,ga,gs,gw=torch.autograd.grad(out.sum(),(x,a,s,w),allow_unused=True)
    aa=a.detach().tolist();ss=s.detach().tolist();ww=w.detach().tolist();f=[aa[i+1]/aa[i]+(ss[i+1]-aa[i+1]*ss[i]/aa[i])*ww[i] for i in range(3)]
    coeff=[ss[i+1]-aa[i+1]*ss[i]/aa[i] for i in range(3)]
    close(gw,w.new_tensor([0.,.7*f[0]*coeff[1]*f[2],.7*f[0]*f[1]*coeff[2]]))
    assert all(g is None or not g.any() for g in (gx,ga,gs))

@pytest.mark.t2
def test_full_chain_finite_difference(impl):
    """At boundary zero all model paths match independent central differences."""
    x=torch.tensor([[.2,-.4]],dtype=torch.float64);a=x.new_tensor([.5,.8,1.]);s=(1-a.square()).sqrt();w=x.new_tensor(.2,requires_grad=True)
    def evaluate(v):return impl.reward_rollout(x,a,s,lambda state,i:v*state+.1,0).square().sum()
    g,=torch.autograd.grad(evaluate(w),w);h=1e-6
    close(g,(evaluate(w.detach()+h)-evaluate(w.detach()-h))/(2*h),atol=1e-8,rtol=1e-6)

@pytest.mark.t2
def test_only_final_transition_gradient(impl):
    """A one-step suffix keeps only the last model parameter live."""
    x=torch.ones(1,1,dtype=torch.float64);a=x.new_tensor([.5,.8,1.]);s=(1-a.square()).sqrt();w=x.new_tensor([.1,.2],requires_grad=True)
    g,=torch.autograd.grad(impl.reward_rollout(x,a,s,lambda state,i:w[i].expand_as(state),1).sum(),w)
    close(g,w.new_tensor([0.,-s[1].item()/a[1].item()]))

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.reward_rollout)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['final'] < .01*m['initial']
    assert m['updates']==45
