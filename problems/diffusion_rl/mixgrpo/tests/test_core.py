import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.mixgrpo.experiment import run

@pytest.mark.t0
def test_window_parameters(impl):
    """The scheduler rejects invalid widths and supports full-trajectory windows."""
    assert impl.window_start(100,4,4,3,2)==0
    for args in ((0,4,5,1,1),(-1,4,2,1,1),(0,4,2,0,1)):
        with pytest.raises(ValueError):impl.window_start(*args)

@pytest.mark.t1
def test_saturating_schedule(impl):
    """Algorithm 1 advances every interval then saturates at T-width."""
    assert [impl.window_start(i,7,3,2,2) for i in range(8)]==[0,0,2,2,4,4,4,4]

@pytest.mark.t1
def test_independent_mixed_moments(impl):
    """ODE and SDE modes differ by the explicit score correction and variance."""
    x=torch.tensor([[1.,-2.]],dtype=torch.float64);v=x.new_tensor([[.2,.3]])
    m,s=impl.mixed_transition(x,v,.6,.4,.5,False);close(m,x-.2*v);assert s.item()==0
    m,s=impl.mixed_transition(x,v,.6,.4,.5,True);q=.25*.6/.4
    close(m,x-.2*(v+q/1.2*(x+.4*v)));close(s,x.new_tensor(math.sqrt(q*.2)))

@pytest.mark.t2
def test_window_only_gradient(impl):
    """No policy gradient reaches deterministic steps outside the selected window."""
    l=torch.zeros(2,5,dtype=torch.float64,requires_grad=True);a=l.new_tensor([1.,-2.])
    g,=torch.autograd.grad(impl.window_loss(l,a,1,2,.2),l);expected=torch.zeros_like(l);expected[:,1:3]=-a[:,None]/4
    close(g,expected)

@pytest.mark.t2
def test_selected_window_mean(impl):
    """Loss reduction counts width rather than the length of the complete rollout."""
    l=torch.tensor([[5.,.1,-.1,5.],[4.,-.2,.2,4.]],dtype=torch.float64);a=l.new_tensor([1.,-1.])
    close(impl.window_loss(l,a,1,2,.2),impl.window_loss(l[:,1:3],a,0,2,.2))

@pytest.mark.t2
def test_velocity_gradient_and_zero_noise(impl):
    """A zero stochastic scale reduces exactly to ODE and sampled states stay frozen."""
    x=torch.ones(2,2,dtype=torch.float64,requires_grad=True);v=torch.ones_like(x,requires_grad=True)
    m,s=impl.mixed_transition(x,v,.8,.6,0.,True);gx,gv=torch.autograd.grad(m.sum(),(x,v),allow_unused=True)
    assert gx is None or not gx.any();close(gv,torch.full_like(v,-.2));assert s.item()==0

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.window_start, impl.mixed_transition, impl.window_loss)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['late_error'] < .15*m['early_error']
    assert m['last_start']==2
