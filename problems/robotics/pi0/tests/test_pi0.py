import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.pi0.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'The official noise-time path accepts a per-example time in the closed unit interval.'
    a=torch.ones(2,2,1,dtype=torch.float64);t=a.new_tensor([0.,1.]);x,u=impl.flow_interpolate(a,-a,t);assert x.shape==a.shape and u.dtype==a.dtype
    with pytest.raises(ValueError):impl.flow_interpolate(a,-a,t+1)
    with pytest.raises(ValueError):impl.euler_step(a,a,.1)

@pytest.mark.t1
def test_time_endpoints_and_sampling_sign(impl):
    'A constant target velocity must integrate from noise at one to actions at zero.'
    a=torch.tensor([[[2.]],[[4.]],[[6.]]],dtype=torch.float64);n=-a;t=a.new_tensor([0.,.25,1.]);x,u=impl.flow_interpolate(a,n,t)
    close(x,a.new_tensor([[[2.]],[[2.]], [[-6.]]]));close(u,-2*a)
    s=n
    for _ in range(5):s=impl.euler_step(s,u,-.2)
    close(s,a)

@pytest.mark.t2
def test_path_gradient_and_fixed_labels(impl):
    'Interpolation is differentiable while supervised labels are fixed data.'
    a=torch.ones(1,2,1,dtype=torch.float64,requires_grad=True);n=torch.zeros_like(a,requires_grad=True);t=a.new_tensor([.3]);x,u=impl.flow_interpolate(a,n,t)
    ga,gn=torch.autograd.grad(x.sum(),(a,n));close(ga,torch.full_like(a,.7));close(gn,torch.full_like(n,.3));assert not u.requires_grad
    v=torch.ones_like(a,requires_grad=True);s=impl.euler_step(a,v,-.2);gv,=torch.autograd.grad(s.sum(),v);close(gv,torch.full_like(v,-.2));close(impl.euler_step(a,v,0.),a)

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.flow_interpolate,impl.euler_step)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['final_loss'] < 1e-10*m['initial_loss']
    assert m['sample_mse']<1e-10
