import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.diffmimic.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'Replay, PD gains and temporal loss each validate their own state/control contract.'
    x=torch.zeros(2,3,dtype=torch.float64);y=impl.replay_state(x,x,1.);assert y.shape==x.shape
    with pytest.raises(ValueError):impl.replay_state(x,x,-1.)
    with pytest.raises(ValueError):impl.pd_torque(x,x,x,-1.,1.)
    with pytest.raises(ValueError):impl.state_matching_loss(x,x)

@pytest.mark.t1
def test_replay_boundary_pd_and_temporal_sum(impl):
    'The strict replay boundary, velocity damping and summed temporal loss are independently calculable.'
    x=torch.tensor([[.5,0.],[1.,0.],[2.,0.]],dtype=torch.float64);r=torch.zeros_like(x)
    close(impl.replay_state(x,r,1.),x.new_tensor([[.5,0.],[0.,0.],[0.,0.]]))
    close(impl.pd_torque(x,x*.5,r,2.,.4),-2.2*x)
    states=torch.tensor([[[1.,2.],[3.,4.]],[[0.,1.],[0.,2.]]],dtype=x.dtype)
    close(impl.state_matching_loss(states,torch.zeros_like(states)),x.new_tensor(17.5))

@pytest.mark.t2
def test_replay_cut_and_loss_gradient(impl):
    'Replay cuts old-state gradients, while PD and retained trajectories remain differentiable.'
    x=torch.tensor([[.5],[2.]],dtype=torch.float64,requires_grad=True);r=torch.zeros_like(x,requires_grad=True)
    y=impl.replay_state(x,r,1.);gx,gr=torch.autograd.grad(y.sum(),(x,r),allow_unused=True);close(gx,x.new_tensor([[1.],[0.]]));assert gr is None
    states=x.reshape(1,2,1);loss=impl.state_matching_loss(states,torch.zeros_like(states));g,=torch.autograd.grad(loss,x);close(g,2*x.detach())
    q=torch.ones(1,2,dtype=x.dtype,requires_grad=True);v=torch.ones_like(q,requires_grad=True);target=torch.ones_like(q,requires_grad=True);gq,gv,gt=torch.autograd.grad(impl.pd_torque(q,v,target,3.,.7).sum(),(q,v,target));close(gq,q.new_full(q.shape,-3.));close(gv,q.new_full(q.shape,-.7));close(gt,q.new_full(q.shape,3.))

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.replay_state,impl.pd_torque,impl.state_matching_loss)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['final_loss']<.9*m['initial_loss']
    assert m['replay_probe']==0.
    assert m['gain']<0.
