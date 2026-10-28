import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.octo.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'Observation-window and action-coordinate padding have distinct declared shapes.'
    x=torch.ones(2,2,3,4,dtype=torch.float64);t=torch.ones(2,2,dtype=torch.bool);a=torch.ones_like(x,dtype=torch.bool)
    y=impl.masked_action_loss(x,x,t,a);assert y.shape==() and y.dtype==x.dtype
    with pytest.raises(ValueError):impl.masked_action_loss(x,x,t[:1],a)

@pytest.mark.t1
def test_intersect_and_normalize(impl):
    'The official masked reduction intersects both masks and divides by the valid fraction.'
    x=torch.tensor([[[[1.,2.],[3.,4.]],[[100.,200.],[300.,400.]]]],dtype=torch.float64)
    a=torch.tensor([[[[True,False],[True,False]],[[True,True],[True,True]]]])
    close(impl.masked_action_loss(x,torch.zeros_like(x),torch.tensor([[True,False]]),a),x.new_tensor(5.))

@pytest.mark.t2
def test_gradients_detached_target_and_empty_mask(impl):
    'Fixed targets, padded-coordinate gradients and all-padding zero follow the loss contract.'
    x=torch.tensor([[[[2.,3.]]]],dtype=torch.float64,requires_grad=True);target=torch.zeros_like(x,requires_grad=True);t=torch.tensor([[True]]);a=torch.tensor([[[[True,False]]]])
    loss=impl.masked_action_loss(x,target,t,a);gx,gt=torch.autograd.grad(loss,(x,target),allow_unused=True)
    close(gx,x.new_tensor([[[[4.,0.]]]]));assert gt is None
    zero=impl.masked_action_loss(x,target,t&False,a);g,=torch.autograd.grad(zero,x);close(zero,x.new_tensor(0.));close(g,torch.zeros_like(x))
    close(impl.masked_action_loss(x.repeat(2,1,1,1),target.repeat(2,1,1,1),t.repeat(2,1),a.repeat(2,1,1,1)),loss)

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.masked_action_loss)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['final_loss'] < 1e-6*m['initial_loss']
    assert m['padded_motion']==0.
