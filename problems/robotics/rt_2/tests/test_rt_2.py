import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.rt_2.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'Task and vocabulary masks must align and allow at least one action token.'
    x=torch.zeros(2,3,dtype=torch.float64);a=torch.tensor([True,False,True]);r=torch.tensor([True,False])
    y=impl.action_probabilities(x,a,r);assert y.shape==x.shape and y.dtype==x.dtype
    with pytest.raises(ValueError):impl.action_probabilities(x,a&False,r)
    with pytest.raises(ValueError):impl.action_probabilities(x,a,r[:1])

@pytest.mark.t1
def test_mixed_task_probabilities(impl):
    'Section 3.2 restricts only robot rows and renormalizes the allowed vocabulary.'
    x=torch.tensor([[0.,math.log(100),math.log(3)],[0.,math.log(100),math.log(3)]],dtype=torch.float64)
    y=impl.action_probabilities(x,torch.tensor([True,False,True]),torch.tensor([True,False]))
    close(y,x.new_tensor([[.25,0.,.75],[1/104,100/104,3/104]]))

@pytest.mark.t2
def test_stability_and_forbidden_gradients(impl):
    'Forbidden logits have zero influence, including under extreme score magnitudes.'
    x=torch.tensor([[1000.,2000.,1000.]],dtype=torch.float64,requires_grad=True)
    y=impl.action_probabilities(x,torch.tensor([True,False,True]),torch.tensor([True]));close(y,x.new_tensor([[.5,0.,.5]]))
    g,=torch.autograd.grad(y[0,0],x);close(g,x.new_tensor([[.25,0.,-.25]]))
    one=impl.action_probabilities(x,torch.tensor([False,True,False]),torch.tensor([True]));close(one,x.new_tensor([[0.,1.,0.]]))

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.action_probabilities)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['final_loss'] < .04*m['initial_loss']
    assert m['forbidden_robot_mass']==0.
    assert m['vlm_mass']>0.
