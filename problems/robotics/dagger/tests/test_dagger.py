import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.dagger.experiment import run

@pytest.mark.t0
def test_shapes_and_validation(impl):
    """Algorithm 3.1 preserves action dimensions and rejects invalid mixture probabilities."""
    x=torch.ones(3,2,dtype=torch.float64); u=torch.zeros(3,dtype=x.dtype)
    y=impl.mixture_actions(x,x*2,u,.5); assert y.shape==x.shape and y.dtype==x.dtype
    with pytest.raises(ValueError): impl.mixture_actions(x,x,u,1.1)
    with pytest.raises(ValueError): impl.mixture_actions(x,x,u+1,.5)
    with pytest.raises(ValueError): impl.aggregate_dataset(x,x,x[:2],x)

@pytest.mark.t1
def test_mixture_is_selection_not_average(impl):
    """A Bernoulli policy mixture chooses one entire action vector; beta is not an interpolation weight."""
    e=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=torch.float64); l=-e; d=e.new_tensor([0.,.4,.9])
    close(impl.mixture_actions(e,l,d,.4),e.new_tensor([[1.,2.],[-3.,-4.],[-5.,-6.]]))
    close(impl.mixture_actions(e,l,d,0.),l); close(impl.mixture_actions(e,l,d,1.),e)

@pytest.mark.t1
def test_all_rounds_remain_in_dataset(impl):
    """Dataset aggregation retains old expert labels in order and allows an initially empty buffer."""
    x=torch.tensor([[1.],[2.]],dtype=torch.float64); y=-2*x
    a,b=impl.aggregate_dataset(x[:0],y[:0],x,y)
    a,b=impl.aggregate_dataset(a,b,x+4,y+8)
    close(a,x.new_tensor([[1.],[2.],[5.],[6.]]));close(b,x.new_tensor([[-2.],[-4.],[6.],[4.]]));close(x,x.new_tensor([[1.],[2.]]))

@pytest.mark.t2
def test_selected_gradient_and_dataset_detachment(impl):
    """Only the selected branch receives gradients; stored expert labels are fixed data."""
    e=torch.ones(2,1,dtype=torch.float64,requires_grad=True); l=torch.ones_like(e,requires_grad=True)
    a=impl.mixture_actions(e,l,e.new_tensor([.1,.9]),.5)
    ge,gl=torch.autograd.grad(a.sum(),(e,l));close(ge,e.new_tensor([[1.],[0.]]));close(gl,e.new_tensor([[0.],[1.]]))
    x,y=impl.aggregate_dataset(e,l,e,l);assert not x.requires_grad and not y.requires_grad

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.mixture_actions,impl.aggregate_dataset)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['final_energy'] < .05*m['initial_energy']
    assert m['examples']==258
    assert abs(m['gain']+.6)<1e-10
