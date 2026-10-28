import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.deepmimic.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'Precomputed quaternion angles and vector-error components retain per-example rewards.'
    p=torch.zeros(2,3,dtype=torch.float64);v=torch.zeros(2,3,3,dtype=p.dtype);e=torch.zeros(2,2,3,dtype=p.dtype);c=torch.zeros(2,3,dtype=p.dtype)
    r=impl.imitation_reward(p,v,e,c);assert r.shape==(2,) and r.dtype==p.dtype
    with pytest.raises(ValueError):impl.imitation_reward(p-1,v,e,c)
    with pytest.raises(ValueError):impl.imitation_reward(p,v[:1],e,c)

@pytest.mark.t1
def test_independent_component_scales(impl):
    'Section 5.3 component scales and weights match an independent scalar calculation.'
    p=torch.tensor([[.2,.3]],dtype=torch.float64);v=p.new_tensor([[[.1,.2,.3],[.4,.5,.6]]]);e=p.new_tensor([[[.1,0.,0.],[0.,.2,0.]]]);c=p.new_tensor([[.1,.2,.3]])
    expected=.65*math.exp(-2*.13)+.1*math.exp(-.1*.91)+.15*math.exp(-40*.05)+.1*math.exp(-10*.14)
    close(impl.imitation_reward(p,v,e,c),p.new_tensor([expected]));close(impl.imitation_reward(p*0,v*0,e*0,c*0),p.new_tensor([1.]))

@pytest.mark.t2
def test_component_gradient_and_batch_independence(impl):
    'Pose gradients and replicated examples preserve independent per-sample reward semantics.'
    p=torch.tensor([[.2,.3]],dtype=torch.float64,requires_grad=True);v=torch.zeros(1,2,3,dtype=p.dtype);e=torch.zeros(1,1,3,dtype=p.dtype);c=torch.zeros(1,3,dtype=p.dtype)
    r=impl.imitation_reward(p,v,e,c);g,=torch.autograd.grad(r.sum(),p);close(g,-2.6*p.detach()*math.exp(-.26))
    many=impl.imitation_reward(p.repeat(3,1),v.repeat(3,1,1),e.repeat(3,1,1),c.repeat(3,1));close(many,r.repeat(3));assert (many>0).all() and (many<=1).all()
    close(impl.imitation_reward(p,v,-e,-c),r)

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.imitation_reward)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['perfect_reward']==1.
    assert m['initial_reward']<.75
    assert m['monotone_steps']==30
    assert m['bounded']
