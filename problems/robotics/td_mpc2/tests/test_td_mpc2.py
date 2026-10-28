import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.td_mpc2.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'SimNorm needs complete groups and elite values must match the candidate axis.'
    x=torch.ones(2,8,dtype=torch.float64);y=impl.simnorm(x,4);assert y.shape==x.shape
    with pytest.raises(ValueError):impl.simnorm(x,3)
    with pytest.raises(ValueError):impl.elite_moments(x.reshape(1,2,8),x[0],1.,.1,1.)

@pytest.mark.t1
def test_groups_and_weighted_moments(impl):
    'Hand-computed group probabilities and weighted centered moments verify both operators.'
    x=torch.tensor([[0.,math.log(3),math.log(2),0.]],dtype=torch.float64)
    close(impl.simnorm(x,2),x.new_tensor([[.25,.75,2/3,1/3]]))
    actions=x.new_tensor([[[0.],[2.]],[[1.],[5.]]]);values=x.new_tensor([0.,math.log(3)/2])
    m,s=impl.elite_moments(actions,values,2.,.1,10.);close(m,x.new_tensor([[1.5],[4.]]));close(s,x.new_tensor([[math.sqrt(.75)],[math.sqrt(3)]]))

@pytest.mark.t2
def test_invariance_bounds_and_gradient(impl):
    'Simplex shifts, single elites, standard-deviation bounds and gradients constrain edge behavior.'
    a=torch.tensor([[[1.],[3.]]],dtype=torch.float64,requires_grad=True);v=a.new_tensor([0.,0.]);m,s=impl.elite_moments(a,v,0.,.2,.5)
    close(m,a.new_tensor([[2.]]));close(s,a.new_tensor([[.5]]));g,=torch.autograd.grad(m.sum(),a);close(g,torch.full_like(a,.5))
    m1,s1=impl.elite_moments(a[:,:1],v[:1],10.,.2,1.);close(m1,a[:,0]);close(s1,a.new_tensor([[.2]]))
    x=torch.tensor([[1.,2.,3.,4.]],dtype=torch.float64,requires_grad=True);z=impl.simnorm(x,2);g,=torch.autograd.grad(z[0,0],x);p=1/(1+math.e);close(g,x.new_tensor([[p*(1-p),-p*(1-p),0.,0.]]))
    close(impl.simnorm(x+x.new_tensor([[100.,100.,-100.,-100.]]),2),z)

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.simnorm,impl.elite_moments)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['final_error'] < .03*m['initial_error']
    assert m['simplex_error']<1e-10
