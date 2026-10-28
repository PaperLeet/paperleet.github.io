import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.rt_1.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'TokenLearner preserves (B,K,C) and requires aligned spatial score/features.'
    x=torch.ones(2,3,4,dtype=torch.float64);s=torch.zeros(2,3,2,dtype=x.dtype)
    y=impl.token_learner_pool(x,s);assert y.shape==(2,2,4) and y.dtype==x.dtype
    with pytest.raises(ValueError):impl.token_learner_pool(x,s[:1])

@pytest.mark.t1
def test_independent_spatial_oracle(impl):
    'Official spatial pooling is checked against explicit scalar exponentials and sums.'
    x=torch.tensor([[[1.,2.],[3.,-1.],[5.,4.]]],dtype=torch.float64);s=x.new_tensor([[[0.,1.],[1.,-1.],[-1.,2.]]])
    expected=[]
    for k in range(2):
        w=[math.exp(float(s[0,l,k])) for l in range(3)];z=sum(w)
        expected.append([sum(w[l]*float(x[0,l,c]) for l in range(3))/z for c in range(2)])
    close(impl.token_learner_pool(x,s),x.new_tensor([expected]))

@pytest.mark.t2
def test_shift_permutation_and_gradients(impl):
    'Spatial shifts/permutations and softmax Jacobians distinguish the normalization axis.'
    x=torch.tensor([[[1.],[2.],[4.]]],dtype=torch.float64,requires_grad=True);s=torch.zeros(1,3,1,dtype=x.dtype,requires_grad=True)
    out=impl.token_learner_pool(x,s);gx,gs=torch.autograd.grad(out.sum(),(x,s))
    close(gx,torch.ones_like(x)/3);close(gs,(x.detach()-7/3)/3)
    close(impl.token_learner_pool(x,s+1000),out);close(impl.token_learner_pool(x[:,[2,0,1]],s[:,[2,0,1]]),out)
    close(impl.token_learner_pool(x[:,:1],s[:,:1]),x[:,:1])

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.token_learner_pool)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['final_loss'] < .001*m['initial_loss']
