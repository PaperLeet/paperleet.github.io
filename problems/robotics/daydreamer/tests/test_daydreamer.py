import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.daydreamer.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'Balanced KL requires aligned categorical latents and alpha in the unit interval.'
    x=torch.zeros(2,3,4,dtype=torch.float64);v=impl.balanced_kl(x,x);assert v.shape==() and v.dtype==x.dtype
    with pytest.raises(ValueError):impl.balanced_kl(x,x,1.1)
    with pytest.raises(ValueError):impl.balanced_kl(x,x[:1])

@pytest.mark.t1
def test_independent_categorical_oracle(impl):
    'Explicit probability-space KL sums latent variables and averages the batch.'
    q=torch.tensor([[[.75,.25],[.2,.8]],[[.5,.5],[.9,.1]]],dtype=torch.float64);p=torch.tensor([[[.5,.5],[.6,.4]],[[.7,.3],[.4,.6]]],dtype=q.dtype)
    expected=sum(float(q[b,k,c])*math.log(float(q[b,k,c]/p[b,k,c])) for b in range(2) for k in range(2) for c in range(2))/2
    for alpha in [0.,.3,1.]:close(impl.balanced_kl(q.log(),p.log(),alpha),q.new_tensor(expected))

@pytest.mark.t2
def test_stop_gradient_allocation(impl):
    'Stopped branches allocate prior and posterior gradients independently of the forward KL.'
    q=torch.tensor([[[.7,.3]]],dtype=torch.float64);p=q.new_tensor([[[.4,.6]]]);ql=q.log().requires_grad_();pl=p.log().requires_grad_();alpha=.8
    value=impl.balanced_kl(ql,pl,alpha);gq,gp=torch.autograd.grad(value,(ql,pl));ratio=(q/p).log();kl=(q*ratio).sum(-1,keepdim=True)
    close(gp,alpha*(p-q));close(gq,(1-alpha)*q*(ratio-kl))
    close(impl.balanced_kl(ql+1000,pl+1000,.5),value)
    zq,zp=torch.autograd.grad(impl.balanced_kl(ql,pl,1.),(ql,pl));close(zq,torch.zeros_like(q));close(zp,p-q)

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.balanced_kl)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['final_kl']<1e-5*m['initial_kl']
    assert m['probability_error']<.002
