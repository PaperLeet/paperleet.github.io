import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.act.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """An empty validity set has no normalized temporal ensemble."""
    a=torch.zeros(3,2); v=torch.ones(3,dtype=torch.bool)
    assert impl.temporal_ensemble(a,v,.1).shape==(2,)
    with pytest.raises(ValueError): impl.temporal_ensemble(a,~v,.1)
    with pytest.raises(ValueError): impl.temporal_ensemble(a,v,-1.)
    with pytest.raises(ValueError): impl.style_kl(a,a[:1])

@pytest.mark.t1
def test_independent_weighted_oracle(impl):
    """Official ACT weights the surviving predictions oldest first, including valid zero actions."""
    a=torch.tensor([[0.,2.],[999.,999.],[4.,6.],[8.,10.]],dtype=torch.float64)
    valid=torch.tensor([True,False,True,True]);w=[math.exp(-.7*i) for i in range(3)]
    expected=[sum(w[i]*a[[0,2,3][i],d].item() for i in range(3))/sum(w) for d in range(2)]
    close(impl.temporal_ensemble(a,valid,.7),a.new_tensor(expected))
    close(impl.temporal_ensemble(a,valid,0.),a.new_tensor([4.,6.]))

@pytest.mark.t1
def test_kl_hand_value(impl):
    """Diagonal Gaussian KL sums dimensions before averaging the batch."""
    m=torch.tensor([[1.,2.],[0.,1.]],dtype=torch.float64);l=m.new_tensor([[0.,math.log(2.)],[math.log(.5),0.]])
    expected=sum(.5*(u*u+math.exp(v)-1-v) for row,logs in zip(m.tolist(),l.tolist()) for u,v in zip(row,logs))/2
    close(impl.style_kl(m,l),m.new_tensor(expected))

@pytest.mark.t2
def test_ensemble_and_kl_gradients(impl):
    """The ensemble is linear in valid predictions and KL has analytic mu/logvar derivatives."""
    a=torch.arange(6,dtype=torch.float64).reshape(3,2).requires_grad_();v=torch.tensor([True,False,True])
    g,=torch.autograd.grad(impl.temporal_ensemble(a,v,0.).sum(),a)
    close(g,a.new_tensor([[.5,.5],[0.,0.],[.5,.5]]))
    m=torch.tensor([[.2,-.4],[.6,.8]],dtype=a.dtype,requires_grad=True);l=torch.zeros_like(m,requires_grad=True)
    gm,gl=torch.autograd.grad(impl.style_kl(m,l),(m,l));close(gm,m/2);close(gl,torch.zeros_like(l))

@pytest.mark.t2
def test_single_prediction_and_batch_duplication(impl):
    """A single prediction survives any decay and KL is invariant to batch replication."""
    a=torch.tensor([[0.,-2.]],dtype=torch.float64);close(impl.temporal_ensemble(a,torch.tensor([True]),1000.),a[0])
    close(impl.style_kl(a.repeat(3,1),a.repeat(3,1)),impl.style_kl(a,a))

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.temporal_ensemble,impl.style_kl)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['ensemble_mse'] < .1*m['latest_mse']
    assert m['final_kl'] < 1e-4*m['initial_kl']
