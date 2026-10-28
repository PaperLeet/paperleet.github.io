import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.pc_flow.experiment import run

@pytest.mark.t0
def test_logit_contract(impl):
    """Scores are vectors and total_steps is a positive integer."""
    x=torch.zeros(3,dtype=torch.float64);assert impl.pcflow_loss(x,x,x,x,.3,4).dtype==x.dtype
    with pytest.raises(ValueError):impl.pcflow_loss(x,x,x,x,.3,0)
    with pytest.raises(ValueError):impl.pcflow_loss(x,x[:,None],x,x,.3,4)

@pytest.mark.t1
def test_scalar_classifier_probability_oracle(impl):
    """Scalar log probabilities independently recover both adjacent-time ratios."""
    xs=[torch.tensor(v,dtype=torch.float64) for v in ([1.,-.5],[.2,.8],[-.3,.7],[.4,-.2])]
    total=0
    for vals in zip(*(x.tolist() for x in xs)):
        logs=[-math.log1p(math.exp(-v)) for v in vals];z=1.2*((logs[0]-logs[1])-(logs[2]-logs[3]));total+=math.log1p(math.exp(-z))
    close(impl.pcflow_loss(*xs,.3,4),xs[0].new_tensor(total/2))

@pytest.mark.t1
def test_stationary_classifier(impl):
    """Equal current/next scores produce zero transition preference and log(2)."""
    x=torch.tensor([2.,-1.]);y=-x
    close(impl.pcflow_loss(x,x,y,y,.7,9),x.new_tensor(math.log(2)))

@pytest.mark.t2
def test_all_four_analytic_gradients(impl):
    """Both times and both trajectories differentiate through log-sigmoid."""
    xs=[torch.tensor([.4,-.6],dtype=torch.float64,requires_grad=True) for _ in range(4)]
    gs=torch.autograd.grad(impl.pcflow_loss(*xs,.5,6),xs)
    base=3*torch.sigmoid(-xs[0].detach())/4
    for g,s in zip(gs,[-1,1,1,-1]):close(g,s*base)

@pytest.mark.t2
def test_beta_time_equivalence(impl):
    """Only beta times T determines the preference temperature."""
    xs=[torch.randn(3,dtype=torch.float64) for _ in range(4)]
    close(impl.pcflow_loss(*xs,.5,4),impl.pcflow_loss(*xs,1.,2))

@pytest.mark.t2
def test_extreme_logits_finite(impl):
    """Log-sigmoid evaluation avoids log(0) for large negative classifier logits."""
    x=torch.tensor([1000.,-1000.],dtype=torch.float64,requires_grad=True);z=torch.zeros_like(x)
    loss=impl.pcflow_loss(x,z,-x,z,.2,5);g,=torch.autograd.grad(loss,x)
    assert torch.isfinite(loss) and torch.isfinite(g).all()

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.pcflow_loss)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['final'] < .12*m['initial']
    assert m['weight'] > 1.
