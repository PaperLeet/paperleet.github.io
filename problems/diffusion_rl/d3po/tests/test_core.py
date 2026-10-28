import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.d3po.experiment import run

@pytest.mark.t0
def test_shape_contract(impl):
    """D3PO requires aligned trajectory matrices, not broadcasted reference vectors."""
    x=torch.zeros(2,3,dtype=torch.float64);assert impl.d3po_loss(x,x,x,x,1.).shape==()
    with pytest.raises(ValueError):impl.d3po_loss(x,x,x[0],x,1.)
    with pytest.raises(ValueError):impl.d3po_loss(x,x,x,x,-1.)

@pytest.mark.t1
def test_transition_scalar_oracle(impl):
    """Independent scalar per-transition losses expose sum-before-logistic mistakes."""
    p=torch.tensor([[2.,-2.],[.3,.8]],dtype=torch.float64);r=p*.1;rp=p*.2+.1;rr=p*.3-.2
    margins=[.7*((a-c)-(b-d)) for a,b,c,d in zip(p.flatten().tolist(),r.flatten().tolist(),rp.flatten().tolist(),rr.flatten().tolist())]
    expected=sum(max(-z,0)+math.log1p(math.exp(-abs(z))) for z in margins)/4
    close(impl.d3po_loss(p,r,rp,rr,.7),p.new_tensor(expected))

@pytest.mark.t1
def test_equal_policy_reference(impl):
    """Zero implicit preference yields log(2), independent of batch/time size."""
    x=torch.randn(3,4,dtype=torch.float64);y=torch.randn_like(x)
    close(impl.d3po_loss(x,y,x,y,4.),x.new_tensor(math.log(2)))

@pytest.mark.t2
def test_gradient_ownership_and_sign(impl):
    """Current winner/loser gradients are opposite; references remain frozen."""
    xs=[torch.zeros(2,3,dtype=torch.float64,requires_grad=True) for _ in range(4)]
    gs=torch.autograd.grad(impl.d3po_loss(*xs,2.),xs,allow_unused=True)
    close(gs[0],xs[0].new_full((2,3),-1/6));close(gs[1],-gs[0]);assert all(g is None or not g.any() for g in gs[2:])

@pytest.mark.t2
def test_time_duplication(impl):
    """Repeating transitions does not change a per-step mean objective."""
    xs=[torch.randn(2,3,dtype=torch.float64) for _ in range(4)]
    close(impl.d3po_loss(*xs,.9),impl.d3po_loss(*(x.repeat(1,4) for x in xs),.9))

@pytest.mark.t2
def test_stable_large_margins(impl):
    """Stable softplus keeps both strongly preferred and disfavored steps finite."""
    x=torch.tensor([[1000.,-1000.]],dtype=torch.float64,requires_grad=True);z=torch.zeros_like(x)
    l=impl.d3po_loss(x,z,z,z,1.);g,=torch.autograd.grad(l,x)
    close(l,x.new_tensor(500.));close(g,x.new_tensor([[0.,-.5]]))

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.d3po_loss)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['final'] < .15*m['initial']
    assert m['minimum_margin'] > .5
