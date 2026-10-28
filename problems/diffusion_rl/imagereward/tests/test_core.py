import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.imagereward.experiment import run

@pytest.mark.t0
def test_shapes_dtype_and_rejections(impl):
    """Public contracts preserve dtype and reject broadcasted preference pairs."""
    for dtype in (torch.float32,torch.float64):
        a=torch.tensor([1.,2.],dtype=dtype);b=a.clone()
        assert impl.ranking_loss(a,b).shape==() and impl.ranking_loss(a,b).dtype==dtype
        x=a.reshape(1,2);close(impl.refl_clean(x,x,.8,.6),x*.5)
        close(a,b)
    with pytest.raises(ValueError):impl.ranking_loss(a,a[:,None])
    with pytest.raises(ValueError):impl.refl_clean(x,x,0.,.3)

@pytest.mark.t1
def test_independent_pairwise_oracle(impl):
    """Independent scalar log1p-exp checks reduction after per-pair loss."""
    p=torch.tensor([2.,-1.,.3],dtype=torch.float64);r=torch.tensor([-1.,2.,.1],dtype=torch.float64)
    expected=sum(max(b-a,0)+math.log1p(math.exp(-abs(b-a))) for a,b in zip(p.tolist(),r.tolist()))/3
    close(impl.ranking_loss(p,r),p.new_tensor(expected))

@pytest.mark.t1
def test_noising_inverse(impl):
    """Amplitude noising followed by ReFL inversion reconstructs every coordinate."""
    x=torch.tensor([[1.,-2.],[3.,.4]],dtype=torch.float64);e=x*.3+1
    close(impl.refl_clean(.8*x+.6*e,e,.8,.6),x)

@pytest.mark.t2
def test_gradient_ownership(impl):
    """Only the predicted noise receives the -sigma/alpha derivative."""
    x=torch.ones(2,3,dtype=torch.float64,requires_grad=True);e=torch.zeros_like(x,requires_grad=True)
    gx,ge=torch.autograd.grad(impl.refl_clean(x,e,.8,.6).sum(),(x,e),allow_unused=True)
    assert gx is None or not gx.any()
    close(ge,e.new_full(e.shape,-.75))

@pytest.mark.t2
def test_extreme_scores_and_gradient(impl):
    """Stable logistic handles both saturated directions with finite gradients."""
    p=torch.tensor([1000.,-1000.],dtype=torch.float64,requires_grad=True);r=torch.zeros_like(p)
    loss=impl.ranking_loss(p,r);g,=torch.autograd.grad(loss,p)
    close(loss,p.new_tensor(500.));close(g,p.new_tensor([0.,-.5]))

@pytest.mark.t2
def test_pair_shift_and_noise_endpoint(impl):
    """Joint score shifts cancel; zero sigma removes the epsilon effect."""
    p=torch.tensor([.2,-.3]);r=torch.tensor([.8,.1]);close(impl.ranking_loss(p,r),impl.ranking_loss(p+4,r+4))
    x=p[None];close(impl.refl_clean(x,x,.5,0),2*x)

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.ranking_loss, impl.refl_clean)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['final'] < .2*m['initial']
    assert m['reward_gain'] > 1.
