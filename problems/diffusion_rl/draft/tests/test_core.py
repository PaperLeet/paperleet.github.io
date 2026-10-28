import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.draft.experiment import run

@pytest.mark.t0
def test_shapes_and_reward_contract(impl):
    """Rewards are per example and re-noising has an explicit sample axis."""
    x=torch.ones(2,3,dtype=torch.float64);n=torch.zeros(2,2,3,dtype=torch.float64)
    assert impl.draft_lv_loss(x,n,.8,.6,lambda z:z*0,lambda z:z.sum(-1)).dtype==x.dtype
    with pytest.raises(ValueError):impl.draft_lv_loss(x,n,.8,.6,lambda z:z,lambda z:z)
    with pytest.raises(ValueError):impl.draft_lv_loss(x,n[0],.8,.6,lambda z:z,lambda z:z.sum(-1))

@pytest.mark.t1
def test_independent_reward_average(impl):
    """Original and two re-noised branches all contribute to the stated average."""
    x=torch.tensor([[.2],[.7]],dtype=torch.float64);n=x.new_tensor([[[-1.],[1.]],[[.5],[-.5]]])
    expected=0
    for b,v in enumerate(x[:,0].tolist()):
        expected+=(v-1)**2
        for k in range(2):expected+=(v+.75*n[k,b,0].item()-.75*.3-1)**2
    close(impl.draft_lv_loss(x,n,.8,.6,lambda z:torch.full_like(z,.3),lambda z:-(z[:,0]-1).square()),x.new_tensor(expected/6))

@pytest.mark.t1
def test_zero_noise_amplitude(impl):
    """At sigma zero all reward values equal the original reward, regardless of denoiser."""
    x=torch.tensor([[.2],[.7]],dtype=torch.float64);n=torch.randn(3,2,1,dtype=torch.float64)
    close(impl.draft_lv_loss(x,n,.8,0.,lambda z:z+100,lambda z:-z[:,0].square()),x.square().mean())

@pytest.mark.t2
def test_anchor_and_noise_detach(impl):
    """The original sample receives only its original-branch reward derivative."""
    x=torch.tensor([[.2],[.7]],dtype=torch.float64,requires_grad=True);n=torch.zeros(2,2,1,dtype=torch.float64,requires_grad=True)
    gx,gn=torch.autograd.grad(impl.draft_lv_loss(x,n,.8,.6,lambda z:z*0,lambda z:-z[:,0].square()),(x,n),allow_unused=True)
    close(gx,x.detach()/3);assert gn is None or not gn.any()

@pytest.mark.t2
def test_denoiser_gradient(impl):
    """Each reconstruction supplies its own derivative to the denoiser parameter."""
    x=torch.tensor([[.2],[.7]],dtype=torch.float64);n=torch.zeros(2,2,1,dtype=torch.float64);w=x.new_tensor(.3,requires_grad=True)
    loss=impl.draft_lv_loss(x,n,.8,.6,lambda z:w.expand_as(z),lambda z:-z[:,0].square());g,=torch.autograd.grad(loss,w)
    expected=(-2*.75*(x-.75*w.detach())).mean()*2/3
    close(g,expected)

@pytest.mark.t2
def test_reordering_extra_noise_draws(impl):
    """Extra sample order cannot change the low-variance aggregate."""
    x=torch.randn(2,2,dtype=torch.float64);n=torch.randn(3,2,2,dtype=torch.float64)
    f=lambda z:z*.2;r=lambda z:-z.square().sum(-1)
    close(impl.draft_lv_loss(x,n,.8,.6,f,r),impl.draft_lv_loss(x,n[torch.tensor([2,0,1])],.8,.6,f,r))

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.draft_lv_loss)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['final'] < .15*m['initial']
    assert .4<m['weight']<.9
