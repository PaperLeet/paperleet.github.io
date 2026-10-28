import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.dancegrpo.experiment import run

@pytest.mark.t0
def test_shapes_and_validation(impl):
    """Prompt/group/reward dimensions remain distinct and malformed groups are rejected."""
    b=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64);n=impl.shared_noise(b,3);assert n.shape==(2,3,2)
    a=impl.group_advantages(torch.ones(2,3,2,dtype=b.dtype));assert a.shape==(2,3)
    with pytest.raises(ValueError):impl.shared_noise(b,1)
    with pytest.raises(ValueError):impl.group_advantages(torch.ones(2,1,2))

@pytest.mark.t1
def test_independent_multireward_oracle(impl):
    """Each reward channel is normalized before channels are added."""
    r=torch.tensor([[[1.,10.],[2.,30.],[4.,20.]],[[8.,4.],[9.,4.],[11.,4.]]],dtype=torch.float64)
    expected=[]
    for prompt in r.tolist():
        means=[sum(row[k] for row in prompt)/3 for k in range(2)];stds=[math.sqrt(sum((row[k]-means[k])**2 for row in prompt)/3) for k in range(2)]
        expected.append([sum((row[k]-means[k])/(stds[k]+1e-8) for k in range(2)) for row in prompt])
    close(impl.group_advantages(r),r.new_tensor(expected))

@pytest.mark.t1
def test_group_initial_noise(impl):
    """Every member of one prompt group starts with exactly its prompt's noise."""
    b=torch.tensor([[1.,-2.],[3.,4.]],dtype=torch.float64);n=impl.shared_noise(b,4)
    for p in range(2):
        for g in range(4):close(n[p,g],b[p])
    n[0,0,0]=9.;assert b[0,0].item()==1. and n[0,1,0].item()==1.

@pytest.mark.t2
def test_prompt_offsets_and_constant_rewards(impl):
    """Independent prompt offsets cancel and constant channels add zero."""
    r=torch.randn(2,4,2,dtype=torch.float64);offset=r.new_tensor([[[100.,-40.]],[[-9.,20.]]])
    close(impl.group_advantages(r),impl.group_advantages(r+offset),atol=1e-10,rtol=1e-10)
    close(impl.group_advantages(torch.ones_like(r)),torch.zeros(2,4,dtype=r.dtype))

@pytest.mark.t2
def test_frozen_noise_and_rewards(impl):
    """Rollout initialization and computed advantages never create trainable targets."""
    b=torch.ones(2,3,requires_grad=True);r=torch.randn(2,4,2,requires_grad=True)
    assert not impl.shared_noise(b,4).requires_grad and not impl.group_advantages(r).requires_grad
    l=torch.zeros(2,4,3,dtype=torch.float64,requires_grad=True);a=torch.randn(2,4,dtype=l.dtype,requires_grad=True)
    gl,ga=torch.autograd.grad(impl.dance_loss(l,a,.2),(l,a),allow_unused=True)
    close(gl,-a.detach()[:,:,None].expand_as(l)/24);assert ga is None or not ga.any()

@pytest.mark.t2
def test_signed_clipping_and_time_mean(impl):
    """Both advantage signs choose the conservative branch and repeat-time means agree."""
    ratios=torch.tensor([[[2.,.5],[2.,.5]]],dtype=torch.float64);a=ratios.new_tensor([[1.,-1.]])
    loss=impl.dance_loss(ratios.log(),a,.2);close(loss,ratios.new_tensor(.275))
    close(loss,impl.dance_loss(ratios.log().repeat(1,1,3),a,.2))

@pytest.mark.t3
def test_real_toy_experiment(impl, artifact):
    """The submitted functions drive every measured update or sampling step."""
    result = run(impl.shared_noise, impl.group_advantages, impl.dance_loss)
    artifact("toy_experiment", result)
    m=result["metrics"]
    assert m['final_error'] < .04*m['initial_error']
    assert m['prompt_groups']==2
