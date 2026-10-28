import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close


def finite_difference(fn, value):
    expected = torch.zeros_like(value)
    for i in range(value.numel()):
        plus, minus = value.detach().clone(), value.detach().clone()
        plus.reshape(-1)[i] += 1e-6
        minus.reshape(-1)[i] -= 1e-6
        expected.reshape(-1)[i] = (fn(plus) - fn(minus)) / 2e-6
    return expected


def scalar_softmax(values):
    z = max(values)
    exps = [math.exp(x-z) for x in values]
    return [x/sum(exps) for x in exps]

from problems.multimodal.siglip_2.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """The mask selects patches and an empty supervision set is rejected."""
    z=torch.zeros(1,2,3,dtype=torch.float64);q=torch.ones_like(z)/3;m=torch.tensor([[True,False]])
    assert impl.masked_patch_distillation(z,q,m,1.).dtype==z.dtype
    with pytest.raises(ValueError): impl.masked_patch_distillation(z,q,~(m|m.flip(1)),1.)
    with pytest.raises(ValueError): impl.masked_patch_distillation(z,q*2,m,1.)

@pytest.mark.t1
def test_scalar_oracle(impl):
    """Manual cross-entropy is normalized by selected patches, not all patches."""
    z=torch.tensor([[[2.,0.],[4.,-1.],[0.,1.]],[[1.,2.],[0.,0.],[2.,1.]]],dtype=torch.float64)
    q=z.new_tensor([[[.8,.2],[1.,0.],[.3,.7]],[[.1,.9],[.5,.5],[.6,.4]]]);m=torch.tensor([[True,False,True],[False,False,True]])
    expected=0.
    for b,p in ((0,0),(0,2),(1,2)):
        probs=scalar_softmax([v/.6 for v in z[b,p].tolist()])
        expected-=sum(a*math.log(v) for a,v in zip(q[b,p].tolist(),probs))/3
    close(impl.masked_patch_distillation(z,q,m,.6),z.new_tensor(expected),atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_stopped_teacher_and_student_gradient(impl):
    """Teacher is stopped; student derivative is mask*(softmax-q)/(M*tau)."""
    z=torch.tensor([[[1.,-.2],[.3,.5]]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[[.2,.8],[.7,.3]]],dtype=torch.float64,requires_grad=True);m=torch.tensor([[True,False]])
    g,t=torch.autograd.grad(impl.masked_patch_distillation(z,q,m,.5),(z,q),allow_unused=True)
    assert t is None
    close(g,(torch.softmax(z.detach()/.5,-1)-q.detach())*m[...,None]/.5,atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_unmasked_values_irrelevant(impl):
    """Changing an unmasked logit cannot change masked feature learning."""
    z=torch.zeros(1,2,3,dtype=torch.float64);q=torch.ones_like(z)/3;m=torch.tensor([[True,False]])
    changed=z.clone();changed[0,1]=z.new_tensor([100.,-100.,0.])
    close(impl.masked_patch_distillation(z,q,m,1.),impl.masked_patch_distillation(changed,q,m,1.))

@pytest.mark.t2
def test_uniform_and_extreme(impl):
    """Uniform predictions have log(K) loss, while stable logits remain finite."""
    z=torch.zeros(2,3,4,dtype=torch.float64);q=torch.ones_like(z)/4;m=torch.ones(2,3,dtype=torch.bool)
    close(impl.masked_patch_distillation(z,q,m,1.),z.new_tensor(math.log(4)))
    z[...,0]=1000.;assert torch.isfinite(impl.masked_patch_distillation(z,q,m,.1))

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.masked_patch_distillation);artifact("toy_experiment",result);m=result["metrics"]
    assert m["final"]<.1*m["initial"] and m["unmasked_movement"]==0.
