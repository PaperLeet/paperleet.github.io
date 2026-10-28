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

from problems.multimodal.internvl3.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Visual tokens name a valid image and per-image increments stay in (0,1]."""
    v=torch.tensor([False,True]);ids=torch.tensor([-1,0]);d=torch.tensor([.5],dtype=torch.float64)
    assert impl.variable_visual_positions(v,ids,d).dtype==d.dtype
    with pytest.raises(ValueError):impl.variable_visual_positions(v,ids+1,d)
    with pytest.raises(ValueError):impl.variable_visual_positions(v,ids,d*0)

@pytest.mark.t1
def test_manual_two_image_positions(impl):
    """Image-specific fractions accumulate through text boundaries rather than resetting."""
    v=torch.tensor([False,True,True,False,True,True,False]);ids=torch.tensor([-1,0,0,-1,1,1,-1]);d=torch.tensor([.5,.25],dtype=torch.float64)
    close(impl.variable_visual_positions(v,ids,d),d.new_tensor([0.,.5,1.,2.,2.25,2.5,3.5]))

@pytest.mark.t2
def test_delta_gradient(impl):
    """An image increment affects its own token and all subsequent positions."""
    v=torch.tensor([False,True,True,False,True]);ids=torch.tensor([-1,0,0,-1,1]);d=torch.tensor([.5,.25],dtype=torch.float64,requires_grad=True)
    g,=torch.autograd.grad(impl.variable_visual_positions(v,ids,d).sum(),d);close(g,d.new_tensor([7.,1.]))

@pytest.mark.t2
def test_unit_delta_and_text(impl):
    """Unit visual increments recover ordinary sequential positions exactly."""
    v=torch.tensor([True,False,True]);ids=torch.tensor([0,-1,0]);d=torch.tensor([1.],dtype=torch.float64)
    close(impl.variable_visual_positions(v,ids,d),torch.arange(3,dtype=d.dtype))
    close(impl.variable_visual_positions(torch.zeros(4,dtype=torch.bool),torch.full((4,),-1,dtype=torch.long),d[:0]),torch.arange(4,dtype=d.dtype))

@pytest.mark.t2
def test_empty_and_first_visual_origin(impl):
    """The declared origin handles both empty input and a visual first token."""
    d=torch.tensor([.25],dtype=torch.float64)
    assert impl.variable_visual_positions(torch.empty(0,dtype=torch.bool),torch.empty(0,dtype=torch.long),d).shape==(0,)
    close(impl.variable_visual_positions(torch.tensor([True,True]),torch.tensor([0,0]),d),d.new_tensor([-.75,-.5]))

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.variable_visual_positions);artifact("toy_experiment",result);m=result["metrics"]
    assert m["compressed_span"]==5. and m["full_span"]==65. and m["text_gap"]==1. and m["visual_gap"]==.0625
