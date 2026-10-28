import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.pi0_5.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'Figure 18 requires prefix and expert blocks with an optional FAST block.'
    v=torch.ones(2,6,dtype=torch.bool);m=impl.hybrid_attention_mask(v,2,2);assert m.shape==(2,6,6) and m.dtype==torch.bool
    with pytest.raises(ValueError):impl.hybrid_attention_mask(v,2,4)
    with pytest.raises(ValueError):impl.hybrid_attention_mask(v.float(),2,2)

@pytest.mark.t1
def test_handwritten_connectivity(impl):
    'A hand-written visibility matrix checks all three blockwise attention rules.'
    v=torch.ones(1,6,dtype=torch.bool);m=impl.hybrid_attention_mask(v,2,2)
    expected=torch.tensor([[[1,1,0,0,0,0],[1,1,0,0,0,0],[1,1,1,0,0,0],[1,1,1,1,0,0],[1,1,0,0,1,1],[1,1,0,0,1,1]]],dtype=torch.bool)
    assert torch.equal(m,expected)

@pytest.mark.t2
def test_padding_empty_fast_and_information_gradients(impl):
    'Padding and FAST labels cannot influence expert outputs, but valid prefix/expert values can.'
    valid=torch.tensor([[True,False,True,True,True]]);m=impl.hybrid_attention_mask(valid,2,1);assert not m[:,:,1].any() and not m[:,1,:].any()
    values=torch.arange(5,dtype=torch.float64).reshape(1,5,1).requires_grad_();weights=m[:,3,:].to(dtype=values.dtype);out=(weights.unsqueeze(-1)*values).sum()/weights.sum()
    g,=torch.autograd.grad(out,values);close(g,values.new_tensor([[[1/3],[0.],[0.],[1/3],[1/3]]]))
    nofast=impl.hybrid_attention_mask(torch.ones(1,4,dtype=torch.bool),2,0);assert nofast[0,2,3] and not nofast[0,0,2]

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.hybrid_attention_mask)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['max_expert_label_leak']==0.
    assert m['prefix_response']>.5
    assert m['fast_causal_edges']==0
