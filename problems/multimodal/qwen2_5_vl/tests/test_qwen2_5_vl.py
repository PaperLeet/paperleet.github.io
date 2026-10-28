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

from problems.multimodal.qwen2_5_vl.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Window indices are integer metadata and malformed merge grids are rejected."""
    p,c=impl.window_permutation(1,4,6,2,2);assert p.dtype==c.dtype==torch.long and p.shape==(24,)
    with pytest.raises(ValueError):impl.window_permutation(1,3,6,2,2)
    with pytest.raises(ValueError):impl.window_permutation(1,4,6,2,0)

@pytest.mark.t1
def test_hand_window_order(impl):
    """A partial right window retains all eight patches without padding or cross-window mixing."""
    p,c=impl.window_permutation(1,4,6,2,2)
    close(p,torch.tensor([0,1,6,7,2,3,8,9,12,13,18,19,14,15,20,21,4,5,10,11,16,17,22,23]));close(c,torch.tensor([0,16,24]))

@pytest.mark.t2
def test_frame_offsets_and_conservation(impl):
    """Each original patch appears once across frames and all windows conserve length."""
    p,c=impl.window_permutation(2,6,8,2,2)
    assert sorted(p.tolist())==list(range(96)) and c[0]==0 and c[-1]==96
    assert all(b>a for a,b in zip(c.tolist(),c.tolist()[1:]))
    p1,c1=impl.window_permutation(1,6,8,2,2);close(p[:48],p1);close(p[48:],p1+48)

@pytest.mark.t2
def test_gather_backward_routes_original_indices(impl):
    """A real downstream gather sends each ordered weight back to its original patch."""
    p,c=impl.window_permutation(1,4,6,2,2);x=torch.arange(24,dtype=torch.float64).requires_grad_();weight=torch.arange(24,dtype=torch.float64)/10
    grad,=torch.autograd.grad((x[p]*weight).sum(),x)
    expected=[0.]*24
    for i,index in enumerate(p.tolist()):expected[index]=weight[i].item()
    close(grad,x.new_tensor(expected))

@pytest.mark.t2
def test_one_cell_and_large_window(impl):
    """Single-patch windows and one oversized window have well-defined cumulative lengths."""
    p,c=impl.window_permutation(1,2,3,1,1);close(p,torch.arange(6));close(c,torch.arange(7))
    p,c=impl.window_permutation(1,2,3,1,10);close(p,torch.arange(6));close(c,torch.tensor([0,6]))

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.window_permutation);artifact("toy_experiment",result);m=result["metrics"]
    assert m["max_window_mean_error"]<1e-12 and m["windows"]>3
