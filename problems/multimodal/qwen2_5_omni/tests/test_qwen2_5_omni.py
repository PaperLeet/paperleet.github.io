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

from problems.multimodal.qwen2_5_omni.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Timestamp streams must be nonnegative and internally ordered."""
    v=torch.tensor([0.,1.],dtype=torch.float64);a=v[:0]
    p,t=impl.time_aligned_interleave(v,a);assert p.dtype==t.dtype==torch.long
    with pytest.raises(ValueError):impl.time_aligned_interleave(v.flip(0),a)
    with pytest.raises(ValueError):impl.time_aligned_interleave(-v,a)
    with pytest.raises(ValueError):impl.time_aligned_interleave(a,a)

@pytest.mark.t1
def test_manual_chunk_order_and_time_grid(impl):
    """Video-first chunks differ from global timestamp sorting and share 40 ms IDs."""
    v=torch.tensor([.2,1.8,2.,4.1],dtype=torch.float64);a=v.new_tensor([0.,.2,1.9,2.,2.04,4.])
    p,t=impl.time_aligned_interleave(v,a)
    close(p,torch.tensor([0,1,4,5,6,2,7,8,3,9]));close(t,torch.tensor([5,45,0,5,47,50,50,51,102,100]))

@pytest.mark.t2
def test_boundary_belongs_to_next_chunk(impl):
    """The 2-second boundary is half-open and equal-time video precedes audio."""
    v=torch.tensor([2.],dtype=torch.float64);a=v.new_tensor([1.999,2.])
    p,t=impl.time_aligned_interleave(v,a);close(p,torch.tensor([1,0,2]));close(t,torch.tensor([49,50,50]))

@pytest.mark.t2
def test_single_modality_and_duplicate_times(impl):
    """One stream and repeated timestamps preserve stable token order."""
    v=torch.tensor([0.,0.,2.,2.],dtype=torch.float64)
    p,t=impl.time_aligned_interleave(v[:0],v);close(p,torch.arange(4));close(t,torch.tensor([0,0,50,50]))
    p,t=impl.time_aligned_interleave(v,v[:0]);close(p,torch.arange(4))

@pytest.mark.t2
def test_shared_clock_not_frame_counter(impl):
    """Irregular video frames align with audio by real time, not frame index."""
    v=torch.tensor([0.,.12,.4],dtype=torch.float64);a=v.new_tensor([0.,.04,.08,.12,.4])
    p,t=impl.time_aligned_interleave(v,a);restored=torch.empty_like(t);restored[p]=t
    close(restored[:3],torch.tensor([0,3,10]));assert restored[1]==restored[6] and restored[2]==restored[7]

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.time_aligned_interleave);artifact("toy_experiment",result);m=result["metrics"]
    assert m["alignment_error"]==0 and m["chunk_order_valid"] and m["last_time_id"]==100
