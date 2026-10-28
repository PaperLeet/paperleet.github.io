import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.ai_infra.zero.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Optimizer states share owner layout and step starts at one."""
    p=torch.ones(2,2,dtype=torch.float64); g=torch.ones(2,4,dtype=torch.float64); z=p*0
    out=impl.zero2_adam_step(p,g,z,z,1,.1)
    assert all(t.shape==p.shape and t.dtype==p.dtype for t in out)
    with pytest.raises(ValueError): impl.zero2_adam_step(p,g,z,z,0,.1)
    with pytest.raises(ValueError): impl.state_bytes(7,2,2)
    with pytest.raises(ValueError): impl.state_bytes(8,2,4)

@pytest.mark.t1
def test_hand_adam_and_reduce_scatter(impl):
    """Opposing replica gradients cancel in owner zero while owner one averages to 2 and -2."""
    p=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64); g=torch.tensor([[1.,2.,3.,-1.],[-1.,-2.,1.,-3.]],dtype=torch.float64)
    before=p.clone(); z=torch.zeros_like(p); q,m,v=impl.zero2_adam_step(p,g,z,z,1,.1,beta1=.5,beta2=.75,eps=.5)
    close(m,torch.tensor([[0.,0.],[1.,-1.]],dtype=torch.float64)); close(v,torch.tensor([[0.,0.],[1.,1.]],dtype=torch.float64))
    close(q,torch.tensor([[1.,2.],[2.92,4.08]],dtype=torch.float64)); close(p,before)

@pytest.mark.t1
def test_memory_stages(impl):
    """For eight parameters over four ranks, stage footprints are 128,56,44,32 bytes."""
    assert [impl.state_bytes(8,4,s) for s in range(4)]==[128,56,44,32]

@pytest.mark.t2
def test_replica_permutation_and_single_rank(impl):
    """Data-rank order is irrelevant and world size one recovers ordinary Adam."""
    p=torch.arange(6,dtype=torch.float64).reshape(3,2); g=torch.arange(18,dtype=torch.float64).reshape(3,6)/7; z=p*0
    one=impl.zero2_adam_step(p,g,z,z,3,.03); two=impl.zero2_adam_step(p,g[[2,0,1]],z,z,3,.03)
    for a,b in zip(one,two): close(a,b)
    for s in range(4): assert impl.state_bytes(8,1,s)==128

@pytest.mark.t2
def test_nonzero_moments_and_zero_gradient(impl):
    """Stored moments still move parameters on a zero-gradient step; moments are not reinitialized."""
    p=torch.ones(1,2,dtype=torch.float64); g=torch.zeros(1,2,dtype=torch.float64); m=p*.2; v=p*.4
    q,a,b=impl.zero2_adam_step(p,g,m,v,2,.1,beta1=.5,beta2=.5,eps=.01)
    close(a,p*.1); close(b,p*.2)
    expected=1-.1*(.1/.75)/((.2/.75)**.5+.01); close(q,p*expected)

@pytest.mark.t3
def test_sharded_training(impl,artifact):
    """Two-rank sharded updates reproduce dense Adam through the entire regression trace."""
    r=run(impl.zero2_adam_step,impl.state_bytes); artifact('zero_training',r)
    assert r['metrics']['trajectory_error']<1e-12
    assert r['metrics']['final_loss']<r['metrics']['initial_loss']*.05
    assert r['metrics']['stage2_bytes']<r['metrics']['replicated_bytes']
