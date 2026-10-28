import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.rdt_1b.experiment import run

@pytest.mark.t0
def test_slot_contract(impl):
    """An embodiment requires unique, in-range semantic slots."""
    a=torch.ones(2,3,2,dtype=torch.float64);u,m=impl.pack_actions(a,[3,0],4)
    assert u.shape==(2,3,4) and m.shape==(4,) and m.dtype==torch.bool
    for slots in ([1,1],[0,4],[1]):
        with pytest.raises(ValueError): impl.pack_actions(a,slots,4)
    with pytest.raises(ValueError): impl.mask_actions(u,torch.ones(4))

@pytest.mark.t1
def test_semantic_scatter_oracle(impl):
    """Local order and global semantic order are distinct, and zero-valued coordinates remain valid."""
    a=torch.tensor([[[2.,0.,-1.],[4.,3.,5.]]],dtype=torch.float64)
    u,m=impl.pack_actions(a,[4,0,2],6)
    close(u,a.new_tensor([[[0.,0.,-1.,0.,2.,0.],[3.,0.,5.,0.,4.,0.]]]))
    assert m.tolist()==[True,False,True,False,True,False]
    close(impl.mask_actions(u+7,m),a.new_tensor([[[7.,0.,6.,0.,9.,0.],[10.,0.,12.,0.,11.,0.]]]))

@pytest.mark.t2
def test_slot_and_mask_gradients(impl):
    """The slot matrix preserves selected-coordinate gradients and execution blocks absent dimensions."""
    a=torch.tensor([[[2.,3.]]],dtype=torch.float64,requires_grad=True)
    u,m=impl.pack_actions(a,[3,1],4);g,=torch.autograd.grad((u*a.new_tensor([1.,2.,3.,4.])).sum(),a)
    close(g,a.new_tensor([[[4.,2.]]]))
    p=torch.ones(1,2,4,dtype=a.dtype,requires_grad=True);g,=torch.autograd.grad(impl.mask_actions(p,m).sum(),p)
    close(g,a.new_tensor([[[0.,1.,0.,1.],[0.,1.,0.,1.]]]))

@pytest.mark.t2
def test_permutation_and_identity(impl):
    """Reordering local coordinates with their slot ids leaves the unified action unchanged."""
    a=torch.arange(6,dtype=torch.float64).reshape(1,2,3)
    u,m=impl.pack_actions(a,[0,2,4],5);v,n=impl.pack_actions(a[:,:,[2,0,1]],[4,0,2],5)
    close(u,v);assert m.tolist()==n.tolist();close(impl.mask_actions(u,m),u)

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.pack_actions,impl.mask_actions)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['final_loss'] < 1e-10*m['initial_loss']
    assert m['invalid_max']==0.
