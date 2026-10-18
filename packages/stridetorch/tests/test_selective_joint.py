"""Traversal-local joint derivatives preserve public gradient routing and lifetime."""
import numpy as np
import pytest

from stridetorch import torch, UnsupportedOperation
from stridetorch._autograd import joint_result


@pytest.mark.parametrize('active',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('selection',['first','last','all'])
@pytest.mark.parametrize('selective',[False,True])
def test_joint_rule_can_opt_into_target_masks_without_changing_legacy_rules(active,selection,selective):
    x=torch.tensor([2.,3.],requires_grad=active[0]);y=torch.tensor([4.,5.],requires_grad=active[1]);calls=[]
    eligible=[i for i,flag in enumerate(active) if flag]
    selected=eligible[:1] if selection=='first' else eligible[-1:] if selection=='last' else eligible
    def backward(seed,needs):
        calls.append(needs)
        return 2*seed if needs[0] else None,3*seed if needs[1] else None
    value=joint_result(2*x.numpy()+3*y.numpy(),(x,y),backward,prune_inputs=selective)
    inputs=tuple((x,y)[i] for i in selected)
    for seed in [np.array([.7,-.2]),np.array([-.3,.9])]:
        result=torch.autograd.grad(value,inputs,torch.tensor(seed),retain_graph=True)
        for i,gradient in zip(selected,result):np.testing.assert_allclose(gradient.numpy(),(2 if i==0 else 3)*seed,rtol=2e-6)
    expected=tuple(flag and i in selected for i,flag in enumerate(active)) if selective else active
    assert calls==[expected,expected] and x.grad is y.grad is None


@pytest.mark.parametrize('selective',[False,True])
def test_shared_joint_inputs_accumulate_all_matching_edges(selective):
    x=torch.tensor([2.,3.],requires_grad=True);frozen=torch.tensor([4.,5.]);calls=[]
    def backward(seed,needs):
        calls.append(needs);return 2*seed,3*seed,None
    value=joint_result(5*x.numpy()+frozen.numpy(),(x,x,frozen),backward,prune_inputs=selective)
    np.testing.assert_allclose(torch.autograd.grad(value,x,torch.tensor([.5,.2]))[0].numpy(),[2.5,1.])
    assert calls==[(True,True,False)]


def test_joint_pruning_uses_construction_flags_and_each_traversals_targets():
    x=torch.tensor([2.],requires_grad=True);y=torch.tensor([3.],requires_grad=True);calls=[]
    def backward(seed,needs):
        calls.append(needs);return 2*seed if needs[0] else None,3*seed if needs[1] else None
    value=joint_result(2*x.numpy()+3*y.numpy(),(x,y),backward,prune_inputs=True)
    x.requires_grad_(False)
    assert torch.autograd.grad(value,y,retain_graph=True)[0].item()==3
    x.requires_grad_(True)
    assert torch.autograd.grad(value,x,retain_graph=True)[0].item()==2
    value.backward(inputs=[x])
    assert calls==[(False,True),(True,False),(True,False)] and x.grad.item()==2 and y.grad is None


def test_joint_pruning_keeps_nonleaf_backward_target_execution():
    x=torch.tensor([2.],requires_grad=True);y=torch.tensor([3.],requires_grad=True);calls=[]
    def backward(seed,needs):
        calls.append(needs);return 2*seed if needs[0] else None,3*seed if needs[1] else None
    value=joint_result(2*x.numpy()+3*y.numpy(),(x,y),backward,prune_inputs=True)
    assert torch.autograd.grad(value,value,retain_graph=True)[0].item()==1 and not calls
    value.backward(inputs=[value])
    assert calls==[(True,True)] and value.grad.item()==1 and x.grad is y.grad is None


@pytest.mark.parametrize('mutated',['x','y','output'])
def test_target_pruning_does_not_relax_saved_storage_guards(mutated):
    x=torch.tensor([2.],requires_grad=True);y=torch.tensor([3.],requires_grad=True);calls=[]
    def backward(seed,needs):
        calls.append(needs);return seed if needs[0] else None,seed if needs[1] else None
    value=joint_result(x.numpy()+y.numpy(),(x,y),backward,prune_inputs=True)
    with torch.no_grad():{'x':x,'y':y,'output':value}[mutated].detach().add_(1)
    with pytest.raises(UnsupportedOperation,match='modified'):torch.autograd.grad(value,x)
    assert not calls and x.grad is y.grad is None


def test_pruned_omission_is_allowed_but_requested_omission_is_rejected():
    x=torch.tensor([2.],requires_grad=True);y=torch.tensor([3.],requires_grad=True)
    value=joint_result(x.numpy()+y.numpy(),(x,y),lambda seed,needs:(seed,None),prune_inputs=True)
    assert torch.autograd.grad(value,x,retain_graph=True)[0].item()==1
    with pytest.raises(RuntimeError,match='required input gradient'):torch.autograd.grad(value,y)
    assert x.grad is y.grad is None
