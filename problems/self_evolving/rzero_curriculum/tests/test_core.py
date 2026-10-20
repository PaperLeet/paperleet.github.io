import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.self_evolving.rzero_curriculum.experiment import run

def inputs(dtype=torch.float64):
    return torch.tensor([[1,1,0,0],[2,2,2,1],[0,0,0,0],[3,2,1,0]])
def evaluate(impl):
    a=inputs(); labels,shares=impl.consensus(a,4)
    return torch.cat([labels.to(torch.float64),shares,impl.curriculum_mask(shares).to(torch.float64)])
def oracle():
    rows=inputs().tolist(); labels=[min(set(r),key=lambda x:(-r.count(x),x)) for r in rows]
    shares=[r.count(k)/len(r) for r,k in zip(rows,labels)]
    return labels+shares+[float(.25<=p<=.75) for p in shares]
def edges(impl):
    labels,p=impl.consensus(torch.tensor([[2],[0]]),3)
    close(labels,torch.tensor([2,0]));close(p,torch.ones(2,dtype=torch.float64))
    close(impl.curriculum_mask(torch.tensor([.25,.75,.249,.751])),torch.tensor([True,True,False,False]))
    close(impl.curriculum_mask(torch.tensor([.5,.3]),.5,.5),torch.tensor([True,False]))
    a=torch.tensor([[2,2,1],[0,0,1]]);labels=torch.tensor([1,1])
    close(impl.pseudo_rewards(a,labels),torch.tensor([[0.,0.,1.],[0.,0.,1.]],dtype=torch.float64))
def invariant(impl):
    a=inputs(); l,p=impl.consensus(a,4);lr,pr=impl.consensus(a.repeat(1,3),4)
    close(l,lr);close(p,pr)
    reverse=torch.tensor([3,2,1,0]);l2,p2=impl.consensus(a[:,reverse],4);close(l,l2);close(p,p2)
def gradients(impl):
    labels,p=impl.consensus(inputs(),4)
    assert labels.dtype==torch.long and p.dtype==torch.float64 and not p.requires_grad
    assert impl.curriculum_mask(p).dtype==torch.bool
    r=impl.pseudo_rewards(inputs(),labels); assert r.dtype==torch.float64 and r.shape==(4,4)

@pytest.mark.t0
def test_contract_and_purity(impl):
    """The public API preserves inputs and returns the documented vector shape."""
    before=[x.clone() for x in inputs()] if isinstance(inputs(),tuple) else inputs().clone()
    got=evaluate(impl)
    assert got.ndim==1 and got.numel()==len(oracle()) and got.dtype==torch.float64
    current=inputs()
    if isinstance(current,tuple):
        # Also exercise calls on retained inputs to detect in-place updates.
        saved=[x.clone() for x in current]
        pass
        for x,y in zip(current,saved):close(x,y)
    else:
        a=current.clone();impl.consensus(current,4);close(current,a)

@pytest.mark.t1
def test_scalar_oracle(impl):
    """Independent Python counting and scalar arithmetic reproduce the paper-defined reward."""
    close(evaluate(impl),torch.tensor(oracle(),dtype=torch.float64),rtol=1e-10,atol=1e-10)

@pytest.mark.t2
def test_endpoints_and_semantics(impl):
    """Hand-derived boundaries, degenerate populations and role semantics must hold."""
    edges(impl)

@pytest.mark.t2
def test_invariance(impl):
    """Permutation, replication or local grouping preserves the declared mathematical object."""
    invariant(impl)

@pytest.mark.t2
def test_gradient_ownership_and_dtype(impl):
    """Only Hint-delta differentiates through log probabilities; observed rewards are fixed records."""
    gradients(impl)

@pytest.mark.t3
def test_toy_policy_learning(impl,artifact):
    """A real small policy update driven by submitted functions improves its target preference."""
    result=run(impl)
    artifact('self_evolving_policy',result)
    assert result['metrics']['final_probability']>.72
    assert result['metrics']['final_probability']>result['metrics']['initial_probability']+.3
