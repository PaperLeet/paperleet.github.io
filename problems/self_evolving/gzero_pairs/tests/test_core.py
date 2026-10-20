import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.self_evolving.gzero_pairs.experiment import run

def inputs(dtype=torch.float64):
    return torch.tensor([3.,-1.,0.,2.,0.],dtype=dtype),torch.tensor([True,True,True,True,False])
def records():
    return [dict(question='q'+str(i),hint='secret'+str(i),unassisted='bad'+str(i),assisted='good'+str(i)) for i in range(5)]
def evaluate(impl):
    return impl.preference_mask(*inputs()).to(torch.float64)
def oracle():
    d,q=inputs();v=d.tolist();s=sorted(v);median=(s[(len(v)-1)//2]+s[len(v)//2])/2
    return [float(a<=median and ok) for a,ok in zip(v,q.tolist())]
def edges(impl):
    close(impl.preference_mask(torch.tensor([1.,1.,1.]),torch.tensor([True,True,True])),torch.ones(3,dtype=torch.bool))
    close(impl.preference_mask(torch.tensor([-2.]),torch.tensor([True])),torch.tensor([True]))
    d=torch.tensor([-3.,-2.,10.,11.]);q=torch.tensor([False,False,True,True])
    close(impl.preference_mask(d,q),torch.zeros(4,dtype=torch.bool))
    assert impl.build_preferences(records(),inputs()[0],torch.zeros(5,dtype=torch.bool))==[]
def invariant(impl):
    d,q=inputs();close(impl.preference_mask(d*4+8,q),impl.preference_mask(d,q))
    result=impl.build_preferences(records(),d,q)
    assert result==[dict(prompt='q1',chosen='good1',rejected='bad1'),dict(prompt='q2',chosen='good2',rejected='bad2')]
def gradients(impl):
    before=records();d,q=inputs();out=impl.build_preferences(before,d,q);out[0]['prompt']='changed'
    assert before==records()
    assert impl.preference_mask(d.requires_grad_(True),q).dtype==torch.bool

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
        impl.preference_mask(*current)
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
