import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.self_evolving.rzero_rewards.experiment import run

def inputs(dtype=torch.float64):
    return torch.tensor([.5,.25,1.,.75,.5],dtype=dtype),torch.tensor([8,8,2,4,5]),torch.tensor([True,True,True,True,False])
def evaluate(impl):
    return impl.challenger_rewards(*inputs())
def oracle():
    p,c,v=inputs(); p,c,v=p.tolist(),c.tolist(),v.tolist()
    return [max(0.,1-2*abs(x-.5)-c.count(k)/len(c)) if ok else 0. for x,k,ok in zip(p,c,v)]
def edges(impl):
    for dtype in (torch.float32,torch.float64):
        p,c,v=inputs(dtype)
        close(impl.challenger_rewards(p,c,v,0),torch.tensor([1,.5,0,.5,0],dtype=dtype))
        close(impl.uncertainty_reward(torch.tensor([0.,.5,1.],dtype=dtype)),torch.tensor([0.,1.,0.],dtype=dtype))
    close(impl.challenger_rewards(torch.tensor([.5]),torch.tensor([8]),torch.tensor([True])),torch.zeros(1))
def invariant(impl):
    p,c,v=inputs(); order=torch.tensor([4,2,0,3,1])
    close(impl.challenger_rewards(p[order],c[order],v[order]),evaluate(impl)[order])
    close(impl.challenger_rewards(1-p,c,v),evaluate(impl))
def gradients(impl):
    p,c,v=inputs(); p.requires_grad_(True)
    assert not impl.challenger_rewards(p,c,v).requires_grad
    assert not impl.uncertainty_reward(p).requires_grad

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
        impl.challenger_rewards(*current)
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
