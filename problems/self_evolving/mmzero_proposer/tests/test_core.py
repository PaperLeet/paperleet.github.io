import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.self_evolving.mmzero_proposer.experiment import run

def inputs(dtype=torch.float64):
    return torch.tensor([[True,False],[True,True],[False,False]]),torch.tensor([[1.,1.],[.2,.6],[1.,1.]],dtype=dtype),torch.tensor([[.5,.5],[.8,.6],[.5,.5]],dtype=dtype),torch.tensor([[.1,-.2,.3],[0.,0.,0.],[.3,.4,.5]],dtype=dtype),torch.tensor([True,True,False])
def evaluate(impl):
    return impl.proposal_rewards(*inputs())
def oracle():
    ex,e,c,a,v=[x.tolist() for x in inputs()];out=[]
    for flags,ee,cc,adjust,ok in zip(ex,e,c,a,v):
        score=sum((min(x,.5)+min(y,1-y)) if flag else 0. for flag,x,y in zip(flags,ee,cc))/len(flags)+sum(adjust)
        out.append(score if ok else -1.)
    return out
def edges(impl):
    shares=torch.tensor([[.25,.25,.25],[.5,.25,.75],[1.,1.,1.],[0.,0.,0.]],dtype=torch.float64)
    expected=[]
    for row in shares.tolist():
        raw=sum((x-.25)*w for x,w in zip(row,[.45,.20,.35]))*4*.5
        expected.append(-max(-.5,min(.5,raw)))
    close(impl.diversity_adjustment(shares,4),torch.tensor(expected,dtype=shares.dtype))
    close(impl.diversity_adjustment(shares,4,0),torch.zeros(4,dtype=shares.dtype))
    ex,e,c,a,v=inputs();v[:]=True;close(impl.proposal_rewards(ex,e,c,a,v)[2:],a[2:].sum(dim=1))
def invariant(impl):
    ex,e,c,a,v=inputs()
    close(impl.proposal_rewards(ex.repeat(1,2),e.repeat(1,2),c.repeat(1,2),a,v),evaluate(impl))
    close(impl.proposal_rewards(ex,e,1-c,a,v),evaluate(impl))
def gradients(impl):
    ex,e,c,a,v=inputs();e.requires_grad_(True);c.requires_grad_(True);a.requires_grad_(True)
    assert not impl.proposal_rewards(ex,e,c,a,v).requires_grad
    assert not impl.diversity_adjustment(torch.ones(2,3,requires_grad=True),8).requires_grad

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
        impl.proposal_rewards(*current)
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
