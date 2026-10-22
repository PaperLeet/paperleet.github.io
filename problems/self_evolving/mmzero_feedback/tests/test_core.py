import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.self_evolving.mmzero_feedback.experiment import run

def inputs(dtype=torch.float64):
    return torch.tensor([True,True,False,False]),torch.tensor([.8,.2,1.,1.],dtype=dtype),torch.tensor([.5,.9,.5,.5],dtype=dtype),torch.tensor([False,False,False,True])
def evaluate(impl):
    return impl.coder_rewards(*inputs())
def oracle():
    ex,e,c,s=[x.tolist() for x in inputs()]
    return [(1+x+min(y,1-y) if ok else -.1)-.05*bad for ok,x,y,bad in zip(ex,e,c,s)]
def edges(impl):
    a=torch.tensor([[1,0,1,0],[2,2,1,2]]);f=torch.tensor([[True,True,False,False],[True,False,True,True]])
    expected=torch.tensor([[.1,1.,0.,.9],[1.,.9,.1,1.]],dtype=torch.float64)
    close(impl.solver_rewards(a,f,3),expected)
    close(impl.solver_rewards(a,f,3,0),f.to(torch.float64))
    close(impl.solver_rewards(torch.tensor([[2]]),torch.tensor([[False]]),3),torch.tensor([[.9]],dtype=torch.float64))
def invariant(impl):
    ex,e,c,s=inputs();close(impl.coder_rewards(ex,e,1-c,s),evaluate(impl))
    a=torch.tensor([[0,1,1]]);f=torch.tensor([[True,False,True]])
    close(impl.solver_rewards(a.repeat(1,2),f.repeat(1,2),2),impl.solver_rewards(a,f,2).repeat(1,2))
def gradients(impl):
    ex,e,c,s=inputs(torch.float32);e.requires_grad_(True);c.requires_grad_(True)
    r=impl.coder_rewards(ex,e,c,s);assert r.dtype==e.dtype and not r.requires_grad

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
        impl.coder_rewards(*current)
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
