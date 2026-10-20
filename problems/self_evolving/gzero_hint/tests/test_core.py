import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.self_evolving.gzero_hint.experiment import run

def inputs(dtype=torch.float64):
    return torch.tensor([[-1.,-2.,-99.],[-2.,-3.,-4.],[5.,6.,7.]],dtype=dtype),torch.tensor([[-3.,-3.,99.],[-1.,-1.,-1.],[0.,0.,0.]],dtype=dtype),torch.tensor([[True,True,False],[True,True,True],[False,False,False]])
def evaluate(impl):
    return impl.hint_delta(*inputs())
def oracle():
    a,b,m=inputs();out=[]
    for aa,bb,mm in zip(a.tolist(),b.tolist(),m.tolist()):
        valid=[x-y for x,y,z in zip(aa,bb,mm) if z];out.append(sum(valid)/len(valid) if valid else 0.)
    return out
def edges(impl):
    p=torch.tensor([.8,.4,.6],dtype=torch.float64)
    got=impl.proposer_rewards(p,torch.tensor([300,100,1000]),torch.tensor([.1,.2,.4],dtype=p.dtype),torch.tensor([True,True,False]))
    close(got,torch.tensor([.67,.2,-1.4],dtype=p.dtype))
    a,b,m=inputs(torch.float32);assert impl.hint_delta(a,b,m).dtype==torch.float32
def invariant(impl):
    a,b,m=inputs();close(impl.hint_delta(a.repeat(1,3),b.repeat(1,3),m.repeat(1,3)),evaluate(impl))
    close(impl.hint_delta(a+17,b+17,m),evaluate(impl))
def gradients(impl):
    a,b,m=inputs();a.requires_grad_(True);b.requires_grad_(True)
    impl.hint_delta(a,b,m).sum().backward()
    expected=torch.tensor([[.5,.5,0],[1/3,1/3,1/3],[0,0,0]],dtype=a.dtype)
    close(a.grad,expected);close(b.grad,-expected)
    p=impl.hint_delta(a,b,m)
    assert not impl.proposer_rewards(p,torch.tensor([100,200,300]),torch.zeros_like(p),torch.ones(3,dtype=torch.bool)).requires_grad

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
        impl.hint_delta(*current)
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
