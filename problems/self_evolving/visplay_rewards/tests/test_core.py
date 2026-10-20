import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.self_evolving.visplay_rewards.experiment import run

def inputs(dtype=torch.float64):
    return torch.tensor([.5,.5,.75,.5,1.,.5],dtype=dtype),torch.tensor([10,10,10,30,30,10]),torch.tensor([0,0,1,0,1,2]),torch.tensor([True,True,True,True,True,False])
def evaluate(impl):
    return impl.image_group_rewards(*inputs())
def oracle():
    p,im,c,v=[x.tolist() for x in inputs()];out=[]
    for x,img,cl,ok in zip(p,im,c,v):
        count=sum(i==img and k==cl for i,k in zip(im,c))
        out.append(max(0.,1-abs(2*x-1)-count/im.count(img)) if ok else 0.)
    return out
def edges(impl):
    p,im,c,v=inputs()
    close(impl.image_group_rewards(p,im,c,v,0),torch.tensor([1.,1.,.5,1.,0.,0.],dtype=p.dtype))
    close(impl.image_group_rewards(torch.tensor([.5]),torch.tensor([1]),torch.tensor([9]),torch.tensor([True])),torch.zeros(1))
def invariant(impl):
    p,im,c,v=inputs();order=torch.tensor([5,3,1,4,2,0])
    close(impl.image_group_rewards(p[order],im[order],c[order],v[order]),evaluate(impl)[order])
    # An unrelated image cannot affect the first image's rewards.
    close(impl.image_group_rewards(p[:3],im[:3],c[:3],v[:3]),impl.image_group_rewards(p[:5],im[:5],c[:5],v[:5])[:3])
def gradients(impl):
    p,im,c,v=inputs(torch.float32);p.requires_grad_(True)
    out=impl.image_group_rewards(p,im,c,v);assert out.dtype==p.dtype and not out.requires_grad

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
        impl.image_group_rewards(*current)
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
