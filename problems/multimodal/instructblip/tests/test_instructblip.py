import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close


def finite_difference(fn, value):
    expected = torch.zeros_like(value)
    for i in range(value.numel()):
        plus, minus = value.detach().clone(), value.detach().clone()
        plus.reshape(-1)[i] += 1e-6
        minus.reshape(-1)[i] -= 1e-6
        expected.reshape(-1)[i] = (fn(plus) - fn(minus)) / 2e-6
    return expected


def scalar_softmax(values):
    z = max(values)
    exps = [math.exp(x-z) for x in values]
    return [x/sum(exps) for x in exps]

from problems.multimodal.instructblip.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """The query count remains fixed and invalid instruction masks fail."""
    q=torch.ones(2,3,dtype=torch.float64);i=torch.ones(4,3,dtype=torch.float64);v=torch.zeros(5,3,dtype=torch.float64)
    assert impl.instruction_query_block(q,i,v,torch.ones(4,dtype=torch.bool)).shape==q.shape
    with pytest.raises(ValueError):impl.instruction_query_block(q,i,v,torch.ones(3,dtype=torch.bool))

@pytest.mark.t1
def test_scalar_two_stage_oracle(impl):
    """A hand-computed two-stage attention requires instruction conditioning before visual attention."""
    q=torch.tensor([[.5,-.2]],dtype=torch.float64);i=q.new_tensor([[1.,.4],[-7.,8.]]);v=q.new_tensor([[1.,0.],[0.,2.]])
    contexts=[q[0].tolist(),i[0].tolist()];weights=scalar_softmax([sum(a*b for a,b in zip(q[0].tolist(),c))/math.sqrt(2) for c in contexts])
    u=[q[0,j].item()+sum(w*c[j] for w,c in zip(weights,contexts)) for j in range(2)];vw=scalar_softmax([u[0]/math.sqrt(2),2*u[1]/math.sqrt(2)])
    close(impl.instruction_query_block(q,i,v,torch.tensor([True,False])),q.new_tensor([[u[0]+vw[0],u[1]+2*vw[1]]]),atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_instruction_gradients(impl):
    """Valid instruction tokens receive gradients and padded tokens receive zero."""
    q=torch.tensor([[.5,.2]],dtype=torch.float64);i=torch.tensor([[.3,1.],[.7,-.4]],dtype=torch.float64,requires_grad=True);v=q.new_tensor([[1.,0.],[0.,1.]])
    fn=lambda z:impl.instruction_query_block(q,z,v,torch.tensor([True,False])).square().sum()
    g,=torch.autograd.grad(fn(i),i);close(g,finite_difference(fn,i),atol=1e-8,rtol=1e-5);close(g[1],torch.zeros_like(g[1]));assert g[0].abs().sum()>0

@pytest.mark.t2
def test_padding_invariance(impl):
    """Arbitrary padded instruction values cannot leak into visual feature extraction."""
    q=torch.ones(1,2,dtype=torch.float64);v=torch.eye(2,dtype=torch.float64);i=q.new_tensor([[.3,.7],[1.,2.]]);mask=torch.tensor([True,False]);j=i.clone();j[1]*=1000
    close(impl.instruction_query_block(q,i,v,mask),impl.instruction_query_block(q,j,v,mask))

@pytest.mark.t2
def test_instruction_changes_output(impl):
    """The same image and queries must produce instruction-dependent features."""
    q=torch.zeros(1,2,dtype=torch.float64);v=torch.eye(2,dtype=torch.float64);mask=torch.tensor([True])
    a=impl.instruction_query_block(q,q.new_tensor([[2.,0.]]),v,mask);b=impl.instruction_query_block(q,q.new_tensor([[0.,2.]]),v,mask)
    assert a[0,0]>a[0,1] and b[0,1]>b[0,0]
    assert torch.isfinite(impl.instruction_query_block(q,q,v,~mask)).all()

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.instruction_query_block);artifact("toy_experiment",result);m=result["metrics"]
    assert m["final"]<.001*m["initial"] and m["feature_margin"]>1.
