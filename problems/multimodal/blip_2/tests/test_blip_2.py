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

from problems.multimodal.blip_2.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """The matching matrix keeps independent image/text batch dimensions."""
    q=torch.ones(2,3,4,dtype=torch.float64);t=torch.ones(5,4,dtype=torch.float64)
    got=impl.max_query_similarity(q,t,q.new_tensor(1.));assert got.shape==(2,5) and got.dtype==q.dtype
    with pytest.raises(ValueError): impl.max_query_similarity(q,t[:,:3],q.new_tensor(1.))

@pytest.mark.t1
def test_manual_winners(impl):
    """Different texts may select different query tokens of the same image."""
    q=torch.tensor([[[2.,0.],[0.,3.],[-1.,0.]]],dtype=torch.float64);t=q.new_tensor([[1.,0.],[0.,1.],[-1.,-1.]])
    close(impl.max_query_similarity(q,t,q.new_tensor(.5)),q.new_tensor([[2.,2.,math.sqrt(2)]]),atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_gradient_winner_routing(impl):
    """Only the maximizing query receives each text's similarity gradient."""
    q=torch.tensor([[[1.,.3],[-1.,.1]]],dtype=torch.float64,requires_grad=True);t=q.new_tensor([[1.,0.]])
    fn=lambda z: impl.max_query_similarity(z,t,q.new_tensor(.7)).sum()
    grad,=torch.autograd.grad(fn(q),q);close(grad,finite_difference(fn,q),atol=1e-8,rtol=1e-5)
    close(grad[:,1],torch.zeros_like(grad[:,1]));assert grad[:,0].abs().sum()>0

@pytest.mark.t2
def test_query_order_and_scale(impl):
    """Reordering queries and rescaling their lengths do not change maximum cosine scores."""
    q=torch.tensor([[[1.,.2],[-.3,1.],[.4,-1.]]],dtype=torch.float64);t=q.new_tensor([[.2,1.],[1.,-.2]]);tau=q.new_tensor(.6)
    close(impl.max_query_similarity(q,t,tau),impl.max_query_similarity(q.flip(1)*3,t*2,tau))

@pytest.mark.t2
def test_zero_and_duplicate_queries(impl):
    """Zero vectors have zero cosine and duplicate maxima do not inflate scores."""
    q=torch.zeros(1,1,2,dtype=torch.float64);t=torch.ones(2,2,dtype=torch.float64)
    close(impl.max_query_similarity(q,t,q.new_tensor(1.)),q.new_zeros(1,2))
    q=q+1;close(impl.max_query_similarity(q,t,q.new_tensor(1.)),impl.max_query_similarity(q.repeat(1,4,1),t,q.new_tensor(1.)))

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.max_query_similarity);artifact("toy_experiment",result);m=result["metrics"]
    assert m["final"]<m["initial"] and m["accuracy"]==1.
