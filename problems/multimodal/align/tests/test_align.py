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

from problems.multimodal.align.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """Scalar loss preserves dtype and rejects invalid temperature and tower shapes."""
    x=torch.eye(3,dtype=torch.float64); old=x.clone()
    assert impl.align_loss(x,x,x.new_tensor(.3)).shape==()
    close(x,old)
    with pytest.raises(ValueError): impl.align_loss(x,x,x.new_tensor(0.))
    with pytest.raises(ValueError): impl.align_loss(x,x[:2],x.new_tensor(1.))

@pytest.mark.t1
def test_scalar_oracle(impl):
    """Independent scalar row/column probabilities include both retrieval directions."""
    x=torch.tensor([[1.,0.],[0.,2.],[1.,1.]],dtype=torch.float64)
    y=x.new_tensor([[.4,1.],[-2.,.3],[1.,-.2]])
    s=[[sum(a*b for a,b in zip(u,v))/math.sqrt(sum(a*a for a in u)*sum(b*b for b in v))/.7 for v in y.tolist()] for u in x.tolist()]
    expected=-sum(math.log(scalar_softmax(s[i])[i])+math.log(scalar_softmax([s[j][i] for j in range(3)])[i]) for i in range(3))/6
    close(impl.align_loss(x,y,x.new_tensor(.7)),x.new_tensor(expected),atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_gradients(impl):
    """Central differences validate both encoder and temperature derivatives."""
    x=torch.tensor([[1.,.2],[-.3,1.]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[.5,1.],[1.,-.4]],dtype=torch.float64,requires_grad=True)
    t=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
    gs=torch.autograd.grad(impl.align_loss(x,y,t),(x,y,t))
    for index,value in enumerate((x,y,t)):
        def fn(z):
            args=[x.detach(),y.detach(),t.detach()];args[index]=z
            return impl.align_loss(*args)
        close(gs[index],finite_difference(fn,value),atol=1e-8,rtol=1e-5)

@pytest.mark.t2
def test_rescaling_and_pair_permutation(impl):
    """Positive row rescaling and a common pair permutation preserve loss."""
    x=torch.tensor([[2.,1.],[-1.,3.],[1.,-.4]],dtype=torch.float64); y=x.flip(0)+.2; t=x.new_tensor(.5)
    close(impl.align_loss(x,y,t),impl.align_loss(x*x.new_tensor([[2.],[.2],[3.]]),y,t))
    p=torch.tensor([2,0,1]);close(impl.align_loss(x,y,t),impl.align_loss(x[p],y[p],t))

@pytest.mark.t2
def test_single_and_zero_features(impl):
    """A singleton has no negatives; all-zero features give log(B)."""
    x=torch.zeros(4,3,dtype=torch.float64)
    close(impl.align_loss(x,x,x.new_tensor(.1)),x.new_tensor(math.log(4)))
    close(impl.align_loss(x[:1],x[:1],x.new_tensor(.1)),x.new_tensor(0.))

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.align_loss);artifact("toy_experiment",result);m=result["metrics"]
    assert m["final"] < .15*m["initial"] and m["retrieval_accuracy"]==1.
