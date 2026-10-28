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

from problems.multimodal.imagebind.experiment import run

@pytest.mark.t0
def test_contract(impl):
    """All modalities must share the image pairing dimensions."""
    x=torch.eye(2,dtype=torch.float64);m=x[None].repeat(3,1,1)
    assert impl.image_anchor_loss(x,m,x.new_tensor(.5)).shape==()
    with pytest.raises(ValueError):impl.image_anchor_loss(x,m[:,:,:1],x.new_tensor(.5))

@pytest.mark.t1
def test_scalar_pair_graph(impl):
    """Only image-to-modality edges contribute, with a mean across modalities."""
    x=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64);m=x.new_tensor([[[1.,1.],[-1.,2.]],[[0.,1.],[1.,0.]]]);total=0.
    for modal in m.tolist():
        scores=[[v[a]/math.sqrt(sum(z*z for z in v))/.8 for v in modal] for a in range(2)]
        total-=sum(math.log(scalar_softmax(scores[i])[i])+math.log(scalar_softmax([scores[j][i] for j in range(2)])[i]) for i in range(2))/8
    close(impl.image_anchor_loss(x,m,x.new_tensor(.8)),x.new_tensor(total),atol=1e-12,rtol=1e-12)

@pytest.mark.t2
def test_all_modalities_receive_gradients(impl):
    """Independent central differences check gradients to every image-paired modality."""
    x=torch.eye(2,dtype=torch.float64);m=torch.tensor([[[.3,1.],[1.,.2]],[[.4,.7],[.9,-.3]]],dtype=torch.float64,requires_grad=True)
    fn=lambda z:impl.image_anchor_loss(x,z,x.new_tensor(.6));g,=torch.autograd.grad(fn(m),m)
    close(g,finite_difference(fn,m),atol=1e-8,rtol=1e-5)
    assert g[0].abs().sum()>0 and g[1].abs().sum()>0

@pytest.mark.t2
def test_modality_order_and_duplication(impl):
    """Modality permutation and duplicating all modalities preserve the average objective."""
    x=torch.eye(2,dtype=torch.float64);m=x.new_tensor([[[1.,.3],[.4,1.]],[[1.,1.],[1.,-.2]]]);t=x.new_tensor(.7)
    close(impl.image_anchor_loss(x,m,t),impl.image_anchor_loss(x,m.flip(0),t))
    close(impl.image_anchor_loss(x,m,t),impl.image_anchor_loss(x,m.repeat(2,1,1),t))

@pytest.mark.t2
def test_single_pair_and_zero_features(impl):
    """No negatives yields zero; zero similarities yield log(batch)."""
    x=torch.zeros(3,2,dtype=torch.float64);m=torch.zeros(2,3,2,dtype=torch.float64)
    close(impl.image_anchor_loss(x,m,x.new_tensor(1.)),x.new_tensor(math.log(3)))
    close(impl.image_anchor_loss(x[:1],m[:,:1],x.new_tensor(1.)),x.new_tensor(0.))

@pytest.mark.t3
def test_behavioral_experiment(impl, artifact):
    """The submitted mechanism drives a deterministic training or retrieval experiment."""
    result=run(impl.image_anchor_loss);artifact("toy_experiment",result);m=result["metrics"]
    assert m["final"]<.1*m["initial"] and m["cross_modal_accuracy"]==1.
