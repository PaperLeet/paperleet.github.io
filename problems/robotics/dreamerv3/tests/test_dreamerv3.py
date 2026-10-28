import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.dreamerv3.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'Two-hot labels append the support axis and require ordered floating centers.'
    x=torch.tensor([0.,1.],dtype=torch.float64);b=x.new_tensor([-1.,0.,1.]);y=impl.two_hot(x,b);assert y.shape==(2,3) and y.dtype==x.dtype
    with pytest.raises(ValueError):impl.two_hot(x,b.flip(0))
    with pytest.raises(ValueError):impl.symlog(torch.tensor([1]))

@pytest.mark.t1
def test_signed_transform_and_nonuniform_support(impl):
    'Signed transforms and hand-calculated weights recover values on nonuniform supports.'
    x=torch.tensor([-3.,0.,8.],dtype=torch.float64);z=x.new_tensor([-math.log(4),0.,math.log(9)])
    close(impl.symlog(x),z);close(impl.symexp(z),x)
    b=x.new_tensor([-2.,0.,3.]);v=x.new_tensor([-4.,-1.,0.,1.,3.,4.]);p=impl.two_hot(v,b)
    close(p,x.new_tensor([[1.,0.,0.],[.5,.5,0.],[0.,1.,0.],[0.,2/3,1/3],[0.,0.,1.],[0.,0.,1.]]));close((p*b).sum(-1),v.clamp(-2,3))

@pytest.mark.t2
def test_transform_gradient_and_fixed_labels(impl):
    'Away-from-zero transform derivatives and detached target labels follow the source formulation.'
    x=torch.tensor([-3.,2.],dtype=torch.float64,requires_grad=True);g,=torch.autograd.grad(impl.symlog(x).sum(),x);close(g,x.new_tensor([.25,1/3]))
    y=torch.tensor([-.7,.4],dtype=torch.float64,requires_grad=True);g,=torch.autograd.grad(impl.symexp(y).sum(),y);close(g,torch.exp(y.detach().abs()))
    p=impl.two_hot(x,x.new_tensor([-4.,0.,4.]));assert not p.requires_grad;close(p.sum(-1),torch.ones_like(x))

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.symlog,impl.symexp,impl.two_hot)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['roundtrip_error']<1e-9
    assert m['final_cross_entropy']<.4*m['initial_cross_entropy']
    assert m['final_kl']<.1
