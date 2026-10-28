import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.openvla.experiment import run

@pytest.mark.t0
def test_contract(impl):
    'The official quantizer uses floating ordered edges and integer token IDs.'
    x=torch.zeros(2,3,dtype=torch.float64);e=torch.linspace(-1,1,5,dtype=x.dtype)
    tok=impl.encode_actions(x,e,20);assert tok.shape==x.shape and tok.dtype==torch.int64
    with pytest.raises(ValueError):impl.encode_actions(x,e,4)
    with pytest.raises(ValueError):impl.decode_actions(tok.float(),e,20)
    with pytest.raises(ValueError):impl.unnormalize_actions(x,x[0],x[0],torch.ones(3,dtype=torch.bool))

@pytest.mark.t1
def test_exact_edges_reverse_ids_and_clipping(impl):
    'Right-open digitize boundaries and reversed vocabulary indices determine exact codes.'
    e=torch.tensor([-1.,-.5,0.,.5,1.],dtype=torch.float64);x=e.new_tensor([-2.,-1.,-.75,-.5,0.,.5,1.,2.])
    tok=impl.encode_actions(x,e,10);assert tok.tolist()==[9,9,9,8,7,6,5,5]
    close(impl.decode_actions(tok,e,10),e.new_tensor([-.75,-.75,-.75,-.25,.25,.75,.75,.75]))
    close(impl.decode_actions(torch.tensor([0,9]),e,10),e.new_tensor([.75,-.75]))

@pytest.mark.t2
def test_masked_physical_units_and_gradients(impl):
    'Quantile scaling applies only to selected coordinates and scales their gradients.'
    x=torch.tensor([[-1.,0.,1.]],dtype=torch.float64,requires_grad=True);lo=x.new_tensor([2.,5.,-4.]);hi=x.new_tensor([6.,7.,4.]);mask=torch.tensor([True,False,True])
    y=impl.unnormalize_actions(x,lo,hi,mask);close(y,x.new_tensor([[2.,0.,4.]]));g,=torch.autograd.grad(y.sum(),x);close(g,x.new_tensor([[2.,1.,4.]]))
    e=torch.linspace(-1,1,256,dtype=x.dtype);end=impl.decode_actions(impl.encode_actions(x,e,32000),e,32000);assert torch.isfinite(end).all()

@pytest.mark.t3
def test_teaching_experiment(impl, artifact):
    """The submitted operators drive the deterministic experiment, not a reference import."""
    result = run(impl.encode_actions,impl.decode_actions,impl.unnormalize_actions)
    artifact("toy_experiment", result)
    m = result["metrics"]
    assert m['max_normalized_error']<=1/32+1e-10
    assert m['max_physical_error']<=1/16+1e-10
    assert m['roundtrip_codes']>80
