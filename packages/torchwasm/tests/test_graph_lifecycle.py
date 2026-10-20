"""NumPy-only graph lifetime and gradient-routing contracts for CPython/WASM."""
import gc
import weakref

import numpy as np
import pytest

from torchwasm import UnsupportedOperation,torch


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_multi_root_named_gradients_and_materialized_unused(dtype):
    kind=getattr(torch,dtype);values=np.array([.2,.4,-.3],dtype=dtype)
    x=torch.tensor(values,requires_grad=True);unused=torch.ones(2,dtype=kind,requires_grad=True)
    middle=x.exp();a=middle.square();b=middle*3
    first=torch.tensor([1.,-.2,.5],dtype=kind);second=torch.tensor([-.4,.3,.2],dtype=kind)
    result=torch.autograd.grad((a,b),{'x':x,'unused':unused},(first,second),materialize_grads=True)
    expected=(2*np.exp(values)*first.numpy()+3*second.numpy())*np.exp(values)
    np.testing.assert_allclose(result['x'].numpy(),expected,rtol=2e-6,atol=2e-6)
    assert list(result)==['x','unused'] and result['unused'].tolist()==[0.,0.]
    assert not result['unused'].requires_grad and x.grad is None


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_retained_gradients_and_explicit_graph_reuse(dtype):
    kind=getattr(torch,dtype);x=torch.tensor([2.,3.],dtype=kind,requires_grad=True)
    middle=x.square();middle.retain_grad();loss=middle.square().sum()
    gradient=torch.autograd.grad(loss,x,retain_graph=True)[0]
    np.testing.assert_array_equal(gradient.numpy(),[32.,108.]);assert x.grad is None
    np.testing.assert_array_equal(middle.grad.numpy(),[8.,18.])
    loss.backward()
    np.testing.assert_array_equal(x.grad.numpy(),[32.,108.])
    np.testing.assert_array_equal(middle.grad.numpy(),[16.,36.])
    with pytest.raises(RuntimeError,match='freed|second time'):loss.backward()
    assert not middle.is_leaf and middle.requires_grad


def test_stateless_view_reduction_paths_can_repeat_without_saving_data():
    x=torch.arange(6,dtype=torch.float64).reshape(2,3).requires_grad_()
    result=(-(x+2).T).mean()
    result.backward();result.backward()
    np.testing.assert_allclose(x.grad.numpy(),np.full((2,3),-1/3),rtol=1e-14)


@pytest.mark.parametrize('retain',[False,True])
def test_loss_history_does_not_pin_saved_arrays_after_release(retain):
    x=torch.ones(8,16,requires_grad=True);middle=x.exp()
    reference=weakref.ref(middle._data)
    loss=middle.square().sum();del middle;gc.collect()
    assert reference() is not None
    loss.backward(retain_graph=retain);gc.collect()
    assert (reference() is not None)==retain
    if retain:loss.backward();gc.collect();assert reference() is None
    assert loss.requires_grad and not loss.is_leaf


def test_grad_target_pruning_and_backward_target_execution_are_distinct():
    x=torch.tensor(2.,requires_grad=True);middle=x.square();first=middle.square();second=middle.square()
    middle.backward()
    assert torch.autograd.grad(first,middle)[0].item()==8
    with pytest.raises(RuntimeError,match='freed|second time'):second.backward(inputs=[middle])
    assert middle.grad.item()==8 and middle.retains_grad and x.grad.item()==4


def test_request_validation_does_not_consume_valid_graph():
    x=torch.tensor([2.,3.],requires_grad=True);y=x.square()
    with pytest.raises(RuntimeError,match='shape'):y.backward(torch.ones(3))
    with pytest.raises(RuntimeError,match='require grad'):torch.autograd.grad(y,torch.ones(2),torch.ones(2))
    y.sum().backward();assert x.grad.tolist()==[4.,6.]
    with pytest.raises(RuntimeError):y.requires_grad_(False)
    with pytest.raises(UnsupportedOperation):x.square().backward(torch.ones(2),create_graph=True)
    with pytest.raises(RuntimeError):x.sum(dtype=torch.int64)
    with torch.no_grad():
        integer=x.sum(dtype=torch.int64)
        assert integer.item()==5 and integer.dtype==torch.int64 and not integer.requires_grad


def test_leaf_ownership_has_no_self_cycle():
    x=torch.tensor(2.,requires_grad=True);x.backward();reference=weakref.ref(x)
    del x
    assert reference() is None
