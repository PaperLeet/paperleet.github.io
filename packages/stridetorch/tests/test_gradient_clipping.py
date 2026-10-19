"""Geometry, training integration, aliases and gradient-buffer ownership contracts."""
import importlib
import sys
import warnings

import numpy as np
import pytest
from stridetorch import torch, as_torch


def parameter(gradient):
    p=torch.zeros(gradient.shape,dtype=gradient.dtype);p.grad=gradient
    return p


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('foreach',[None,False,True])
@pytest.mark.parametrize('enabled',[False,True])
def test_known_norm_geometry_preserves_buffers_and_caller_mode(dtype,foreach,enabled):
    g=torch.tensor([3.,-4.],dtype=getattr(torch,dtype));p=parameter(g)
    with torch.set_grad_enabled(enabled):
        total=torch.nn.utils.clip_grad_norm_((x for x in [p]),2.,foreach=foreach)
        assert torch.is_grad_enabled() is enabled
    assert p.grad is g and total.shape==() and not total.requires_grad
    np.testing.assert_allclose(total.numpy(),5.,rtol=1e-6)
    np.testing.assert_allclose(g.numpy(),np.array([3.,-4.])*2/(5+1e-6),rtol=2e-6)


@pytest.mark.parametrize('order,expected',[(0.,2.),(1.,10.),(2.,np.sqrt(34)),(float('inf'),4.),(-float('inf'),3.)])
def test_norm_aggregates_per_gradient_norms(order,expected):
    params=[parameter(torch.tensor([3.,4.],dtype=torch.float64)),parameter(torch.tensor([-3.],dtype=torch.float64))]
    # p=0 counts the nonzero per-gradient norms; retain the native two-stage rule.
    actual=torch.nn.utils.clip_grad_norm_(params,0.,norm_type=order)
    np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-12)
    for p in params:np.testing.assert_array_equal(p.grad.numpy(),np.zeros(p.shape))


@pytest.mark.parametrize('foreach',[None,False,True])
def test_shared_gradient_entries_are_not_deduplicated(foreach):
    g=torch.tensor([3.,4.],dtype=torch.float64);p=parameter(g);q=parameter(g)
    norm=torch.nn.utils.clip_grad_norm_([p,q],1.,foreach=foreach)
    np.testing.assert_allclose(norm.numpy(),np.sqrt(50.),rtol=1e-12)
    scale=1/(np.sqrt(50.)+1e-6)
    np.testing.assert_allclose(g.numpy(),np.array([3.,4.])*scale**2,rtol=1e-12)
    assert p.grad is q.grad is g


@pytest.mark.parametrize('foreach',[None,False,True])
def test_value_clipping_writes_strided_gradient_storage(foreach):
    root=np.arange(24,dtype='float64').reshape(4,6)-10;expected=root.copy()
    g=torch.from_numpy(root[::2,1::2]);p=parameter(g);stride=g.stride()
    expected[::2,1::2]=np.clip(expected[::2,1::2],-2.,2.)
    torch.nn.utils.clip_grad_value_((x for x in [p]),2.,foreach=foreach)
    assert p.grad is g and g.stride()==stride
    np.testing.assert_array_equal(root,expected)


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_rejection_precedes_all_gradient_writes(bad):
    first=torch.tensor([3.,4.]);second=torch.tensor([bad,1.]);params=[parameter(first),parameter(second)]
    before=first.numpy().copy();version=first._version
    with pytest.raises(RuntimeError):torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    np.testing.assert_array_equal(first.numpy(),before);assert first._version==version
    assert torch.is_grad_enabled()


@pytest.mark.parametrize('kind',['norm','value'])
def test_clipping_preserves_differentiable_gradient_history(kind):
    root=torch.tensor([3.,4.],dtype=torch.float64,requires_grad=True);g=root*2;p=parameter(g)
    if kind=='norm':torch.nn.utils.clip_grad_norm_(p,1.)
    else:torch.nn.utils.clip_grad_value_(p,1.)
    assert p.grad is g and g.requires_grad and not g.is_leaf
    g.sum().backward();np.testing.assert_array_equal(root.grad.numpy(),[2.,2.])


def test_clipping_invalidates_saved_gradient_values():
    g=torch.tensor([3.,4.],requires_grad=True);saved=g.square().sum();p=parameter(g)
    torch.nn.utils.clip_grad_norm_(p,1.)
    with pytest.raises(RuntimeError):saved.backward()


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_empty_norm_follows_default_dtype_and_missing_gradients(dtype):
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch,dtype))
        for params in [[],[torch.ones(2)]]:
            value=torch.nn.utils.clip_grad_norm_(params,1.)
            assert value.item()==0 and value.dtype==getattr(torch,dtype)
    finally:torch.set_default_dtype(old)


def test_value_conversion_precedes_generator_consumption():
    consumed=[]
    def parameters():consumed.append(True);yield parameter(torch.ones(2))
    with pytest.raises(ValueError):torch.nn.utils.clip_grad_value_(parameters(),'invalid')
    assert consumed==[]


def test_alias_context_restores_imports_and_exposes_nested_utilities():
    previous={name:value for name,value in sys.modules.items() if name=='torch' or name.startswith('torch.')}
    with as_torch() as namespace:
        direct=importlib.import_module('torch.nn.utils');nested=importlib.import_module('torch.nn.utils.clip_grad')
        assert direct is namespace.nn.utils
        for name in ['clip_grad_norm_','clip_grad_norm','clip_grad_value_']:assert getattr(direct,name) is getattr(nested,name)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always');direct.clip_grad_norm([],1.)
        assert len(seen)==1 and seen[0].category is FutureWarning
    assert {name:value for name,value in sys.modules.items() if name=='torch' or name.startswith('torch.')}==previous


def test_clipping_in_a_complete_sgd_training_step():
    x=torch.tensor([3.,4.],dtype=torch.float64,requires_grad=True);optimizer=torch.optim.SGD([x],lr=.1)
    optimizer.zero_grad();x.square().sum().backward();gradient=x.grad
    total=torch.nn.utils.clip_grad_norm_([x],1.);optimizer.step()
    assert x.grad is gradient
    np.testing.assert_allclose(total.numpy(),10.,rtol=1e-12)
    np.testing.assert_allclose(x.detach().numpy(),np.array([3.,4.])-.1*np.array([6.,8.])/(10+1e-6),rtol=1e-12)
