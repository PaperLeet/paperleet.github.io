"""Determinant error-policy isolation and independent 1x1 polynomial oracles."""
import warnings

import numpy as np
import pytest

from torchwasm import torch, UnsupportedOperation


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('which',['zero','negative_zero','tiny','subnormal','maximum','negative_maximum'])
@pytest.mark.parametrize('seed',[0.,1.,2.,-1.,np.inf,np.nan])
def test_singleton_determinant_and_slope_are_exact_at_extreme_scales(dtype,which,seed):
    info=np.finfo(dtype)
    with np.errstate(all='ignore'):
        values=dict(zero=dtype(0),negative_zero=dtype(-0.),tiny=dtype(info.tiny),subnormal=np.nextafter(dtype(0),dtype(1)),maximum=dtype(info.max),negative_maximum=dtype(-info.max))
    a=values[which];x=torch.tensor([[a]],dtype=dtype,requires_grad=True);g=torch.tensor(seed,dtype=dtype)
    before=np.geterr()
    with np.errstate(all='raise'):
        y=x.det();y.backward(g)
        np.testing.assert_array_equal(y.numpy(),a)
        assert np.signbit(y.item())==np.signbit(a)
        np.testing.assert_array_equal(x.grad.numpy(),np.full((1,1),seed,dtype=dtype))
    assert np.geterr()==before


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('n',[2,32,64,128])
@pytest.mark.parametrize('policy',['raise','warn','call'])
def test_finite_factorization_is_independent_of_numpy_error_policy(dtype,n,policy):
    data=np.eye(n,dtype=dtype);x=torch.tensor(data,requires_grad=True);events=[]
    callback=lambda *args:events.append(args)
    old=np.geterr();old_call=np.geterrcall()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        with np.errstate(all=policy,call=callback):
            y=torch.linalg.det(x);y.backward()
            assert y.item()==1.;np.testing.assert_array_equal(x.grad.numpy(),data)
            assert np.geterr()==dict(divide=policy,over=policy,under=policy,invalid=policy)
            assert np.geterrcall() is callback
    assert np.geterr()==old and np.geterrcall() is old_call
    assert not recorded and not events


@pytest.mark.parametrize('shape',[(1,1),(3,1,1),(2,3,1,1),(0,1,1)])
@pytest.mark.parametrize('layout',['C','strided'])
def test_singleton_vjp_has_native_seed_alias_and_layout_but_forward_is_owned(shape,layout):
    data=np.ones(shape,dtype=np.float64)
    if layout=='strided':data=np.ones((*shape[:-2],2,2),dtype=np.float64)[...,::2,::2]
    x=torch.from_numpy(data).requires_grad_();value=x.det()
    batch=shape[:-2];seed=torch.ones(batch,dtype=torch.float64)
    gradient=torch.autograd.grad(value,x,seed)[0]
    assert not np.shares_memory(value.numpy(),x.numpy())
    if x.numel():
        assert gradient.stride()==seed.stride()+(1,1)
        assert np.shares_memory(gradient.numpy(),seed.numpy())
        gradient.add_(2);np.testing.assert_array_equal(seed.numpy(),np.full(batch,3.))
    else:assert gradient.stride()==x.stride()
    y=x.det();seed=torch.ones(batch,dtype=torch.float64);y.backward(seed)
    assert not np.shares_memory(x.grad.numpy(),seed.numpy())


@pytest.mark.parametrize('target',['input','output'])
def test_singleton_det_keeps_native_saved_value_mutation_guards(target):
    x=torch.ones(1,1,dtype=torch.float64,requires_grad=True);y=x.det()
    with torch.no_grad():(x if target=='input' else y).detach().add_(1)
    with pytest.raises((RuntimeError,UnsupportedOperation)):y.backward()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(0,1,1),(2,0,1,1)])
def test_empty_factory_singleton_vjp_preserves_native_input_layout(dtype,shape):
    x=torch.ones(shape,dtype=dtype,requires_grad=True);value=x.det()
    gradient=torch.autograd.grad(value,x,torch.ones_like(value))[0]
    assert gradient.shape==shape and gradient.stride()==x.stride()
    assert value.stride()==(1,)*(len(shape)-2)


@pytest.mark.parametrize('bad',[np.array([[np.nan]]),np.array([[np.inf]]),np.ones((2,3)),np.ones((2,))])
def test_policy_is_restored_on_existing_invalid_inputs(bad):
    x=torch.tensor(bad);before=np.geterr();callback=np.geterrcall()
    with pytest.raises((RuntimeError,UnsupportedOperation)):torch.det(x)
    assert np.geterr()==before and np.geterrcall() is callback
