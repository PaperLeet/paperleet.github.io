"""Analytic diagonal derivatives across shared/independent routes and empty guards."""
import numpy as np
import pytest

from stridetorch import torch, UnsupportedOperation, _linalg


def laid_out(data,layout):
    value=torch.from_numpy(data.copy()) if layout=='C' else torch.from_numpy(data.swapaxes(-1,-2).copy()).transpose(-1,-2)
    return value.detach().requires_grad_()


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('shape',[(32,32),(2,32,32)])
@pytest.mark.parametrize('layout',['C','F'])
@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('selected',['a','e','both'])
def test_partial_inverse_jvp_derivatives_match_independent_diagonal_rules(dtype,shape,layout,shared,selected,monkeypatch):
    monkeypatch.setattr(_linalg,'_INVERSE_JVP_SHARED',shared)
    diagonal=np.arange(32,dtype='float64')*.01+2;direction=np.arange(32,dtype='float64')*.002+.2
    weight=np.arange(32,dtype='float64')*.003+.3
    a_data=np.broadcast_to(np.diag(diagonal),shape).astype(dtype);e_data=np.broadcast_to(np.diag(direction),shape).astype(dtype)
    seed=np.broadcast_to(np.diag(weight),shape).astype(dtype)
    a=laid_out(a_data,layout);e=laid_out(e_data,'C');p,value=torch.func.jvp(torch.linalg.inv,(a,),(e,))
    p_seen=[];p.register_hook(lambda g:p_seen.append(1))
    np.testing.assert_allclose(value.numpy(),np.broadcast_to(np.diag(-direction/diagonal**2),shape),rtol=3e-6,atol=0)
    inputs=(a,e) if selected=='both' else (a,) if selected=='a' else (e,)
    for factor in [1.,-.7]:
        gradients=torch.autograd.grad(value,inputs,torch.tensor(seed*factor),retain_graph=True)
        for target,g in zip(('a','e') if selected=='both' else (selected,),gradients):
            expected=np.diag(2*weight*direction/diagonal**3 if target=='a' else -weight/diagonal**2)
            np.testing.assert_allclose(g.numpy(),np.broadcast_to(expected*factor,shape),rtol=3e-6,atol=0)
    assert not p_seen and a.grad is e.grad is None
    np.testing.assert_allclose(torch.autograd.grad(p,a,torch.tensor(seed))[0].numpy(),np.broadcast_to(np.diag(-weight/diagonal**2),shape),rtol=3e-6,atol=0)
    assert p_seen==[1]


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('shape',[(0,2,2),(2,0,0)])
@pytest.mark.parametrize('selected',['a','e','both'])
def test_empty_interleaved_direction_keeps_native_version_guard(dtype,shape,selected):
    data=np.empty(shape,dtype=dtype)
    a=torch.from_numpy(data.copy()).requires_grad_()
    interleaved=np.moveaxis(np.ascontiguousarray(np.moveaxis(data,0,-1)),-1,0)
    e=torch.from_numpy(interleaved).requires_grad_();_,value=torch.func.jvp(torch.inverse,(a,),(e,))
    with torch.no_grad():e.detach().add_(1)
    inputs=(a,e) if selected=='both' else (a,) if selected=='a' else (e,)
    with pytest.raises(UnsupportedOperation,match='modified'):torch.autograd.grad(value,inputs,torch.ones_like(value))
    assert a.grad is e.grad is None
