"""Independent polynomial oracles and native saved-factor JVP contracts."""
import gc
import warnings
import weakref

import numpy as np
import pytest

from torchwasm import torch, _determinants


def layout_tensor(data,layout,requires=True):
    if layout=='F':value=torch.from_numpy(data.T.copy()).T
    elif layout=='S':
        backing=np.zeros((data.shape[0]*2,data.shape[1]*2),dtype=data.dtype)
        backing[::2,::2]=data;value=torch.from_numpy(backing)[::2,::2]
    else:value=torch.from_numpy(data.copy())
    return value.detach().requires_grad_(requires)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('alias',['top','linalg','method'])
@pytest.mark.parametrize('layout',['C','F','S'])
def test_det_jvp_aliases_and_both_reverse_channels_match_2x2_polynomial(dtype,alias,layout):
    data=np.array([[2.,.3],[.4,3.]],dtype=dtype);direction=np.array([[.2,.4],[-.1,.3]],dtype=dtype)
    x=layout_tensor(data,layout);e=layout_tensor(direction,layout)
    fn=torch.det if alias=='top' else torch.linalg.det if alias=='linalg' else lambda a:a.det()
    p,t=torch.func.jvp(fn,(x,),(e,),strict=True)
    a,b,c,d=map(float,data.ravel());det=a*d-b*c
    cofactor=np.array([[d,-c],[-b,a]]);slope=float((cofactor*direction).sum());trace=slope/det
    np.testing.assert_allclose(p.numpy(),det,rtol=3e-6,atol=0)
    np.testing.assert_allclose(t.numpy(),slope,rtol=3e-6,atol=0)
    ga,ge=torch.autograd.grad(t,(x,e),torch.tensor(.7,dtype=dtype),retain_graph=True)
    np.testing.assert_allclose(ga.numpy(),.7*trace*cofactor,rtol=3e-6,atol=3e-8)
    np.testing.assert_allclose(ge.numpy(),.7*cofactor,rtol=3e-6,atol=3e-8)
    gp=torch.autograd.grad(p,x)[0];np.testing.assert_allclose(gp.numpy(),cofactor,rtol=3e-6,atol=3e-8)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('which',['zero','negative_zero','one','maximum','subnormal'])
@pytest.mark.parametrize('needs',[(True,True),(True,False),(False,True)])
def test_singleton_jvp_preserves_native_epsilon_and_ieee_contract(dtype,which,needs):
    with np.errstate(all='ignore'):
        info=np.finfo(dtype);values=dict(zero=dtype(0),negative_zero=dtype(-0.),one=dtype(1),maximum=dtype(info.max),subnormal=np.nextafter(dtype(0),dtype(1)))
        a=values[which];coefficient=dtype(info.eps) if a==0 else a
        trace=np.divide(dtype(1),coefficient);slope=trace*a;de=np.reciprocal(coefficient)*a
    x=torch.tensor([[a]],dtype=dtype,requires_grad=needs[0]);e=torch.ones(1,1,dtype=dtype,requires_grad=needs[1])
    p,t=torch.func.jvp(torch.det,(x,),(e,))
    assert np.signbit(p.item())==np.signbit(a)
    np.testing.assert_allclose(t.numpy(),slope,rtol=3e-6,atol=0,equal_nan=True)
    if slope==0:assert np.signbit(t.item())==np.signbit(slope)
    active=tuple(v for v,needed in zip((x,e),needs) if needed);gradients=iter(torch.autograd.grad(t,active))
    if needs[0]:np.testing.assert_allclose(next(gradients).numpy(),[[trace]],rtol=3e-6,atol=0,equal_nan=True)
    if needs[1]:np.testing.assert_allclose(next(gradients).numpy(),[[de]],rtol=3e-6,atol=0,equal_nan=True)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['C','F','S'])
@pytest.mark.parametrize('needs',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('singular',['rank_one','zero'])
def test_singular_jvp_saved_lu_contract_keeps_ordinary_cofactor_extension(dtype,layout,needs,singular):
    data=np.array([[1.,2.],[2.,4.]] if singular=='rank_one' else [[0.,0.],[0.,0.]],dtype=dtype)
    x=layout_tensor(data,layout,needs[0]);e=torch.ones(2,2,dtype=dtype,requires_grad=needs[1])
    p,t=torch.func.jvp(torch.det,(x,),(e,));assert t.item()==0.
    active=tuple(v for v,needed in zip((x,e),needs) if needed)
    for gradient in torch.autograd.grad(t,active):np.testing.assert_array_equal(gradient.numpy(),0.)
    ordinary=layout_tensor(data,layout);ordinary.det().backward()
    expected=np.array([[4.,-2.],[-2.,1.]]) if singular=='rank_one' else np.zeros((2,2))
    np.testing.assert_allclose(ordinary.grad.numpy(),expected,rtol=3e-6,atol=3e-6)


@pytest.mark.parametrize('needs',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('n',[1,2])
@pytest.mark.parametrize('target',['input','direction','primal','tangent'])
@pytest.mark.parametrize('channel',['primal','tangent'])
def test_jvp_saved_value_mutations_match_native_guards(needs,n,target,channel):
    a=torch.tensor(np.eye(n)*2,requires_grad=needs[0]);e=torch.ones(n,n,dtype=torch.float64,requires_grad=needs[1])
    p,t=torch.func.jvp(torch.det,(a,),(e,));out=p if channel=='primal' else t
    with torch.no_grad():{'input':a,'direction':e,'primal':p,'tangent':t}[target].detach().add_(1)
    if channel=='primal' and not needs[0]:assert not out.requires_grad;return
    guarded=target=='primal' or target=='input' and needs[0]
    if guarded:
        with pytest.raises(RuntimeError):out.sum().backward()
    else:out.sum().backward()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('n',[1,2])
@pytest.mark.parametrize('needs',[(True,True),(True,False),(False,True)])
def test_returned_primal_hooks_are_separate_from_private_tangent_factor(dtype,n,needs):
    x=torch.tensor(np.eye(n)*2,dtype=dtype,requires_grad=needs[0]);e=torch.ones(n,n,dtype=dtype,requires_grad=needs[1])
    p,t=torch.func.jvp(torch.det,(x,),(e,));seen=[]
    if p.requires_grad:p.retain_grad();p.register_hook(lambda g:seen.append(g.detach().clone()))
    t.sum().backward(retain_graph=True)
    assert p.grad is None and seen==[]
    if p.requires_grad:
        p.sum().backward();assert len(seen)==1
        np.testing.assert_array_equal(p.grad.numpy(),1.)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('singular',[False,True])
@pytest.mark.parametrize('policy',['raise','warn','call'])
def test_jvp_preserves_numpy_policy_callbacks_without_warnings(dtype,singular,policy):
    x=torch.tensor(np.diag([0. if singular else 2.,3.]),dtype=dtype,requires_grad=True);e=torch.ones_like(x,requires_grad=True)
    before=np.geterr();handler=np.geterrcall();events=[];callback=lambda *args:events.append(args)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with np.errstate(all=policy,call=callback):
            _,t=torch.func.jvp(torch.det,(x,),(e,));t.backward()
            assert np.geterr()==dict(divide=policy,over=policy,under=policy,invalid=policy)
            assert np.geterrcall() is callback
    assert not events and not caught and np.geterr()==before and np.geterrcall() is handler


@pytest.mark.parametrize('selected',['a','e','both'])
def test_jvp_reverse_prunes_unrequested_pullbacks(monkeypatch,selected):
    x=torch.eye(2,dtype=torch.float64).requires_grad_();e=torch.ones(2,2,dtype=torch.float64,requires_grad=True)
    _,t=torch.func.jvp(torch.det,(x,),(e,));calls=[];original=_determinants._diagonal_solve
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(_determinants,'_diagonal_solve',counted)
    torch.autograd.grad(t,(x,e) if selected=='both' else x if selected=='a' else e)
    assert len(calls)==(2 if selected=='both' else 1)


@pytest.mark.parametrize('n',[2,8])
def test_jvp_snapshots_do_not_retain_nonleaf_tensor_or_its_original_array(n):
    leaf=torch.eye(n,dtype=torch.float64).requires_grad_();x=leaf*2;e=torch.ones(n,n,dtype=torch.float64,requires_grad=True)
    owner=weakref.ref(x);storage=weakref.ref(x._data);p,t=torch.func.jvp(torch.det,(x,),(e,))
    del x;gc.collect();assert owner() is None and storage() is None
    (p+t).sum().backward();assert leaf.grad is not None and e.grad is not None


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(0,0),(2,0,0),(0,2,2),(2,0,1,1)])
def test_empty_jvp_no_grad_has_native_canonical_outputs(dtype,shape):
    x=torch.ones(shape,dtype=dtype,requires_grad=True);e=torch.ones_like(x,requires_grad=True)
    with torch.no_grad():p,t=torch.func.jvp(torch.det,(x,),(e,),strict=True)
    assert not p.requires_grad and not t.requires_grad
    np.testing.assert_array_equal(p.numpy(),1.);np.testing.assert_array_equal(t.numpy(),0.)
    assert p.stride()==t.stride()==(1,)*(len(shape)-2)
