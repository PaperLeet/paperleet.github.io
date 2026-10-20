"""Scalar determinant oracles, unused sign edges and saved-factor JVPs."""
import gc
import math
import pickle
import weakref

import numpy as np
import pytest

from torchwasm import torch, as_torch, UnsupportedOperation


def scalar_slogdet(a):
    d=complex(a[0,0])*complex(a[1,1])-complex(a[0,1])*complex(a[1,0])
    return d/abs(d),math.log(abs(d))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.complex64,torch.complex128])
@pytest.mark.parametrize('layout',['C','F','strided'])
def test_slogdet_aliases_and_complex_forward_match_scalar_2x2_oracle(dtype,layout):
    data=np.array([[2.,.3],[.4,3.]],dtype=dtype)
    if np.dtype(dtype).kind=='c':data+=np.asarray([[.2,-.1],[.3,.4]],dtype=dtype)*np.asarray(1j,dtype=dtype)
    if layout=='F':data=np.asfortranarray(data)
    if layout=='strided':
        backing=np.zeros((4,4),dtype=dtype);backing[::2,::2]=data;data=backing[::2,::2]
    expected_sign,expected_log=scalar_slogdet(data);x=torch.from_numpy(data)
    if np.dtype(dtype).kind!='c':expected_sign=expected_sign.real
    for fn in [torch.slogdet,torch.linalg.slogdet,lambda x:x.slogdet()]:
        out=fn(x);assert out._fields==('sign','logabsdet') and out.sign.dtype==np.dtype(dtype)
        assert out.logabsdet.dtype==np.dtype('float32' if np.dtype(dtype).itemsize in [4,8] and np.dtype(dtype).char in 'fF' else 'float64')
        np.testing.assert_allclose(out.sign.numpy(),expected_sign,rtol=3e-6,atol=3e-6);np.testing.assert_allclose(out.logabsdet.numpy(),expected_log,rtol=3e-6,atol=3e-6)
    expected=expected_log+1j*np.angle(expected_sign) if np.dtype(dtype).kind=='c' else expected_log
    np.testing.assert_allclose(torch.logdet(x).numpy(),expected,rtol=3e-6,atol=3e-6)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('negative',[False,True])
@pytest.mark.parametrize('kind',['slogdet','linalg','logdet','method'])
def test_real_log_gradients_match_independent_inverse_formula(dtype,negative,kind):
    data=np.array([[2.,.3],[.4,3.]],dtype=dtype)
    if negative:data[0]*=-1
    determinant=float(data[0,0])*float(data[1,1])-float(data[0,1])*float(data[1,0]);cofactor=np.array([[data[1,1],-data[1,0]],[-data[0,1],data[0,0]]])/determinant
    x=torch.tensor(data,requires_grad=True)
    value=torch.slogdet(x).logabsdet if kind=='slogdet' else torch.linalg.slogdet(x).logabsdet if kind=='linalg' else x.logdet() if kind=='method' else torch.logdet(x)
    value.backward(torch.tensor(.7,dtype=dtype));expected=cofactor*.7
    if negative and kind in ['logdet','method']:expected=np.zeros_like(expected);assert np.isnan(value.item())
    np.testing.assert_allclose(x.grad.numpy(),expected,rtol=3e-6,atol=3e-6)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('shape',[(0,0),(0,2,2),(2,0,0),(2,3,2,2)])
def test_empty_and_batched_logdet_geometry(dtype,shape):
    data=np.broadcast_to(np.eye(shape[-1],dtype=dtype),shape).copy();x=torch.tensor(data,requires_grad=True)
    out=torch.slogdet(x);assert out.sign.shape==out.logabsdet.shape==shape[:-2]
    np.testing.assert_array_equal(out.sign.numpy(),1.);np.testing.assert_array_equal(out.logabsdet.numpy(),0.)
    out.logabsdet.sum().backward();np.testing.assert_array_equal(x.grad.numpy(),data)


@pytest.mark.parametrize('kind',['slogdet','logdet'])
@pytest.mark.parametrize('scale',[1e-100,1e100])
def test_log_determinants_avoid_product_underflow_and_overflow(kind,scale):
    x=torch.tensor(np.eye(4)*scale,dtype=torch.float64,requires_grad=True);out=torch.slogdet(x).logabsdet if kind=='slogdet' else torch.logdet(x)
    assert out.item()==pytest.approx(4*math.log(scale),rel=2e-14)
    out.backward();np.testing.assert_allclose(x.grad.numpy(),np.eye(4)/scale,rtol=3e-14,atol=0.)


@pytest.mark.parametrize('seed',[1.,float('inf'),float('nan')])
def test_real_sign_has_an_unused_reverse_edge_and_retains_its_own_gradient(seed):
    x=torch.eye(2,dtype=torch.float64).requires_grad_();sign=torch.slogdet(x).sign
    seen=[];x.register_hook(lambda g:seen.append(g));sign.retain_grad();sign.backward(torch.tensor(seed,dtype=torch.float64))
    assert sign.requires_grad and x.grad is None and seen==[]
    np.testing.assert_array_equal(sign.grad.numpy(),np.asarray(seed))
    x=torch.eye(2,requires_grad=True);sign=torch.slogdet(x).sign
    assert torch.autograd.grad(sign,x,allow_unused=True)==(None,)
    with pytest.raises(RuntimeError):torch.autograd.grad(torch.slogdet(x).sign,x)


@pytest.mark.parametrize('target',['input','sign','logabs'])
@pytest.mark.parametrize('part',['sign','logabs'])
def test_slogdet_guards_input_and_sign_but_not_logabs_values(target,part):
    x=torch.eye(2,dtype=torch.float64).requires_grad_();out=torch.slogdet(x)
    with torch.no_grad():{'input':x,'sign':out.sign,'logabs':out.logabsdet}[target].detach().add_(1.)
    value=out.sign if part=='sign' else out.logabsdet
    if target!='logabs':
        with pytest.raises(UnsupportedOperation):value.backward()
    else:
        value.backward()
        if part=='sign':assert x.grad is None
        else:np.testing.assert_array_equal(x.grad.numpy(),np.eye(2))


@pytest.mark.parametrize('kind',['slogdet','logdet'])
@pytest.mark.parametrize('negative',[False,True])
def test_log_jvp_and_reverse_direction_follow_saved_factor_contract(kind,negative):
    data=np.array([[2.,.3],[.4,3.]])
    if negative:data[0]*=-1
    a=torch.tensor(data,requires_grad=True);direction=torch.tensor([[.2,.4],[-.1,.3]],dtype=torch.float64,requires_grad=True)
    fn=torch.slogdet if kind=='slogdet' else torch.logdet;primal,tangent=torch.func.jvp(fn,(a,),(direction,))
    if kind=='slogdet':assert tangent.sign.item()==0. and not tangent.sign.requires_grad;primal,tangent=primal.logabsdet,tangent.logabsdet
    det=data[0,0]*data[1,1]-data[0,1]*data[1,0];cofactor=np.array([[data[1,1],-data[1,0]],[-data[0,1],data[0,0]]])/det
    if negative and kind=='logdet':cofactor*=0
    expected=float((cofactor*direction.numpy()).sum());assert tangent.item()==pytest.approx(expected,rel=3e-14,abs=3e-14)
    ga,ge=torch.autograd.grad(tangent,(a,direction),allow_unused=True);assert ga is None
    np.testing.assert_allclose(ge.numpy(),cofactor,rtol=3e-14,atol=3e-14)


def test_real_logdet_graph_releases_matrix_storage_without_retaining_nonleaf_tensor():
    a=torch.eye(8,dtype=torch.float64).requires_grad_();x=a*2;owner=weakref.ref(x);storage=weakref.ref(x._data);out=torch.slogdet(x)
    del x;gc.collect();assert owner() is None and storage() is not None
    out.logabsdet.backward();gc.collect();assert storage() is None


def test_aliases_keyword_validation_and_named_tuple_pickle():
    x=torch.eye(2,dtype=torch.float64)
    with as_torch():
        import torch as alias
        assert alias.linalg.slogdet(A=x).logabsdet.item()==0.
        assert alias.slogdet(input=x).sign.item()==1.
        assert alias.logdet(input=x).item()==0.
    restored=pickle.loads(pickle.dumps(torch.slogdet(x)));assert restored.sign.item()==1. and restored.logabsdet.item()==0.
    with pytest.raises(TypeError):torch.linalg.slogdet(input=x)
    with pytest.raises(TypeError):torch.logdet(x,out=None)
    with pytest.raises(UnsupportedOperation):torch.slogdet(x,out=(x,x))
    for bad in [torch.ones(2,3),torch.ones(3),torch.ones(2,2,dtype=torch.int64)]:
        with pytest.raises((RuntimeError,UnsupportedOperation)):torch.logdet(bad)


@pytest.mark.parametrize('kind',['slogdet','logdet'])
@pytest.mark.parametrize('target',['input','direction','tangent','primal'])
def test_log_jvp_factor_snapshot_allows_native_unsaved_mutations(kind,target):
    a=torch.tensor([[2.,.3],[.4,3.]],dtype=torch.float64,requires_grad=True);e=torch.full_like(a,.2,requires_grad=True)
    p,t=torch.func.jvp(torch.slogdet if kind=='slogdet' else torch.logdet,(a,),(e,))
    if kind=='slogdet':p,t=p.logabsdet,t.logabsdet
    with torch.no_grad():{'input':a,'direction':e,'tangent':t,'primal':p}[target].detach().add_(1.)
    t.backward();assert a.grad is None
    np.testing.assert_allclose(e.grad.numpy(),[[3/5.88,-.4/5.88],[-.3/5.88,2/5.88]],rtol=3e-14,atol=3e-14)


@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
def test_complex_singular_logdet_and_real_nan_phase_match_native_forward_conventions(dtype):
    x=torch.tensor([[1.,0.],[0.,0.]],dtype=dtype);out=torch.slogdet(x)
    assert out.sign.item()==0 and np.isneginf(out.logabsdet.item())
    value=torch.logdet(x).item();assert np.isneginf(value.real) and np.isnan(value.imag)
    x=torch.tensor([[np.nan,0.],[0.,1.]],dtype=torch.float64);out=torch.slogdet(x)
    assert out.sign.item()==0 and np.isnan(out.logabsdet.item())


@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
@pytest.mark.parametrize('entry',[np.inf,np.nan])
def test_complex_nonfinite_logdet_checks_each_component_independently(dtype,entry):
    value=torch.logdet(torch.tensor([[entry,0.],[0.,1.]],dtype=dtype)).item()
    assert np.isnan(value.real) and np.isnan(value.imag)


def test_log_jvp_singular_ieee_results_do_not_change_numpy_error_policy():
    x=torch.tensor([[1.,0.],[0.,0.]],dtype=torch.float64);e=torch.ones_like(x)
    before=np.seterr(all='raise')
    try:
        _,t=torch.func.jvp(torch.logdet,(x,),(e,));assert np.isnan(t.item())
        assert np.geterr()==dict(divide='raise',over='raise',under='raise',invalid='raise')
    finally:np.seterr(**before)


@pytest.mark.parametrize('first',['sign','logabsdet'])
@pytest.mark.parametrize('second',['sign','logabsdet'])
def test_slogdet_outputs_share_release_state(first,second):
    x=torch.eye(2,dtype=torch.float64).requires_grad_();out=torch.slogdet(x)
    getattr(out,first).backward()
    with pytest.raises(RuntimeError):getattr(out,second).backward()


@pytest.mark.parametrize('first',['sign','logabsdet'])
def test_slogdet_shared_state_survives_retained_backward(first):
    x=torch.eye(2,dtype=torch.float64).requires_grad_();out=torch.slogdet(x)
    getattr(out,first).backward(retain_graph=True);out.logabsdet.backward()
    np.testing.assert_array_equal(x.grad.numpy(),np.eye(2)*(1 if first=='sign' else 2))


def test_sign_only_backward_releases_shared_storage_and_saved_ancestors():
    a=torch.eye(8,dtype=torch.float64).requires_grad_();x=a*2;storage=weakref.ref(x._data);out=torch.slogdet(x);pending=x.sum()
    del x;gc.collect();assert storage() is not None
    out.sign.backward();gc.collect();assert storage() is None and a.grad is None
    with pytest.raises(RuntimeError):out.logabsdet.backward()
    with pytest.raises(RuntimeError):pending.backward()


def test_regular_det_reverse_does_not_factor_an_svd(monkeypatch):
    x=torch.tensor([[2.,.3],[.4,3.]],dtype=torch.float64,requires_grad=True);value=torch.det(x)
    def forbidden(*args,**kwargs):raise AssertionError('SVD is unnecessary for this invertible determinant')
    monkeypatch.setattr(np.linalg,'svd',forbidden);value.backward()
    np.testing.assert_allclose(x.grad.numpy(),[[3.,-.4],[-.3,2.]],rtol=3e-14,atol=3e-14)


@pytest.mark.parametrize('data',[[[1.,0.],[0.,0.]],[[1.,2.],[2.,4.]],[[0.,0.],[0.,0.]]])
def test_singular_det_retains_existing_polynomial_cofactor_gradient(data):
    a=np.asarray(data);x=torch.tensor(a,requires_grad=True);x.det().backward()
    np.testing.assert_allclose(x.grad.numpy(),[[a[1,1],-a[1,0]],[-a[0,1],a[0,0]]],atol=3e-14,rtol=3e-14)


@pytest.mark.parametrize('mode',['grad','backward_inputs'])
def test_unused_edges_respect_nonleaf_grad_frontiers(mode):
    a=torch.eye(2,dtype=torch.float64).requires_grad_();x=a*2;sign=torch.slogdet(x).sign
    if mode=='grad':
        assert torch.autograd.grad(sign,x,allow_unused=True)==(None,)
        x.sum().backward();np.testing.assert_array_equal(a.grad.numpy(),np.full((2,2),2.))
    else:
        sign.backward(inputs=(x,));assert x.grad is None and a.grad is None
        with pytest.raises(RuntimeError):x.sum().backward()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('shape',[(0,3,3),(2,0,3,3)])
def test_empty_log_outputs_and_directional_outputs_have_canonical_strides(dtype,shape):
    x=torch.ones(*shape,dtype=dtype);p,t=torch.func.jvp(torch.slogdet,(x,),(torch.ones_like(x),))
    expected=(1,)*(len(shape)-2)
    for value in [p.sign,p.logabsdet,t.sign,t.logabsdet,torch.logdet(x)]:assert value.stride()==expected


@pytest.mark.parametrize('shape',[(2,2),(2,2,2)])
def test_log_vjp_and_reverse_over_jvp_follow_native_column_major_layout(shape):
    a=torch.eye(2,dtype=torch.float64).expand(*shape).clone().requires_grad_();e=torch.ones_like(a,requires_grad=True)
    g=torch.autograd.grad(a.logdet().sum(),a)[0];_,d=torch.func.jvp(torch.logdet,(a,),(e,));ge=torch.autograd.grad(d.sum(),e)[0]
    expected=(1,2) if len(shape)==2 else (4,1,2)
    assert g.stride()==ge.stride()==expected
