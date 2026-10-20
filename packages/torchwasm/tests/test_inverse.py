"""Inverse aliases, analytic 2x2 derivatives and native forward-AD contracts."""
import gc
import warnings
import weakref

import numpy as np
import pytest

from torchwasm import torch, UnsupportedOperation


def tensor_layout(data,layout,requires=False):
    if layout=='F':value=torch.from_numpy(data.swapaxes(-1,-2).copy()).transpose(-1,-2)
    elif layout=='S':
        backing=np.zeros((*data.shape[:-2],data.shape[-2]*2,data.shape[-1]*2),dtype=data.dtype)
        backing[...,::2,::2]=data;value=torch.from_numpy(backing)[...,::2,::2]
    else:value=torch.from_numpy(data.copy())
    return value.detach().requires_grad_(requires)


def inverse2(a):
    a,b,c,d=map(complex,a.ravel());det=a*d-b*c
    return np.array([[d,-b],[-c,a]])/det


def product2(a,b):
    return np.array([[sum(a[i,k]*b[k,j] for k in range(2)) for j in range(2)] for i in range(2)])


def inverse_jacobians2(a,e,weight):
    p=inverse2(a).real;value=-product2(product2(p,e),p);ga=np.empty((2,2));ge=np.empty((2,2));gp=np.empty((2,2))
    for i in range(2):
        for j in range(2):
            dp=np.array([[-p[k,i]*p[j,l] for l in range(2)] for k in range(2)])
            change=-product2(product2(dp,e),p)-product2(product2(p,e),dp)
            unit=np.zeros((2,2));unit[i,j]=1
            ga[i,j]=float((weight*change).sum());ge[i,j]=float((weight*(-product2(product2(p,unit),p))).sum());gp[i,j]=float((weight*dp).sum())
    return p,value,ga,ge,gp


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.complex64,np.complex128])
@pytest.mark.parametrize('layout',['C','F','S'])
@pytest.mark.parametrize('alias',['top','linalg','A','legacy_input','method'])
def test_inverse_aliases_match_independent_2x2_formula_and_native_layout(dtype,layout,alias):
    data=np.array([[2.,.3],[.4,3.]],dtype=dtype)
    if np.dtype(dtype).kind=='c':data+=np.array([[.2,-.1],[.3,-.2]],dtype=dtype)*np.asarray(1j,dtype=dtype)
    x=tensor_layout(data,layout)
    value=torch.inverse(x) if alias=='top' else torch.linalg.inv(x) if alias=='linalg' else torch.linalg.inv(A=x) if alias=='A' else torch.linalg.inv(input=x) if alias=='legacy_input' else x.inverse()
    expected=inverse2(data);expected=expected if np.dtype(dtype).kind=='c' else expected.real
    np.testing.assert_allclose(value.numpy(),expected,rtol=3e-6,atol=3e-8)
    assert value.dtype==np.dtype(dtype) and value.stride()==(1,2)
    assert not np.shares_memory(value.numpy(),x.numpy())


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['C','F','S'])
@pytest.mark.parametrize('needs',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('alias',['top','linalg','method'])
def test_inverse_jvp_and_both_reverse_paths_match_scalar_jacobian(dtype,layout,needs,alias):
    data=np.array([[2.,.3],[.4,3.]],dtype=dtype);direction=np.array([[.2,.4],[-.1,.3]],dtype=dtype);weight=np.array([[.7,-.2],[.1,.9]],dtype=dtype)
    a=tensor_layout(data,layout,needs[0]);e=tensor_layout(direction,layout,needs[1]);fn=torch.inverse if alias=='top' else torch.linalg.inv if alias=='linalg' else lambda x:x.inverse()
    p,t=torch.func.jvp(fn,(a,),(e,),strict=True);expected=inverse_jacobians2(data.astype('float64'),direction.astype('float64'),weight.astype('float64'))
    np.testing.assert_allclose(p.numpy(),expected[0],rtol=3e-6,atol=3e-8);np.testing.assert_allclose(t.numpy(),expected[1],rtol=3e-6,atol=3e-8)
    assert p.stride()==t.stride()==(1,2)
    active=tuple(x for x,flag in zip((a,e),needs) if flag);gradients=iter(torch.autograd.grad(t,active,torch.tensor(weight),retain_graph=True))
    if needs[0]:np.testing.assert_allclose(next(gradients).numpy(),expected[2],rtol=3e-6,atol=3e-8)
    if needs[1]:np.testing.assert_allclose(next(gradients).numpy(),expected[3],rtol=3e-6,atol=3e-8)
    if needs[0]:np.testing.assert_allclose(torch.autograd.grad(p,a,torch.tensor(weight))[0].numpy(),expected[4],rtol=3e-6,atol=3e-8)


@pytest.mark.parametrize('layout_pair',[('C','C'),('F','C'),('S','S')])
@pytest.mark.parametrize('needs',[(True,True),(True,False),(False,True)])
@pytest.mark.parametrize('target',['input','direction','primal','tangent'])
@pytest.mark.parametrize('channel',['primal','tangent'])
def test_inverse_jvp_mutations_follow_saved_inverse_and_direction_copy(layout_pair,needs,target,channel):
    data=np.array([[2.,.3],[.4,3.]]);direction=np.array([[.2,.4],[-.1,.3]])
    a=tensor_layout(data,layout_pair[0],needs[0]);e=tensor_layout(direction,layout_pair[1],needs[1]);same=a.stride()==e.stride()
    p,t=torch.func.jvp(torch.inverse,(a,),(e,));out=p if channel=='primal' else t
    with torch.no_grad():{'input':a,'direction':e,'primal':p,'tangent':t}[target].detach().add_(1)
    if not out.requires_grad:return
    guarded=target=='primal' or channel=='tangent' and target=='direction' and needs[0] and same
    if guarded:
        with pytest.raises(RuntimeError,match='modified'):out.sum().backward()
    else:
        out.sum().backward();expected=inverse_jacobians2(data,direction,np.ones((2,2)))
        if needs[0]:np.testing.assert_allclose(a.grad.numpy(),expected[4] if channel=='primal' else expected[2],rtol=3e-12,atol=3e-14)
        if needs[1] and channel=='tangent':np.testing.assert_allclose(e.grad.numpy(),expected[3],rtol=3e-12,atol=3e-14)


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.complex64,np.complex128])
@pytest.mark.parametrize('shape',[(0,0),(2,0,0),(0,2,2),(2,0,1,1)])
def test_empty_inverse_has_native_final_column_major_axes(dtype,shape):
    x=torch.ones(shape,dtype=dtype);value=x.inverse()
    assert value.shape==shape and value.dtype==np.dtype(dtype)
    expected=(1,1) if len(shape)==2 else (1,)*(len(shape)) if shape[-1] in (0,1) else (4,1,2)
    assert value.stride()==expected


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(0,0),(2,0,0),(0,2,2),(2,0,1,1)])
def test_empty_inverse_jvp_and_raw_gradients_have_canonical_geometry(dtype,shape):
    a=torch.ones(shape,dtype=dtype,requires_grad=True);e=torch.ones_like(a,requires_grad=True);p,t=torch.func.jvp(torch.inverse,(a,),(e,));ga,ge=torch.autograd.grad(t,(a,e),torch.ones_like(t))
    assert t.shape==ga.shape==ge.shape==shape
    expected=(4,2,1) if shape==(0,2,2) else (1,)*len(shape)
    assert t.stride()==ga.stride()==ge.stride()==expected


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.complex64,np.complex128])
@pytest.mark.parametrize('which',['maximum','tiny','subnormal'])
def test_singleton_inverse_uses_dtype_reciprocal_without_lapack_overflow(dtype,which):
    d=np.dtype(dtype);real=np.dtype('float32' if d.char in 'fF' else 'float64');info=np.finfo(real)
    with np.errstate(all='ignore'):
        v=real.type(info.max) if which=='maximum' else real.type(info.tiny) if which=='tiny' else np.nextafter(real.type(0),real.type(1))
        data=np.array([[complex(v,v) if d.kind=='c' and which!='subnormal' else v]],dtype=d);expected=np.reciprocal(data)
    value=torch.inverse(torch.tensor(data)).numpy()
    np.testing.assert_allclose(value,expected,rtol=3e-6,atol=0,equal_nan=True)


@pytest.mark.parametrize('policy',['raise','warn','call'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_inverse_jvp_restores_numpy_policy_and_callback(policy,dtype):
    x=torch.tensor([[2.,.3],[.4,3.]],dtype=dtype,requires_grad=True);e=torch.ones(2,2,dtype=dtype,requires_grad=True);events=[];handler=lambda *args:events.append(args);before=np.geterr();old=np.geterrcall()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with np.errstate(all=policy,call=handler):
            p,t=torch.func.jvp(torch.inverse,(x,),(e,));t.sum().backward()
            assert np.geterr()==dict(divide=policy,over=policy,under=policy,invalid=policy) and np.geterrcall() is handler
    assert not caught and not events and np.geterr()==before and np.geterrcall() is old


@pytest.mark.parametrize('needs',[(True,True),(True,False),(False,True)])
def test_inverse_returned_primal_hook_is_separate_from_tangent_branch(needs):
    a=torch.eye(2,dtype=torch.float64).requires_grad_(needs[0]);e=torch.ones(2,2,dtype=torch.float64,requires_grad=needs[1]);p,t=torch.func.jvp(torch.inverse,(a,),(e,));seen=[]
    if p.requires_grad:p.retain_grad();p.register_hook(lambda g:seen.append(g.detach().clone()))
    t.sum().backward(retain_graph=True);assert p.grad is None and seen==[]
    if p.requires_grad:p.sum().backward();assert len(seen)==1;np.testing.assert_array_equal(p.grad.numpy(),1.)


def test_inverse_does_not_retain_original_nonleaf_tensor_or_its_array():
    leaf=torch.eye(3,dtype=torch.float64).requires_grad_();a=leaf*2;e=torch.ones(3,3,dtype=torch.float64,requires_grad=True);owner=weakref.ref(a);storage=weakref.ref(a._data);p,t=torch.func.jvp(torch.inverse,(a,),(e,));del a;gc.collect()
    assert owner() is None and storage() is None
    (p+t).sum().backward();assert leaf.grad is not None and e.grad is not None


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_inverse_jvp_with_same_primal_and_direction_accumulates_both_paths(dtype):
    data=np.array([[2.,.3],[.4,3.]],dtype=dtype);a=torch.tensor(data,requires_grad=True);p,t=torch.func.jvp(torch.inverse,(a,),(a,));t.sum().backward();expected=inverse_jacobians2(data.astype('float64'),data.astype('float64'),np.ones((2,2)))
    np.testing.assert_allclose(a.grad.numpy(),expected[2]+expected[3],rtol=3e-6,atol=3e-8)


def test_inverse_rejects_unsupported_or_ambiguous_calls():
    a=torch.eye(2)
    for call in [lambda:torch.linalg.inv(),lambda:torch.linalg.inv(A=a,input=a),lambda:torch.inverse([[1.]]),lambda:torch.linalg.inv(typo=a)]:
        with pytest.raises(TypeError):call()
    with pytest.raises(UnsupportedOperation):torch.inverse(a,out=a)
    with pytest.raises(UnsupportedOperation):torch.tensor([[2.+.1j]],dtype=torch.complex128,requires_grad=True)
    complex_a=torch.tensor([[2.+.1j]],dtype=torch.complex128)
    with torch.no_grad():assert not torch.inverse(complex_a).requires_grad
    with pytest.raises(UnsupportedOperation):torch.func.jvp(torch.inverse,(complex_a,),(torch.ones_like(complex_a),))
