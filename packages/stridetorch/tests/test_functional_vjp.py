"""Functional derivative contracts: independent analytic and finite-difference oracles."""
import numpy as np
import pytest
from stridetorch import torch, as_torch, UnsupportedOperation

@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('requires_grad',[False,True])
@pytest.mark.parametrize('strided',[False,True])
def test_multi_output_broadcast_vjp_preserves_callers(dtype,enabled,requires_grad,strided):
    data=np.sin(np.arange(6).reshape(2,3)*.37).astype(dtype);bias=np.array([.2,-.3,.7],dtype=dtype)
    x=torch.tensor(data.T if strided else data,dtype=getattr(torch,dtype),requires_grad=requires_grad)
    if strided:x=x.T
    b=torch.tensor(bias,requires_grad=requires_grad);x.grad=torch.full_like(x,19);b.grad=torch.full_like(b,23)
    sx=torch.tensor([.4,-.7],dtype=getattr(torch,dtype));sb=torch.tensor([.3,.8,-.2],dtype=getattr(torch,dtype));calls=[];hooks=[]
    if requires_grad:x.register_hook(lambda g:hooks.append('x'));b.register_hook(lambda g:hooks.append('b'))
    def function(a,c):
        calls.append(torch.is_grad_enabled());return (a*c).sin().add(.2*a.square()).sum(1),a.mean(0)+c.exp()
    with torch.set_grad_enabled(enabled):
        for scale in [1.,-2.]:
            output,grad=torch.autograd.functional.vjp(function,(x,b),(sx*scale,sb*scale))
            assert torch.is_grad_enabled()==enabled
            expected_x=scale*(sx.numpy()[:,None]*(np.cos(data*bias)*bias+.4*data)+sb.numpy()[None,:]/2)
            expected_b=scale*((sx.numpy()[:,None]*np.cos(data*bias)*data).sum(0)+sb.numpy()*np.exp(bias))
            for actual,expected in [(grad[0],expected_x),(grad[1],expected_b),(output[0],(np.sin(data*bias)+.2*data**2).sum(1)),(output[1],data.mean(0)+np.exp(bias))]:
                np.testing.assert_allclose(actual.numpy(),expected,rtol=3e-5 if dtype=='float32' else 3e-12,atol=3e-6 if dtype=='float32' else 3e-12)
                assert not actual.requires_grad
    assert calls==[True,True] and hooks==[]
    np.testing.assert_array_equal(x.grad.numpy(),19);np.testing.assert_array_equal(b.grad.numpy(),23)
    np.testing.assert_array_equal(x.numpy(),data);np.testing.assert_array_equal(b.numpy(),bias)

@pytest.mark.parametrize('strict',[False,True])
def test_vjp_matches_independent_finite_differences(strict):
    data=np.array([[.2,-.5],[.8,.3]],dtype=np.float64);bias=np.array([.4,-.7]);seed=np.array([[.6,-.2],[.1,.9]])
    def numpy_function(a,b):return np.tanh(a@b)+np.exp(a*.13)
    def function(a,b):return (a@b).tanh()+(.13*a).exp()
    a=torch.tensor(data);b=torch.tensor(bias);out,got=torch.autograd.functional.vjp(function,(a,b),torch.tensor(seed),strict=strict)
    for i,original in enumerate([data,bias]):
        expected=np.empty_like(original)
        for index in np.ndindex(original.shape):
            plus=[data.copy(),bias.copy()];minus=[data.copy(),bias.copy()];plus[i][index]+=1e-6;minus[i][index]-=1e-6
            expected[index]=np.sum((numpy_function(*plus)-numpy_function(*minus))*seed)/(2e-6)
        np.testing.assert_allclose(got[i].numpy(),expected,rtol=1e-6,atol=2e-9)
    np.testing.assert_allclose(out.numpy(),numpy_function(data,bias),rtol=1e-12,atol=1e-12)


def test_duplicate_arguments_and_captured_original_are_distinct():
    x=torch.tensor([2.,3.],requires_grad=True);x.grad=torch.tensor([17.,19.]);calls=[]
    out,grads=torch.autograd.functional.vjp(lambda a,b:(calls.append(1) or a*2+b*3),(x,x),torch.ones(2))
    assert calls==[1];np.testing.assert_array_equal(out.numpy(),[10,15]);np.testing.assert_array_equal(grads[0].numpy(),[2,2]);np.testing.assert_array_equal(grads[1].numpy(),[3,3])
    out,gradient=torch.autograd.functional.vjp(lambda a:a*x,x,torch.ones(2));np.testing.assert_array_equal(gradient.numpy(),[2,3]);np.testing.assert_array_equal(x.grad.numpy(),[17,19])

@pytest.mark.parametrize('kind',['unused','constant','mixed_constant','integer'])
def test_disconnected_inputs_materialize_zero_without_strict(kind):
    x=torch.tensor([2.,3.]);y=torch.tensor([4.,5.]);functions=dict(unused=lambda a,b:a*2,constant=lambda a,b:torch.ones(2),mixed_constant=lambda a,b:(a*2,torch.ones(2)),integer=lambda a,b:torch.ones(2,dtype=torch.int64))
    seeds=(torch.ones(2),torch.ones(2)) if kind=='mixed_constant' else torch.ones(2)
    out,grads=torch.autograd.functional.vjp(functions[kind],(x,y),seeds)
    np.testing.assert_array_equal(grads[0].numpy(),[2,2] if kind in ['unused','mixed_constant'] else [0,0]);np.testing.assert_array_equal(grads[1].numpy(),[0,0]);assert all(not g.requires_grad for g in grads)
    with pytest.raises(RuntimeError):torch.autograd.functional.vjp(functions[kind],(x,y),seeds,strict=True)

@pytest.mark.parametrize('tuple_input',[False,True])
@pytest.mark.parametrize('tuple_output',[False,True])
def test_single_element_implicit_seed_and_container_structure(tuple_input,tuple_output):
    x=torch.tensor([2.],requires_grad=True)
    def function(a):
        y=a.square();return (y,) if tuple_output else y
    out,g=torch.autograd.functional.vjp(function,(x,) if tuple_input else x)
    assert isinstance(out,tuple)==tuple_output and isinstance(g,tuple)==tuple_input
    actual_out=out[0] if tuple_output else out;actual_g=g[0] if tuple_input else g
    np.testing.assert_array_equal(actual_out.numpy(),[4]);np.testing.assert_array_equal(actual_g.numpy(),[4]);assert not actual_out.requires_grad and not actual_g.requires_grad


def test_identity_keeps_output_storage_and_does_not_alias_seed_copy_contract():
    x=torch.tensor([2.,3.],requires_grad=True);seed=torch.tensor([.3,.7],requires_grad=True)
    out,g=torch.autograd.functional.vjp(lambda a:a,x,seed)
    assert np.shares_memory(out.numpy(),x.numpy()) and not out.requires_grad and not g.requires_grad
    np.testing.assert_array_equal(g.numpy(),seed.numpy());assert x.grad is None and seed.grad is None

@pytest.mark.parametrize('case',['input_list','output_list','seed_list','seed_shape','seed_count','nonscalar_implicit'])
def test_argument_validation_restores_outer_gradient_mode(case):
    x=torch.ones(2);cases=dict(input_list=lambda:torch.autograd.functional.vjp(lambda a:a,[x],x),output_list=lambda:torch.autograd.functional.vjp(lambda a:[a],x,x),seed_list=lambda:torch.autograd.functional.vjp(lambda a:a,x,[x]),seed_shape=lambda:torch.autograd.functional.vjp(lambda a:a,x,torch.ones(1)),seed_count=lambda:torch.autograd.functional.vjp(lambda a:(a,a),x,(x,)),nonscalar_implicit=lambda:torch.autograd.functional.vjp(lambda a:a,x))
    with torch.no_grad():
        with pytest.raises((TypeError,RuntimeError)):cases[case]()
        assert not torch.is_grad_enabled()


def test_user_exception_and_unsupported_higher_order_are_explicit():
    calls=[]
    def function(a):calls.append(1);raise ValueError('user exception')
    with torch.no_grad():
        with pytest.raises(ValueError,match='user exception'):torch.autograd.functional.vjp(function,torch.ones(1))
        assert not torch.is_grad_enabled()
        with pytest.raises(UnsupportedOperation,match='Higher-order'):torch.autograd.functional.vjp(function,torch.ones(1),create_graph=True)
        assert not torch.is_grad_enabled()
    assert calls==[1]


def test_supported_import_alias_restores_existing_modules():
    import sys,types
    previous={name:module for name,module in sys.modules.items() if name=='torch' or name.startswith('torch.')}
    with as_torch() as t:
        from torch.autograd.functional import vjp
        assert vjp is t.autograd.functional.vjp
        out,g=vjp(lambda a:a.square().sum(),t.tensor([2.,3.]));assert g.tolist()==[4,6]
    actual={name:module for name,module in sys.modules.items() if name=='torch' or name.startswith('torch.')};assert actual==previous
