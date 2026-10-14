"""Native differential checks for exact forward AD and reverse-over-forward graphs."""
import subprocess
import sys

import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


def compare(actual, expected, dtype):
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), equal_nan=True,
        atol=3e-5 if dtype == np.float32 else 2e-11, rtol=3e-5 if dtype == np.float32 else 2e-11)


BINARY = [lambda x,y:x+y, lambda x,y:y-x, lambda x,y:x*y, lambda x,y:y*x,
          lambda x,y:x/y, lambda x,y:y/x, lambda x,y:(x-y).square(), lambda x,y:x**-.5+y**0]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('function', BINARY)
def test_broadcast_binary_primal_tangent_and_both_reverse_gradients(dtype, function):
    rng = np.random.RandomState(137)
    data = [rng.rand(2, 3).astype(dtype)+1, rng.rand(3).astype(dtype)+1,
            rng.randn(2, 3).astype(dtype), rng.randn(3).astype(dtype)]
    a = [wasm.Tensor(v, requires_grad=True) for v in data]
    b = [torch.tensor(v, requires_grad=True) for v in data]
    actual = wasm.jvp(function, tuple(a[:2]), tuple(a[2:]))
    expected = torch.func.jvp(function, tuple(b[:2]), tuple(b[2:]))
    for x, y in zip(actual, expected): compare(x, y, dtype)
    (actual[0].square().sum()+actual[1].square().sum()).backward()
    (expected[0].square().sum()+expected[1].square().sum()).backward()
    for x, y in zip(a, b):
        if y.grad is None: assert x.grad is None or not x.grad.any()
        else: compare(x.grad, y.grad, dtype)


UNARY = [lambda x:x.exp(), lambda x:x.log(), lambda x:x.sin(), lambda x:x.cos(), lambda x:x.tanh(),
         lambda x:x.sigmoid(), lambda x:x.relu(), lambda x:x.softmax(-1), lambda x:x.log_softmax(-1)]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('function', UNARY)
def test_nonlinear_jvp_and_reverse_over_forward_match_native(dtype, function):
    rng = np.random.RandomState(139)
    data = rng.rand(2, 3).astype(dtype)+.1
    tangent = rng.randn(2, 3).astype(dtype)
    a, da = wasm.Tensor(data, requires_grad=True), wasm.Tensor(tangent, requires_grad=True)
    b, db = torch.tensor(data, requires_grad=True), torch.tensor(tangent, requires_grad=True)
    actual, expected = wasm.jvp(function, (a,), (da,)), torch.func.jvp(function, (b,), (db,))
    for x,y in zip(actual, expected): compare(x,y,dtype)
    (actual[0].sum()+actual[1].sum()).backward(); (expected[0].sum()+expected[1].sum()).backward()
    compare(a.grad,b.grad,dtype); compare(da.grad,db.grad,dtype)


@pytest.mark.parametrize('function', [lambda x:x.reshape(3,2).T.flatten(), lambda x:x.unsqueeze(0).expand(4,-1,-1).mean(0),
    lambda x:x[:,1:].sum(-1), lambda x:x.flip((0,)).repeat(2,1), lambda x:x.diagonal(), lambda x:x.clone().double()])
def test_linear_views_reductions_and_strides_keep_tangents(function):
    data = np.arange(6.,dtype=np.float64).reshape(3,2).T
    tangent = data*.1+.3
    a, b = wasm.jvp(function,(wasm.Tensor(data),),(wasm.Tensor(tangent),)), torch.func.jvp(function,(torch.tensor(data),),(torch.tensor(tangent),))
    for x,y in zip(a,b):compare(x,y,np.float64)


@pytest.mark.parametrize('left,right', [((3,), (3,)), ((2,3), (3,4)), ((2,1,3,4),(1,5,4,2))])
def test_matmul_forward_and_reverse_gradients_for_vectors_and_batches(left,right):
    rng=np.random.RandomState(149)
    data=[rng.randn(*shape) for shape in (left,right,left,right)]
    a=[wasm.Tensor(v,requires_grad=True) for v in data]; b=[torch.tensor(v,requires_grad=True) for v in data]
    fn=lambda x,y:x@y
    actual,expected=wasm.jvp(fn,tuple(a[:2]),tuple(a[2:])),torch.func.jvp(fn,tuple(b[:2]),tuple(b[2:]))
    for x,y in zip(actual,expected):compare(x,y,np.float64)
    (actual[0].sum()+actual[1].square().sum()).backward();(expected[0].sum()+expected[1].square().sum()).backward()
    for x,y in zip(a,b):compare(x.grad,y.grad,np.float64)


def test_linear_silu_network_and_closed_parameters_retain_reverse_gradients():
    rng=np.random.RandomState(151)
    a=wasm.Sequential(wasm.Linear(3,4,dtype=np.float64),wasm.SiLU(),wasm.Linear(4,2,dtype=np.float64))
    b=torch.nn.Sequential(torch.nn.Linear(3,4,dtype=torch.float64),torch.nn.SiLU(),torch.nn.Linear(4,2,dtype=torch.float64))
    with wasm.no_grad(),torch.no_grad():
        for ap,bp in zip(a.parameters(),b.parameters()):
            data=rng.randn(*ap.shape)*.3;ap[...] = wasm.Tensor(data);bp.copy_(torch.tensor(data))
    data,tangent=rng.randn(2,3),rng.randn(2,3)
    x,dx=wasm.Tensor(data,requires_grad=True),wasm.Tensor(tangent,requires_grad=True)
    y,dy=torch.tensor(data,requires_grad=True),torch.tensor(tangent,requires_grad=True)
    actual,expected=wasm.jvp(a,(x,),(dx,)),torch.func.jvp(b,(y,),(dy,))
    for v,w in zip(actual,expected):compare(v,w,np.float64)
    (actual[0].square().sum()+actual[1].square().sum()).backward();(expected[0].square().sum()+expected[1].square().sum()).backward()
    for v,w in [(x,y),(dx,dy),*zip(a.parameters(),b.parameters())]:compare(v.grad,w.grad,np.float64)


def test_structured_outputs_single_call_detach_strict_and_no_grad_semantics():
    x=wasm.Tensor(np.array([1.,2.]),requires_grad=True);dx=wasm.Tensor(np.array([.3,.4]),requires_grad=True)
    calls=[]
    def function(x):calls.append(1);return {'p':x.square(),'tail':[x[1:],x.detach()]}
    p,t=wasm.jvp(function,(x,),(dx,))
    assert calls==[1]
    np.testing.assert_allclose(t['p'].numpy(),[.6,1.6]);np.testing.assert_equal(t['tail'][1].numpy(),[0.,0.])
    with pytest.raises(RuntimeError):wasm.jvp(lambda x:x.detach(),(x,),(dx,),strict=True)
    with wasm.no_grad():
        p,t=wasm.jvp(lambda x:x.square(),(x,),(dx,))
        assert not p.requires_grad and not t.requires_grad
        np.testing.assert_allclose(t.numpy(),[.6,1.6])


def test_tangent_is_not_estimated_by_perturbing_floating_point_inputs():
    x=wasm.Tensor(np.array([2.**54]));dx=wasm.Tensor(np.array([1.]))
    # Adding a unit perturbation is unrepresentable here, but the exact JVP is 2.
    p,t=wasm.jvp(lambda x:x+x,(x,),(dx,))
    np.testing.assert_equal(t.numpy(),[2.]);assert p.item()==2.**55


def test_invalid_arguments_unsupported_rules_escaping_and_exception_cleanup():
    x=wasm.Tensor(np.eye(2));v=wasm.Tensor(np.ones((2,2)))
    for primals,tangents in [([],[]),((),()),((x,),()),((x,),(wasm.Tensor(np.ones(3)),))]:
        with pytest.raises((RuntimeError,wasm.BrowserUnsupported)):wasm.jvp(lambda x:x,primals,tangents)
    for fn in [lambda x:x.norm(),lambda x:x.numpy(),lambda x:x.requires_grad_(),
               lambda x:wasm.jvp(lambda y:y,(x,),(x,))[0]]:
        with pytest.raises(wasm.BrowserUnsupported):wasm.jvp(fn,(x,),(v,))
        assert not wasm._jvp_active
    def mutation(x):x[0]=0;return x
    with pytest.raises(wasm.BrowserUnsupported):wasm.jvp(mutation,(x,),(v,))
    np.testing.assert_equal(x.numpy(),np.eye(2))
    held=[]
    wasm.jvp(lambda x:(held.append(x) or x),(x,),(v,))
    with pytest.raises(wasm.BrowserUnsupported):held[0]+1
    with pytest.raises(wasm.BrowserUnsupported):wasm.jvp(lambda y:held[0]+y,(x,),(v,))
    p,t=wasm.jvp(lambda x:x*2,(x,),(v,));np.testing.assert_equal(t.numpy(),np.ones((2,2))*2)
    with pytest.raises(wasm.BrowserUnsupported):wasm.jvp(lambda x:(x,x),(x,),(v,),has_aux=True)


def test_public_api_guards_do_not_silently_drop_unknown_forward_derivatives():
    script='''from paperleet.browser import numpy_autograd as w
t=w.install();x=t.tensor([[1.,.2],[.2,2.]],dtype=t.float64);v=t.ones_like(x)
for fn in (t.nan_to_num,t.unique,t.linalg.eigvalsh,t.linalg.norm):
 try:t.func.jvp(fn,(x,),(v,))
 except w.BrowserUnsupported:pass
 else:raise AssertionError('An unsupported forward API looked like a constant')
 assert not w._jvp_active
 p,d=t.func.jvp(lambda z:t.cat((z,z),0),(x,),(v,))
 assert d.shape==(4,2) and (d==1).all()
p,d=t.func.jvp(lambda z:t.where(z>0,z.square(),-z),(x,),(v,));assert t.allclose(d,2*x)
'''
    out=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=15)
    assert out.returncode==0,out.stdout+out.stderr
