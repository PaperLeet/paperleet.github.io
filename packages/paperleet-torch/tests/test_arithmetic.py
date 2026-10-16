"""NumPy-only arithmetic workflows; also run unchanged under pinned Pyodide."""
import warnings

import numpy as np
import pytest

from paperleet_torch import torch, as_torch, UnsupportedOperation


def test_dtype_scalar_category_complex_and_true_division_contracts():
    a=torch.tensor([1,2],dtype=torch.int8);b=torch.tensor([3,4],dtype=torch.uint8)
    assert (a+b).dtype==torch.int16
    assert (a/b).dtype==torch.float32
    np.testing.assert_allclose((a/b).numpy(),np.array([1/3,.5],dtype=np.float32))
    assert (a+128).tolist()==[-127,-126]
    assert (torch.tensor([1.],dtype=torch.float32)+torch.tensor(2.,dtype=torch.float64)).dtype==torch.float32
    assert torch.add(2,torch.tensor(3.,dtype=torch.float16)).dtype==torch.float16
    z=torch.add(torch.tensor([1+.5j],dtype=torch.complex64),torch.tensor([2.],dtype=torch.float64))
    assert z.dtype==torch.complex128 and z.item()==3+.5j
    with pytest.raises(UnsupportedOperation):torch.tensor([1.],dtype=torch.float16)+1j
    with pytest.raises(UnsupportedOperation):torch.ones(2)-True


def test_function_method_aliases_alpha_and_exact_large_rounded_integer_values():
    a,b=torch.tensor([1.,2.]),torch.tensor([3.,4.])
    assert torch.add(input=a,other=b,alpha=.5,out=None).tolist()==[2.5,4.]
    assert torch.subtract(a,b,alpha=2).tolist()==a.sub(b,alpha=2).tolist()
    assert torch.multiply(a,b).tolist()==a.mul(b).tolist()
    assert torch.divide(a,b).tolist()==a.div(b).tolist()
    assert torch.true_divide(a,b).tolist()==a.true_divide(b).tolist()
    values=[2**62+3,-2**62-3,2**63-1,-2**63+1]
    x=torch.tensor(values)
    for mode in ['floor','trunc']:
        expected=[v//3 if mode=='floor' else (abs(v)//3)*(1 if v>=0 else -1) for v in values]
        assert torch.div(x,3,rounding_mode=mode).tolist()==expected


def test_broadcast_arithmetic_gradients_against_numpy_derivatives():
    raw=np.array([[1.,2.,3.],[4.,5.,6.]],dtype=np.float64)
    rhs=np.array([.5,1.5,2.5],dtype=np.float64)
    a=torch.tensor(raw,requires_grad=True);b=torch.tensor(rhs,requires_grad=True)
    z=torch.div(torch.add(a,b,alpha=.125),torch.mul(b,.5))
    z.square().mean().backward()
    expected=2*raw/rhs+.25;seed=2*expected/raw.size
    np.testing.assert_allclose(z.numpy(),expected,rtol=1e-13)
    np.testing.assert_allclose(a.grad.numpy(),seed*2/rhs,rtol=1e-13)
    np.testing.assert_allclose(b.grad.numpy(),(seed*(-2*raw/rhs**2)).sum(0),rtol=1e-13)


def test_jvp_and_reverse_over_forward_against_independent_numpy_expressions():
    raw=np.array([[1.,2.,3.],[4.,5.,6.]],dtype=np.float64)
    rhs=np.array([.5,1.5,2.5],dtype=np.float64)
    vx=np.full_like(raw,.125);vy=np.array([.25,.5,.75],dtype=np.float64)
    a,b,da,db=[torch.tensor(v,requires_grad=True) for v in [raw,rhs,vx,vy]]
    function=lambda x,y:torch.div(torch.add(x,y,alpha=.125),torch.mul(y,.5))
    p,t=torch.func.jvp(function,(a,b),(da,db))
    (p.square().mean()+t.square().mean()).backward()
    z=2*raw/rhs+.25;tz=2*vx/rhs-2*raw*vy/rhs**2;scale=2/raw.size
    np.testing.assert_allclose(p.numpy(),z,rtol=1e-13);np.testing.assert_allclose(t.numpy(),tz,rtol=1e-13)
    expected=[scale*(z*2/rhs-tz*2*vy/rhs**2),
              (scale*(-z*2*raw/rhs**2+tz*(-2*vx/rhs**2+4*raw*vy/rhs**3))).sum(0),
              scale*tz*2/rhs,(scale*tz*(-2*raw/rhs**2)).sum(0)]
    for tensor,gradient in zip([a,b,da,db],expected):np.testing.assert_allclose(tensor.grad.numpy(),gradient,rtol=1e-13,atol=1e-13)


def test_unchanged_torch_imports_train_two_layers_against_manual_numpy_updates():
    with as_torch():
        import torch as lib
        raw=np.array([[1.,-.5],[.25,2.],[-1.,1.]],dtype=np.float64)
        target=np.array([[.5],[1.],[-.25]],dtype=np.float64)
        w=np.array([[.2,-.1,.3],[.4,.25,-.2]],dtype=np.float64)
        v=np.array([[.1],[-.2],[.3]],dtype=np.float64);bias=np.array([.2,-.1,.1],dtype=np.float64)
        parameters=[lib.nn.Parameter(value) for value in [w.copy(),v.copy(),bias.copy()]]
        x,y=lib.tensor(raw),lib.tensor(target);optimizer=lib.optim.SGD(parameters,lr=.025)
        for _ in range(4):
            hidden=np.tanh(raw@w+.25*bias);pred=hidden@v;dp=2*(pred-target)/target.size
            dh=(dp@v.T)*(1-hidden**2);gradients=[raw.T@dh,hidden.T@dp,.25*dh.sum(0)]
            optimizer.zero_grad()
            hp=lib.add(x@parameters[0],parameters[2],alpha=.25).tanh()
            loss=lib.div(lib.sub(hp@parameters[1],y).square().sum(),target.size)
            loss.backward()
            for parameter,gradient in zip(parameters,gradients):np.testing.assert_allclose(parameter.grad.numpy(),gradient,rtol=1e-12,atol=1e-14)
            optimizer.step();w-=.025*gradients[0];v-=.025*gradients[1];bias-=.025*gradients[2]
        for actual,expected in zip(parameters,[w,v,bias]):np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-12,atol=1e-14)


def test_arithmetic_overflow_and_errors_restore_numpy_error_configuration():
    previous=np.seterr(all='raise')
    try:
        a=torch.tensor([1.,2.]);b=torch.tensor([0.,0.])
        assert np.isinf(torch.div(a,b).numpy()).all()
        assert np.isinf((a+1e300).numpy()).all()
        assert np.isinf(torch.mul(a,1e300).numpy()).all()
        with pytest.raises(RuntimeError):torch.add(a,a,alpha=1e300)
        with pytest.raises(RuntimeError):torch.add(torch.ones(2,dtype=torch.float16),1,alpha=1e10)
        with pytest.raises(OverflowError):a+2**64
        with pytest.raises(RuntimeError):torch.div(torch.tensor([1]),0,rounding_mode='trunc')
        assert np.geterr()==dict(divide='raise',over='raise',under='raise',invalid='raise')
    finally:np.seterr(**previous)


def test_linear_arithmetic_repeated_backward_and_zero_alpha_paths():
    a=torch.tensor([1.,2.],requires_grad=True);b=torch.tensor([3.,4.],requires_grad=True)
    value=torch.sub(a,b,alpha=0)
    value.sum().backward();value.sum().backward()
    assert a.grad.tolist()==[2.,2.] and b.grad.tolist()==[0.,0.]
    with pytest.raises(UnsupportedOperation):torch.add(a,b,out=torch.zeros(2))
    with pytest.raises(TypeError):torch.mul(a,b,alpha=1)
    with pytest.raises(RuntimeError):torch.div(a,b,rounding_mode='invalid')


def test_rounded_vjp_jvp_and_number_protocols_are_consistent():
    for mode in ['floor','trunc']:
        a=torch.tensor([-2.5,1.25,3.75],requires_grad=True);b=torch.tensor(1.5,requires_grad=True)
        result=torch.div(a,b,rounding_mode=mode);result.sum().backward()
        assert a.grad.tolist()==[0.,0.,0.] and b.grad.item()==0
        p,t=torch.func.jvp(lambda x:torch.div(x,b,rounding_mode=mode),(a,),(torch.ones_like(a),))
        np.testing.assert_array_equal(p.numpy(),result.numpy());assert t.tolist()==[0.,0.,0.]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a=torch.tensor([1.,2.]);p,t=torch.func.jvp(lambda x:torch.mul(x,np.complex64(2+.5j)),(a,),(torch.ones_like(a),))
    assert p.tolist()==[2.,4.] and t.tolist()==[2.,2.]
    with pytest.raises(UnsupportedOperation):torch.func.jvp(lambda x:torch.sub(x,True),(a,),(torch.ones_like(a),))


def test_rounded_pullbacks_release_operand_storage_and_complex_ad_is_explicit():
    import gc
    import weakref
    a=torch.tensor([1.25,2.75],requires_grad=True);b=torch.tensor(1.5,requires_grad=True)
    left,right=a*.5,b*.5
    references=[weakref.ref(left._data),weakref.ref(right._data)]
    loss=torch.div(left,right,rounding_mode='floor').sum()
    del left,right;gc.collect()
    assert all(ref() is None for ref in references)
    loss.backward();assert a.grad.tolist()==[0.,0.] and b.grad.item()==0
    with pytest.raises(UnsupportedOperation):a+1j
    with torch.no_grad():assert (a+1j).dtype==torch.complex64
    scalar=1.+2.**-40
    x=torch.tensor([1.,2.],dtype=torch.float64)
    p,t=torch.func.jvp(lambda z:torch.mul(z,scalar),(x,),(torch.ones_like(x),))
    np.testing.assert_array_equal(p.numpy(),np.array([1.,2.])*scalar)
    np.testing.assert_array_equal(t.numpy(),np.full(2,scalar))
