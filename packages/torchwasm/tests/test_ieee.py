"""Independent reciprocal, IEEE reverse kernels and callback-policy workflows."""
import warnings

import numpy as np
import pytest

from torchwasm import torch, UnsupportedOperation


@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
def test_complex_zero_reciprocal_has_stable_library_sign_policy(dtype):
    source=np.array([complex(real,imag) for real in [0.,-0.] for imag in [0.,-0.]],dtype=dtype)
    for method in [False,True]:
        x=torch.tensor(source)
        value=x.reciprocal() if method else torch.reciprocal(x)
        result=value.numpy()
        assert result.dtype==source.dtype and result.shape==source.shape
        assert np.isinf(result.real).all() and np.isnan(result.imag).all()
        np.testing.assert_array_equal(np.signbit(result.real),np.signbit(source.real))


def test_reciprocal_real_integer_complex_values_and_signed_zero():
    x=torch.tensor([0.,-0.,2.,-4.,np.inf,-np.inf,np.nan],dtype=torch.float64)
    y=torch.reciprocal(x).numpy()
    expected=np.array([np.inf,-np.inf,.5,-.25,0.,-0.,np.nan])
    np.testing.assert_array_equal(y,expected)
    np.testing.assert_array_equal(np.signbit(y[:-1]),np.signbit(expected[:-1]))
    assert torch.reciprocal(torch.tensor([1,2,4])).dtype==torch.float32
    assert torch.reciprocal(torch.tensor([1,2,4])).tolist()==[1.,.5,.25]
    z=torch.reciprocal(torch.tensor([complex(-0.,0.),1+1j],dtype=torch.complex128)).numpy()
    assert np.isneginf(z.real[0]) and np.isnan(z.imag[0])
    assert z[1]==.5-.5j
    with pytest.raises(TypeError):torch.reciprocal(2)
    with pytest.raises(UnsupportedOperation):torch.reciprocal(x,out=torch.zeros_like(x))


def test_reflected_operator_and_functional_division_keep_distinct_extreme_values():
    x=torch.tensor([1e-310],dtype=torch.float64)
    assert np.isinf((1e-300/x).item())
    np.testing.assert_allclose(torch.div(1e-300,x).item(),1e10,rtol=1e-14)
    p,t=torch.func.jvp(lambda y:torch.div(1e-300,y),(x,),(torch.zeros_like(x),))
    np.testing.assert_allclose(p.item(),1e10,rtol=1e-14)
    assert t.item()==0 and np.signbit(t.item())
    p,t=torch.func.jvp(lambda y:1e-300/y,(x,),(torch.zeros_like(x),))
    assert np.isinf(p.item()) and np.isnan(t.item())


def test_reciprocal_analytic_reverse_and_reverse_over_forward():
    raw=np.array([.5,1.25,-2.],dtype=np.float64)
    tangent=np.array([.125,-.25,.5],dtype=np.float64)
    x=torch.tensor(raw,requires_grad=True);v=torch.tensor(tangent,requires_grad=True)
    p,t=torch.func.jvp(torch.reciprocal,(x,),(v,))
    (p.square().sum()+t.square().sum()).backward()
    expected=1/raw;tv=-tangent/raw**2
    np.testing.assert_allclose(p.numpy(),expected,rtol=1e-14)
    np.testing.assert_allclose(t.numpy(),tv,rtol=1e-14)
    np.testing.assert_allclose(x.grad.numpy(),-2/raw**3-4*tangent**2/raw**5,rtol=1e-14)
    np.testing.assert_allclose(v.grad.numpy(),2*tangent/raw**4,rtol=1e-14)


def test_reciprocal_saved_output_release_retention_and_mutation_guards():
    x=torch.tensor([.5,2.],requires_grad=True);y=x.reciprocal()
    with torch.no_grad():x.fill_(3.)
    y.sum().backward(retain_graph=True);y.sum().backward()
    assert x.grad.tolist()==[-8.,-.5]
    with pytest.raises(RuntimeError):y.sum().backward()
    z=torch.tensor([.5,2.],requires_grad=True).reciprocal()
    with torch.no_grad():z.detach().fill_(3.)
    with pytest.raises(UnsupportedOperation):z.sum().backward()


def test_backward_ieee_kernels_fan_in_existing_grad_and_seed_cast_ignore_external_raise():
    maximum=float(np.finfo(np.float32).max)
    previous=np.seterr(all='raise')
    try:
        x=torch.tensor([1.],requires_grad=True);(x/0.).sum().backward();assert np.isposinf(x.grad.item())
        x=torch.tensor([1.],requires_grad=True);(x*maximum).backward(torch.tensor([2.]));assert np.isposinf(x.grad.item())
        x=torch.tensor([1.],requires_grad=True);(x*maximum+x*maximum).sum().backward();assert np.isposinf(x.grad.item())
        x=torch.tensor([1.],requires_grad=True);x.grad=torch.tensor([maximum]);x.backward(torch.tensor([maximum]));assert np.isposinf(x.grad.item())
        x=torch.tensor([1.],requires_grad=True);x.backward(torch.tensor([1e300],dtype=torch.float64));assert np.isposinf(x.grad.item())
        assert np.geterr()==dict.fromkeys(['divide','over','under','invalid'],'raise')
    finally:np.seterr(**previous)


def test_tensor_callbacks_observe_caller_errors_and_can_change_policy():
    previous=np.seterr(all='raise')
    try:
        x=torch.tensor([1.],requires_grad=True);events=[]
        def changed(g):events.append(np.geterr());np.seterr(over='ignore')
        x.register_hook(changed);x.square().sum().backward()
        assert events==[dict.fromkeys(['divide','over','under','invalid'],'raise')]
        assert np.geterr()['over']=='ignore'
        np.seterr(all='raise');z=torch.tensor([1.],requires_grad=True)
        # Pyodide's pinned NumPy reports the policy but its FPU flags do not
        # necessarily trap. Match direct NumPy behavior in this same runtime.
        try: direct = np.divide(1.,0.)
        except FloatingPointError: traps = True
        else: traps = False
        observed = []
        def post(t): observed.append(np.divide(1.,0.))
        z.register_post_accumulate_grad_hook(post)
        if traps:
            with pytest.raises(FloatingPointError):z.square().sum().backward()
        else:
            z.square().sum().backward()
            assert len(observed)==1 and np.isposinf(observed[0]) and np.isposinf(direct)
        assert np.geterr()==dict.fromkeys(['divide','over','under','invalid'],'raise')
    finally:np.seterr(**previous)


def test_absent_constant_jvp_terms_do_not_generate_infinity_times_zero():
    x=torch.tensor([np.inf,-np.inf,1.],dtype=torch.float64)
    direction=torch.tensor([1.,1.,-0.],dtype=torch.float64)
    p,t=torch.func.jvp(lambda z:z*2.,(x,),(direction,))
    np.testing.assert_array_equal(t.numpy(),np.array([2.,2.,-0.]))
    assert np.signbit(t.numpy()[-1])
    p,t=torch.func.jvp(lambda z:z/2.,(x,),(direction,))
    np.testing.assert_array_equal(t.numpy(),np.array([.5,.5,-0.]))
    p,t=torch.func.jvp(lambda z:torch.add(z,torch.ones_like(z),alpha=np.inf),(x,),(direction,))
    np.testing.assert_array_equal(t.numpy(),direction.numpy())


def test_constant_broadcast_and_mixed_dtype_direction_preserve_zero_sign():
    x=torch.tensor(2.,dtype=torch.float32);v=torch.tensor(-0.,dtype=torch.float32)
    p,t=torch.func.jvp(lambda z:z+torch.ones(3,dtype=torch.float64),(x,),(v,))
    assert p.dtype==torch.float64 and t.dtype==torch.float64 and t.shape==(3,)
    assert np.signbit(t.numpy()).all()
    p,t=torch.func.jvp(lambda z:torch.tensor([True,False])/z,(x,),(torch.ones_like(x),))
    np.testing.assert_array_equal(t.numpy(),np.array([-.25,-0.],dtype=np.float32))


def test_jvp_numpy_complex_number_alpha_is_parsed_once():
    x=torch.tensor([1.,2.]);v=torch.ones_like(x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        p,t=torch.func.jvp(lambda z:torch.add(z,torch.ones_like(z),alpha=np.complex64(2+.5j)),(x,),(v,))
    assert p.tolist()==[3.,4.] and t.tolist()==[1.,1.]
    assert sum('complex' in str(w.message).lower() for w in caught)==1


def test_reflected_scalar_private_inverse_survives_public_mutation_and_releases():
    for mutated in ['input','output']:
        x=torch.tensor([.5,2.],requires_grad=True);y=2./x
        with torch.no_grad():(x if mutated=='input' else y.detach()).fill_(3.)
        y.sum().backward(retain_graph=True);y.sum().backward()
        assert x.grad.tolist()==[-16.,-1.]
        with pytest.raises(RuntimeError):y.sum().backward()


def test_late_hooks_keep_changed_error_handler_and_policy_even_on_exception():
    for raises in [False,True]:
        x=torch.tensor([2.],requires_grad=True);y=x.square();events=[]
        def replacement(message,status):events.append('handler')
        def late(g):events.append(('late',np.geterr()['divide'],np.geterrcall() is replacement))
        def first(g):
            np.seterr(divide='call');np.seterrcall(replacement)
            if raises:raise ValueError('callback failure')
            x.register_hook(late)
        y.register_hook(first)
        previous=np.seterr(all='raise');previous_handler=np.seterrcall(None)
        try:
            if raises:
                with pytest.raises(ValueError,match='callback failure'):y.sum().backward()
                assert events==[]
            else:
                y.sum().backward()
                assert events==[('late','call',True)]
            assert np.geterr()==dict(divide='call',over='raise',under='raise',invalid='raise')
            assert np.geterrcall() is replacement
        finally:np.seterr(**previous);np.seterrcall(previous_handler)


def test_direct_division_gradient_avoids_denominator_square_overflow_and_underflow():
    for dtype,exponent in [(torch.float32,20),(torch.float64,200)]:
        for power in [exponent,-exponent]:
            value=10.**power
            for mode in ['operator','method','function','number']:
                x=torch.tensor([value],dtype=dtype,requires_grad=True)
                y=torch.tensor([value],dtype=dtype,requires_grad=True)
                z=x/y if mode=='operator' else x.div(y) if mode=='method' else torch.div(x if mode=='function' else value,y)
                z.backward(torch.ones_like(z))
                expected=np.array([-1./value],dtype=dtype)
                np.testing.assert_allclose(y.grad.numpy(),expected,rtol=3e-6,atol=0.)
                if mode!='number':np.testing.assert_allclose(x.grad.numpy(),-expected,rtol=3e-6,atol=0.)
