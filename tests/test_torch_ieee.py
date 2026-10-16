"""Native IEEE arithmetic, reciprocal history and NumPy-policy isolation."""
import itertools
import warnings

import numpy as np
import pytest
import torch as native

from stridetorch import torch, UnsupportedOperation

DTYPES=['bool','uint8','uint16','uint32','uint64','int8','int16','int32','int64',
        'float16','float32','float64','complex64','complex128']


def equal(actual,expected):
    equal_arrays(actual.numpy(),expected.detach().numpy())


def equal_arrays(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape
    with np.errstate(all='ignore'):
        if a.dtype.kind=='c':
            np.testing.assert_allclose(a.real,b.real,equal_nan=True,rtol=3e-6,atol=1e-14)
            np.testing.assert_allclose(a.imag,b.imag,equal_nan=True,rtol=3e-6,atol=1e-14)
        else:
            np.testing.assert_allclose(a,b,equal_nan=True,rtol=1e-3 if a.dtype.itemsize==2 else 3e-6,atol=1e-14)
            mask=b==0
            np.testing.assert_array_equal(np.signbit(a[mask]),np.signbit(b[mask]))


def equal_reciprocal(actual,expected,source):
    if source.dtype.kind!='c':
        return equal(actual,expected)
    a,b=actual.numpy(),expected.detach().numpy()
    assert a.dtype==b.dtype==source.dtype and a.shape==b.shape==source.shape
    zero=(source.real==0)&(source.imag==0)
    # Native Torch 2.14.1 produces different real infinity signs at complex
    # zero on macOS arm64 and Linux x86_64. Its inf/nan classification is
    # shared; every nonzero input still uses the ordinary strict comparison.
    for value in (a,b):
        assert np.isinf(value.real[zero]).all()
        assert np.isnan(value.imag[zero]).all()
    equal_arrays(a[~zero],b[~zero])


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('shape',[(),(7,),(2,7),(0,)])
def test_reciprocal_values_dtype_shape_and_method_match_native(dtype,shape):
    size=int(np.prod(shape)) if shape else 1
    raw=np.resize(np.array([0.,-0.,1.,-2.,np.inf,-np.inf,np.nan]) if dtype.startswith(('float','complex')) else np.array([0,1,2,3]),size).reshape(shape).astype(dtype)
    for method in [False,True]:
        expected=native.tensor(raw).reciprocal() if method else native.reciprocal(native.tensor(raw))
        actual=torch.tensor(raw).reciprocal() if method else torch.reciprocal(torch.tensor(raw))
        equal_reciprocal(actual,expected,raw)


@pytest.mark.parametrize('dtype',['complex64','complex128'])
def test_complex_reciprocal_ieee_grid_matches_native(dtype):
    values=[0.,-0.,1.,-2.,float('inf'),float('-inf'),float('nan'),1e-40,1e30]
    raw=np.array([complex(a,b) for a,b in itertools.product(values,repeat=2)],dtype=dtype)
    equal_reciprocal(torch.reciprocal(torch.tensor(raw)),native.reciprocal(native.tensor(raw)),raw)


@pytest.mark.parametrize('case',['finite','zero_real','zero_imag','nonzero_inf_sign'])
def test_reciprocal_native_comparison_rejects_wrong_values_and_classifications(case):
    if case=='finite':
        source=np.array([1+0j]);actual=[2+0j];expected=[1+0j]
    elif case=='zero_real':
        source=np.array([0j]);actual=[complex(np.nan,np.nan)];expected=[complex(np.inf,np.nan)]
    elif case=='zero_imag':
        source=np.array([0j]);actual=[complex(np.inf,0.)];expected=[complex(np.inf,np.nan)]
    else:
        source=np.array([complex(1e-310,0.)]);actual=[complex(-np.inf,np.nan)];expected=[complex(np.inf,np.nan)]
    with pytest.raises(AssertionError):
        equal_reciprocal(torch.tensor(actual,dtype=torch.complex128),native.tensor(expected,dtype=native.complex128),source)


@pytest.mark.parametrize('dtype',['complex64','complex128'])
def test_reciprocal_native_comparison_accepts_both_observed_zero_sign_conventions(dtype):
    source=np.array([complex(real,imag) for real in [0.,-0.] for imag in [0.,-0.]]+[1+1j],dtype=dtype)
    actual=torch.reciprocal(torch.tensor(source))
    original=native.reciprocal(native.tensor(source)).numpy()
    zero=source==0
    for signed in [False,True]:
        expected=original.copy()
        expected.real[zero]=np.copysign(np.inf,source.real[zero]) if signed else np.inf
        equal_reciprocal(actual,native.tensor(expected),source)


@pytest.mark.parametrize('dtype',['float16','float32','float64'])
@pytest.mark.parametrize('scalar',[0.,1.,-2.,1e-300,1e300])
def test_reflected_division_uses_native_reciprocal_then_multiply(dtype,scalar):
    kind=np.dtype(dtype);tiny=np.finfo(kind).tiny
    raw=np.array([tiny/2,tiny,np.finfo(kind).max,0.,-0.,1.,.3,np.inf,-np.inf,np.nan],dtype=kind)
    equal(scalar/torch.tensor(raw),scalar/native.tensor(raw))
    equal(torch.div(scalar,torch.tensor(raw)),native.div(scalar,native.tensor(raw)))


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('function',[lambda lib,x:lib.reciprocal(x),lambda lib,x:1.5/x,
                                     lambda lib,x:lib.div(1.5,x)])
def test_reciprocal_reverse_jvp_and_reverse_over_forward_match_native(dtype,function):
    results=[]
    for lib in (native,torch):
        x=lib.tensor([.3,1.25,-2.5],dtype=getattr(lib,dtype),requires_grad=True)
        t=lib.tensor([.25,-.5,.125],dtype=getattr(lib,dtype),requires_grad=True)
        p,v=lib.func.jvp(lambda z:function(lib,z),(x,),(t,))
        (p.square().sum()+v.square().sum()).backward()
        results.append((p,v,x.grad,t.grad))
    for a,b in zip(results[1],results[0]):equal(a,b)


@pytest.mark.parametrize('dtype,value,seed',[('float32',1e-20,1e-30),('float64',1e-200,1e-300),('float64',1e-310,0.)])
def test_reciprocal_derivative_order_preserves_native_overflow(dtype,value,seed):
    records=[]
    for lib in (native,torch):
        x=lib.tensor([value],dtype=getattr(lib,dtype),requires_grad=True);s=lib.tensor([seed],dtype=getattr(lib,dtype))
        p,t=lib.func.jvp(lib.reciprocal,(x,),(s,));lib.reciprocal(x).backward(s)
        records.append((p,t,x.grad))
    for a,b in zip(records[1],records[0]):equal(a,b)


@pytest.mark.parametrize('mutated',['input','output'])
def test_reciprocal_versions_guard_only_the_saved_output(mutated):
    for lib in (native,torch):
        x=lib.tensor([.5,2.],requires_grad=True);y=x.reciprocal()
        with lib.no_grad():(x if mutated=='input' else y.detach()).fill_(3.)
        if mutated=='input':
            y.sum().backward();assert x.grad.tolist()==[-4.,-.25]
        else:
            with pytest.raises(RuntimeError if lib is native else UnsupportedOperation):y.sum().backward()


@pytest.mark.parametrize('case',['div_zero','mul_overflow','fan_in','leaf_existing','seed_cast','multi_root','joint'])
@pytest.mark.parametrize('policy',['raise','warn','ignore'])
def test_reverse_kernels_and_accumulation_isolate_numpy_error_settings(case,policy):
    records=[];maximum=float(np.finfo(np.float32).max)
    for lib in (native,torch):
        x=lib.tensor([1.],requires_grad=True)
        seed=lib.tensor([2.])
        if case=='div_zero':value=x/0.
        elif case=='mul_overflow':value=x*maximum
        elif case=='fan_in':value=x*maximum+x*maximum;seed=lib.ones(1)
        elif case=='leaf_existing':x.grad=lib.tensor([maximum]);value=x;seed=lib.tensor([maximum])
        elif case=='seed_cast':value=x;seed=lib.tensor([1e300],dtype=lib.float64)
        elif case=='multi_root':value=x;seed=lib.tensor([maximum])
        else:
            x=lib.tensor([[1.,2.]],requires_grad=True)
            value=lib.nn.functional.linear(x,lib.tensor([[maximum,maximum]]));seed=lib.tensor([[2.]])
        previous=np.seterr(all=policy)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if case=='multi_root':lib.autograd.backward((value,value),(seed,seed))
                else:value.backward(seed)
                assert not any(issubclass(v.category,RuntimeWarning) for v in caught)
            assert np.geterr()==dict.fromkeys(['divide','over','under','invalid'],policy)
            records.append(x.grad)
        finally:np.seterr(**previous)
    equal(records[1],records[0])


@pytest.mark.parametrize('phase',['leaf','nonleaf','post'])
def test_user_gradient_hooks_keep_caller_numpy_error_policy_and_errors(phase):
    for lib in (native,torch):
        x=lib.tensor([2.],requires_grad=True);y=x.square();events=[]
        def hook(value):
            events.append(np.geterr().copy())
            np.divide(1.,0.)
        if phase=='post':x.register_post_accumulate_grad_hook(hook)
        elif phase=='nonleaf':y.register_hook(hook)
        else:x.register_hook(hook)
        previous=np.seterr(all='raise')
        try:
            with pytest.raises(FloatingPointError):y.sum().backward()
            assert events==[dict.fromkeys(['divide','over','under','invalid'],'raise')]
            assert np.geterr()==events[0]
        finally:np.seterr(**previous)


@pytest.mark.parametrize('phase',['leaf','nonleaf','post'])
def test_hook_policy_changes_persist_across_other_numeric_nodes_and_after_backward(phase):
    for lib in (native,torch):
        x=lib.tensor([2.],requires_grad=True);y=x.square();events=[]
        def hook(value):
            events.append(np.geterr().copy());np.seterr(over='ignore',divide='warn')
        if phase=='post':x.register_post_accumulate_grad_hook(hook)
        elif phase=='nonleaf':y.register_hook(hook)
        else:x.register_hook(hook)
        previous=np.seterr(all='raise')
        try:
            y.sum().backward()
            assert events==[dict.fromkeys(['divide','over','under','invalid'],'raise')]
            assert np.geterr()==dict(divide='warn',over='ignore',under='raise',invalid='raise')
        finally:np.seterr(**previous)


def test_functional_and_python_reflected_division_keep_distinct_extreme_jvps():
    for named in [False,True]:
        pairs=[]
        for lib in (native,torch):
            x=lib.tensor([1e-310],dtype=lib.float64);v=lib.zeros_like(x)
            pairs.append(lib.func.jvp(lambda y:lib.div(1e-300,y) if named else 1e-300/y,(x,),(v,)))
        for a,b in zip(pairs[1],pairs[0]):equal(a,b)


@pytest.mark.parametrize('dtype',['bool','int64','float32','float64'])
def test_tensor_numerator_forward_subclass_uses_direct_division(dtype):
    results=[]
    for lib in (native,torch):
        n=lib.tensor([1,2],dtype=getattr(lib,dtype));x=lib.tensor([.5,2.],dtype=lib.float64)
        results.append(lib.func.jvp(lambda y:n/y,(x,),(lib.ones_like(x),)))
    for a,b in zip(results[1],results[0]):equal(a,b)


@pytest.mark.parametrize('function',[lambda lib,x:x*2.,lambda lib,x:x/2.,lambda lib,x:x+2.,
    lambda lib,x:lib.add(x,lib.ones_like(x),alpha=float('inf')),
    lambda lib,x:lib.sub(x,lib.ones_like(x),alpha=float('inf'))])
def test_undefined_constant_tangents_do_not_create_zero_times_infinity(function):
    values=[]
    for lib in (native,torch):
        x=lib.tensor([float('inf'),-float('inf'),1.],dtype=lib.float64);t=lib.tensor([1.,1.,-0.],dtype=lib.float64)
        values.append(lib.func.jvp(lambda y:function(lib,y),(x,),(t,)))
    for a,b in zip(values[1],values[0]):equal(a,b)


@pytest.mark.parametrize('function',[lambda lib,x:lib.ones(2,dtype=lib.float64)*x,
    lambda lib,x:lib.add(lib.ones(2,dtype=lib.float64),x,alpha=.5),
    lambda lib,x:lib.sub(lib.ones(2,dtype=lib.float64),x,alpha=.5),lambda lib,x:x+lib.ones(2,dtype=lib.float64)])
def test_constant_broadcast_and_mixed_dtype_jvp_has_native_output_shape(function):
    values=[]
    for lib in (native,torch):
        x=lib.tensor(2.,dtype=lib.float32);t=lib.tensor(-0.,dtype=lib.float32)
        values.append(lib.func.jvp(lambda y:function(lib,y),(x,),(t,)))
    for a,b in zip(values[1],values[0]):equal(a,b)


@pytest.mark.parametrize('hooked',[False,True])
def test_numpy_error_handlers_are_suppressed_only_inside_numeric_kernels(hooked):
    for lib in (native,torch):
        x=lib.tensor([1.],requires_grad=True);value=x/0.;calls=[]
        def handler(message,status):calls.append((message,status))
        def hook(g):np.divide(1.,0.)
        if hooked:x.register_hook(hook)
        previous=np.seterr(all='call');previous_handler=np.seterrcall(handler)
        try:
            value.backward(lib.ones(1))
            assert len(calls)==int(hooked)
            assert np.geterr()==dict.fromkeys(['divide','over','under','invalid'],'call')
        finally:np.seterr(**previous);np.seterrcall(previous_handler)


@pytest.mark.parametrize('mutated',['input','output'])
def test_reflected_number_division_saves_private_inverse_and_allows_public_mutation(mutated):
    for lib in (native,torch):
        x=lib.tensor([.5,2.],requires_grad=True);y=2./x
        with lib.no_grad():(x if mutated=='input' else y.detach()).fill_(3.)
        y.sum().backward(retain_graph=True);y.sum().backward()
        assert x.grad.tolist()==[-16.,-1.]
        with pytest.raises(RuntimeError):y.sum().backward()


@pytest.mark.parametrize('raises',[False,True])
def test_callback_error_handler_and_policy_changes_survive_numerical_scopes(raises):
    for lib in (native,torch):
        x=lib.tensor([2.],requires_grad=True);y=x.square();calls=[]
        def replacement(message,status):calls.append(message)
        def first(g):
            np.seterr(divide='call');np.seterrcall(replacement)
            if raises:raise ValueError('callback failure')
        def late(g):np.divide(1.,0.)
        def register(g):first(g);x.register_hook(late)
        y.register_hook(register)
        old=np.seterr(all='raise');old_handler=np.seterrcall(None)
        try:
            if raises:
                with pytest.raises(ValueError,match='callback failure'):y.sum().backward()
            else:y.sum().backward();assert len(calls)==1
            assert np.geterr()['divide']=='call' and np.geterrcall() is replacement
        finally:np.seterr(**old);np.seterrcall(old_handler)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('mode',['operator','function','method'])
@pytest.mark.parametrize('scale',['large','small','ordinary'])
@pytest.mark.parametrize('seed_value',[1.,0.,-0.,1e-20,1e20])
def test_direct_division_denominator_vjp_preserves_native_quotient_first_order(dtype,mode,scale,seed_value):
    exponent=20 if dtype=='float32' else 200
    value=10.**exponent if scale=='large' else 10.**-exponent if scale=='small' else 2.
    results=[]
    for lib in (native,torch):
        x=lib.tensor([value,-value],dtype=getattr(lib,dtype),requires_grad=True)
        y=lib.tensor([value,value],dtype=getattr(lib,dtype),requires_grad=True)
        seed=lib.tensor([seed_value,seed_value],dtype=getattr(lib,dtype))
        z=x/y if mode=='operator' else lib.div(x,y) if mode=='function' else x.div(y)
        z.backward(seed);results.append((z,x.grad,y.grad))
    for actual,expected in zip(results[1],results[0]):
        equal(actual,expected)
        np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),rtol=3e-6,atol=0.,equal_nan=True)


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('scale',['large','small'])
@pytest.mark.parametrize('numerator',['number','tensor'])
def test_direct_division_constant_numerator_and_broadcast_extreme_vjp(dtype,scale,numerator):
    exponent=20 if dtype=='float32' else 200
    value=10.**(exponent if scale=='large' else -exponent);results=[]
    for lib in (native,torch):
        y=lib.tensor([[value],[value]],dtype=getattr(lib,dtype),requires_grad=True)
        first=value if numerator=='number' else lib.tensor([value,-value,value],dtype=getattr(lib,dtype))
        z=lib.div(first,y);z.backward(lib.ones_like(z));results.append(y.grad)
    np.testing.assert_allclose(results[1].numpy(),results[0].numpy(),rtol=3e-6,atol=0.,equal_nan=True)


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_extreme_direct_division_reverse_over_forward_preserves_native_gradients(dtype):
    value=1e20 if dtype=='float32' else 1e200;results=[]
    for lib in (native,torch):
        x=lib.tensor([value],dtype=getattr(lib,dtype),requires_grad=True)
        y=lib.tensor([value],dtype=getattr(lib,dtype),requires_grad=True)
        v=lib.tensor([1.],dtype=getattr(lib,dtype),requires_grad=True)
        p,t=lib.func.jvp(lambda a,b:a/b,(x,y),(v,v))
        (p+t).backward();results.append((p,t,x.grad,y.grad,v.grad))
    for actual,expected in zip(results[1],results[0]):
        equal(actual,expected)
        np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),rtol=3e-6,atol=0.,equal_nan=True)
