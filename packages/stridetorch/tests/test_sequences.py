"""Sequence rounding, scalar controls, leaf gradients and allocation bounds."""
from fractions import Fraction

import numpy as np
import pytest

from stridetorch import UnsupportedOperation, torch


@pytest.fixture(autouse=True)
def restore_default():
    previous=torch.get_default_dtype()
    try:yield
    finally:torch.set_default_dtype(previous)


@pytest.mark.parametrize('dtype',[torch.int8,torch.int16,torch.int32,torch.int64])
@pytest.mark.parametrize('start,end,count',[(-1,1,4),(1,-1,4),(-1.8,2.8,4),(-7,4,17),(0,7,3)])
def test_integer_linspace_truncates_cast_endpoints_and_samples(dtype,start,end,count):
    a,b=int(start),int(end)
    expected=[int(Fraction(a)+Fraction(b-a)*Fraction(i,count-1)) for i in range(count)]
    value=torch.linspace(start,end,count,dtype=dtype)
    assert value.dtype==dtype
    np.testing.assert_array_equal(value.numpy(),expected)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64,torch.complex64,torch.complex128])
@pytest.mark.parametrize('count',[0,1,2,3,17,101])
def test_linear_sequence_shape_dtype_storage_and_finite_endpoints(dtype,count):
    x=torch.linspace(-1,1,count,dtype=dtype)
    assert x.shape==(count,) and x.dtype==dtype and x.stride()==(1,)
    assert x.is_leaf and not x.requires_grad
    if count:
        assert x[0].item()==-1
        if count>1:assert x[-1].item()==1
    other=torch.linspace(-1,1,count,dtype=dtype)
    assert not np.shares_memory(x.numpy(),other.numpy())


@pytest.mark.parametrize('count',[2,3,4,5,17])
def test_half_wide_linspace_has_finite_widened_intermediates(count):
    # Current upstream CPU opmath: round endpoints to half, interpolate wider,
    # then round outputs once. Native 2.14.1 overflows this finite span in half.
    values=torch.linspace(-32768,32767,count,dtype=torch.float16)
    oracle=np.array([float(Fraction(-32768)+Fraction(65536)*Fraction(i,count-1)) for i in range(count)]).astype('float16')
    assert np.isfinite(values.numpy()).all()
    np.testing.assert_array_equal(values.numpy(),oracle)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64])
def test_arange_float_grid_does_not_collapse_small_steps(dtype):
    start=2048 if dtype==torch.float16 else 1e8 if dtype==torch.float32 else 2**53
    value=torch.arange(start,start+20,1,dtype=dtype)
    expected=np.array([start+i for i in range(20)],dtype=dtype)
    np.testing.assert_array_equal(value.numpy(),expected)
    assert np.unique(value.numpy()).size>1


@pytest.mark.parametrize('dtype',[torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64])
def test_arange_uses_original_bounds_for_size_and_truncated_integer_step(dtype):
    np.testing.assert_array_equal(torch.arange(.2,3.,.5,dtype=dtype).numpy(),np.zeros(6,dtype=dtype))
    np.testing.assert_array_equal(torch.arange(1.8,5.5,1.2,dtype=dtype).numpy(),[1,2,3,4])


def test_arange_int64_preserves_large_integer_offsets():
    for start in [2**53-3,2**63-5,-2**63]:
        value=torch.arange(start,start+4,dtype=torch.int64)
        np.testing.assert_array_equal(value.numpy(),np.array([start+i for i in range(4)],dtype='int64'))
    np.testing.assert_array_equal(torch.arange(3,-3,-2).numpy(),[3,1,-1])


def test_arange_keyword_overloads_and_complex_real_controls():
    np.testing.assert_array_equal(torch.arange(end=3).numpy(),[0,1,2])
    np.testing.assert_array_equal(torch.arange(start=1,end=4,step=2).numpy(),[1,3])
    np.testing.assert_array_equal(torch.arange(1,end=4).numpy(),[1,2,3])
    np.testing.assert_array_equal(torch.arange(0j,3+0j).numpy(),[0,1,2])
    for args,kw in [((3,),{'step':2}),((),{'end':3,'step':2}),((1,3),{'start':0}),((1,3,2),{'step':2}),((),{'start':3})]:
        with pytest.raises(TypeError):torch.arange(*args,**kw)


@pytest.mark.parametrize('args',[(3,1,1),(1,3,-1),(0,1,0),(0,1,float('nan')),(float('inf'),1,1),(0,float('inf'),1),(1j,2,1)])
def test_arange_rejects_invalid_bounds_without_allocating(args):
    with pytest.raises(RuntimeError):torch.arange(*args)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64])
def test_infinite_arange_step_matches_native_empty_float_range(dtype):
    value=torch.arange(0,1,float('inf'),dtype=dtype)
    assert value.shape==(0,) and value.stride()==(1,)
    with pytest.raises(RuntimeError):torch.arange(0,1,float('inf'),dtype=torch.int64)


@pytest.mark.parametrize('dtype',[torch.bool,torch.uint16,torch.uint32,torch.uint64])
@pytest.mark.parametrize('name',['linspace','logspace'])
def test_storage_only_dtypes_support_empty_and_singleton_but_not_kernels(dtype,name):
    fn=getattr(torch,name)
    assert fn(0,1,0,dtype=dtype).shape==(0,)
    assert fn(0,1,1,dtype=dtype).item()==(0 if name=='linspace' else 1)
    with pytest.raises(UnsupportedOperation):fn(0,1,2,dtype=dtype)
    with pytest.raises(UnsupportedOperation):torch.arange(2,dtype=dtype)


def test_uint64_singleton_scalar_conversion_is_not_sequence_constructor_storage():
    assert torch.linspace(-1,0,1,dtype=torch.uint64).item()==2**64-1
    assert torch.linspace(1.9,0,1,dtype=torch.uint64).item()==1
    assert torch.logspace(1.9,0,1,dtype=torch.uint64).item()==79
    with pytest.raises(RuntimeError):torch.linspace(-.5,0,1,dtype=torch.uint64)


@pytest.mark.parametrize('name',['linspace','logspace'])
@pytest.mark.parametrize('value',[True,np.bool_(True),3.,-1])
def test_steps_require_nonnegative_integer_controls(name,value):
    error=RuntimeError if type(value) is int and value<0 else TypeError
    with pytest.raises(error):getattr(torch,name)(0,1,value)


@pytest.mark.parametrize('name',['linspace','logspace'])
def test_scalar_integer_tensor_steps_and_endpoint_shapes(name):
    fn=getattr(torch,name)
    assert fn(0,1,np.int64(3)).shape==(3,)
    assert fn(0,1,torch.tensor(3)).shape==(3,)
    for value in [torch.tensor([3]),torch.tensor(3.),torch.tensor(True)]:
        with pytest.raises(TypeError):fn(0,1,value)
    with pytest.raises(RuntimeError):fn(torch.ones(1),1,3)
    for count in [0,1,3]:
        with pytest.raises(RuntimeError):fn(0j,1+0j,count,dtype=torch.float64)


@pytest.mark.parametrize('name',['arange','linspace','logspace'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_factories_create_independent_grad_leaves_and_preserve_error_mode(name,dtype):
    torch.set_default_dtype(dtype);start=torch.tensor(0.,requires_grad=True)
    previous=np.seterr(all='raise')
    try:
        options={'step':1.} if name=='arange' else {'steps':3}
        if name=='arange':
            with pytest.raises(TypeError):torch.arange(start,2.,requires_grad=True,**options)
            start=start.detach()
        value=getattr(torch,name)(start,2.,requires_grad=True,**options)
        assert value.is_leaf and value.requires_grad and value.dtype==dtype
        value.square().sum().backward()
        assert start.grad is None
        np.testing.assert_array_equal(value.grad.numpy(),2*value.numpy())
        assert np.geterr()==dict.fromkeys(['divide','over','under','invalid'],'raise')
    finally:np.seterr(**previous)


def test_forward_mode_factory_control_boundaries_match_native():
    x=torch.tensor(0.,requires_grad=True)
    primal,tangent=torch.func.jvp(lambda a:torch.arange(a,3.),(x,),(torch.ones_like(x),))
    np.testing.assert_array_equal(primal.numpy(),[0,1,2]);np.testing.assert_array_equal(tangent.numpy(),0)
    for name in ['linspace','logspace']:
        with pytest.raises(UnsupportedOperation):torch.func.jvp(lambda a:getattr(torch,name)(a,2.,3),(x,),(torch.ones_like(x),))
        p,j=torch.func.jvp(lambda _:getattr(torch,name)(0.,2.,3),(x,),(torch.ones_like(x),))
        np.testing.assert_array_equal(j.numpy(),0);assert p.dtype==torch.float32


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64])
def test_logspace_real_bases_and_integer_endpoint_casting(dtype):
    np.testing.assert_array_equal(torch.logspace(0,2,3,dtype=dtype).numpy(),[1,10,100])
    np.testing.assert_array_equal(torch.logspace(0,2,3,base=-2,dtype=dtype).numpy(),[1,-2,4])
    np.testing.assert_array_equal(torch.logspace(0,2,3,base=0,dtype=dtype).numpy(),[1,0,0])
    with np.errstate(all='raise'):
        value=torch.logspace(0,.5,3,base=-2,dtype=dtype)
    assert value[0].item()==1 and np.isnan(value.numpy()[1:]).all()
    assert torch.logspace(1.9,2.9,1,dtype=torch.int64).item()==79
    np.testing.assert_array_equal(torch.logspace(1.9,2.9,2,dtype=torch.int64).numpy(),[10,100])
    np.testing.assert_array_equal(torch.logspace(-1,1,4,dtype=torch.int64).numpy(),[0,0,2,10])


@pytest.mark.parametrize('dtype',[torch.complex64,torch.complex128])
def test_complex_nonfinite_components_and_zero_log_base(dtype):
    value=torch.linspace(float('inf'),2,3,dtype=dtype).numpy()
    np.testing.assert_array_equal(np.isnan(value.real),[True,False,True]);assert np.isposinf(value.real[1])
    np.testing.assert_array_equal(value.imag,0)
    value=torch.logspace(-1,1,3,base=0,dtype=dtype).numpy()
    assert np.isposinf(value.real[0]) and np.isnan(value.imag[0])
    assert np.isnan(value.real[1]) and np.isnan(value.imag[1])
    assert value[2]==0j


def test_half_singleton_fill_uses_opmath_conversion_but_multi_element_endpoints_are_checked():
    assert np.isposinf(torch.linspace(70000,0,1,dtype=torch.float16).item())
    with pytest.raises(RuntimeError):torch.linspace(70000,0,2,dtype=torch.float16)
    assert torch.linspace(70000,0,0,dtype=torch.float16).shape==(0,)
    assert np.isposinf(torch.logspace(5,0,1,dtype=torch.float16).item())


def test_small_fixed_interval_templates_are_bounded_immutable_and_independently_owned():
    from stridetorch import _sequences
    cache=_sequences._unit_grid_bytes;cache.cache_clear()
    a=torch.linspace(0,1,17,requires_grad=True);b=torch.linspace(0,1,17)
    assert cache.cache_info().hits>=1 and not np.shares_memory(a.numpy(),b.numpy())
    with torch.no_grad():a[0]=7.
    assert b[0].item()==0 and torch.linspace(0,1,17)[0].item()==0
    blob=cache(torch.float32.str,17,0);assert type(blob) is bytes
    with pytest.raises(ValueError):np.frombuffer(blob,dtype='float32').setflags(write=True)
    for count in range(2,100):torch.linspace(-1,1,count)
    assert cache.cache_info().currsize<=64
    before=cache.cache_info()
    torch.linspace(.25,3.75,16);torch.linspace(0,1,257)
    assert cache.cache_info()==before


@pytest.mark.parametrize('name',['arange','linspace','logspace'])
def test_browser_budget_and_unsupported_options_fail_before_large_allocations(name):
    args=(16_000_001,) if name=='arange' else (0,1,16_000_001)
    with pytest.raises(UnsupportedOperation):getattr(torch,name)(*args)
    args=(3,) if name=='arange' else (0,1,3)
    for kw in [{'out':torch.empty(3)},{'pin_memory':True},{'layout':'strided'},{'device':'cuda'}]:
        with pytest.raises(UnsupportedOperation):getattr(torch,name)(*args,**kw)
    for kw in [{'requires_grad':1},{'pin_memory':1},{'unknown_option':1}]:
        with pytest.raises(TypeError):getattr(torch,name)(*args,**kw)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64,torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64])
@pytest.mark.parametrize('count',[0,1,17])
def test_integer_control_ranges_have_owned_storage_for_all_overloads(dtype,count):
    expected=np.asarray(list(range(count)),dtype=dtype)
    calls=[lambda:torch.arange(count,dtype=dtype),lambda:torch.arange(0,count,dtype=dtype),
           lambda:torch.arange(0,count,1,dtype=dtype),lambda:torch.arange(end=count,dtype=dtype),
           lambda:torch.arange(start=0,end=count,step=1,dtype=dtype)]
    values=[fn() for fn in calls]
    for value in values:
        np.testing.assert_array_equal(value.numpy(),expected)
        assert value.dtype==dtype and value.stride()==(1,) and value.is_leaf
    if count:
        values[0][0]=7
        for value in values[1:]:np.testing.assert_array_equal(value.numpy(),expected)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.int8,torch.uint8])
@pytest.mark.parametrize('mode',['ignore','warn','raise'])
def test_large_unit_ranges_preserve_overflow_storage_and_callers_error_mode(dtype,mode):
    # Integer wrapping and half overflow must not depend on global NumPy fenv.
    with np.errstate(all='ignore'):expected=np.asarray(list(range(65521)),dtype='int64').astype(dtype)
    previous=np.seterr(all=mode)
    try:
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            value=torch.arange(65521,dtype=dtype)
        np.testing.assert_array_equal(value.numpy(),expected)
        assert not caught and np.geterr()==dict.fromkeys(['divide','over','under','invalid'],mode)
    finally:np.seterr(**previous)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32,torch.float64])
def test_integer_control_float_ranges_create_requested_leaves_inside_no_grad(dtype):
    torch.arange(17,dtype=dtype)
    with torch.no_grad():value=torch.arange(17,dtype=dtype,requires_grad=True)
    assert value.is_leaf and value.requires_grad
    value.square().sum().backward()
    np.testing.assert_array_equal(value.grad.numpy(),np.asarray([2*i for i in range(17)],dtype=dtype))
