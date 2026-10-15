"""Independent native contracts for arithmetic promotion, APIs and derivatives."""
import itertools
import operator
import warnings

import numpy as np
import pytest
import torch as native

from paperleet_torch import torch, UnsupportedOperation

DTYPES = ['bool','uint8','uint16','uint32','uint64','int8','int16','int32','int64',
          'float16','float32','float64','complex64','complex128']
OPS = {'add':operator.add,'sub':operator.sub,'mul':operator.mul,'div':operator.truediv}


def tensor(lib, dtype, scalar=False):
    raw = np.array([1, 2], dtype=dtype)
    if raw.dtype.kind == 'c': raw += np.array([.25j, .5j], dtype=dtype)
    return lib.tensor(raw[:1].reshape(()) if scalar else raw)


def compare(call):
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try: expected = call(native)
        except Exception as error:
            with pytest.raises(type(error)): call(torch)
            return
        if str(expected.dtype) == 'torch.complex32':
            with pytest.raises(UnsupportedOperation): call(torch)
            return
        actual = call(torch)
        reference = expected.detach().numpy()
        assert actual.dtype == reference.dtype and actual.shape == reference.shape
        tolerance = 1e-3 if reference.dtype.itemsize == 2 else 3e-6 if reference.dtype in (np.dtype('float32'),np.dtype('complex64')) else 2e-13
        np.testing.assert_allclose(actual.numpy(), reference, equal_nan=True, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('name', OPS)
@pytest.mark.parametrize('left,right', tuple(itertools.product(DTYPES, repeat=2)))
@pytest.mark.parametrize('mode', ['vectors','left_scalar','right_scalar','both_scalar'])
def test_tensor_pair_promotion_matches_native_in_operator_function_and_method(name,left,right,mode):
    for style in ['operator','function','method']:
        def run(lib):
            a,b=tensor(lib,left,mode in ('left_scalar','both_scalar')),tensor(lib,right,mode in ('right_scalar','both_scalar'))
            return OPS[name](a,b) if style=='operator' else getattr(lib,name)(a,b) if style=='function' else getattr(a,name)(b)
        compare(run)


SCALARS = [True,2,-.5,1+.5j,np.bool_(True),np.int16(2),np.uint64(1),np.uint64(2**63),
           np.float32(.5),np.float64(.5),np.complex64(1+.5j),np.complex128(1+.5j),
           2**63,2**64-1,2**64,-2**63-1,1e300,float('nan'),float('inf')]


@pytest.mark.parametrize('name', OPS)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('value', SCALARS)
@pytest.mark.parametrize('scalar,reverse', [(False,False),(True,False),(False,True),(True,True)])
def test_function_number_promotion_and_parser_match_native(name,dtype,value,scalar,reverse):
    def run(lib):
        x=tensor(lib,dtype,scalar)
        return getattr(lib,name)(value,x) if reverse else getattr(lib,name)(x,value)
    compare(run)


@pytest.mark.parametrize('name', ['add','sub'])
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('alpha', [True,False,0,2,-1,.25,2.,1+.5j,np.bool_(True),np.float32(.25),
                                  np.complex64(1+.5j),128,-128,1e300])
def test_alpha_category_overflow_and_dtype_match_native(name,dtype,alpha):
    compare(lambda lib:getattr(lib,name)(tensor(lib,dtype),tensor(lib,dtype),alpha=alpha))


@pytest.mark.parametrize('dtype', ['uint8','int8','int32','int64','float16','float32','float64'])
@pytest.mark.parametrize('mode', ['floor','trunc',None])
@pytest.mark.parametrize('reverse', [False,True])
def test_division_rounding_negative_scalar_and_aliases(dtype,mode,reverse):
    for name in ['div','divide']:
        def run(lib):
            x=lib.tensor([-7,-1,1,7] if dtype!='uint8' else [1,2,7,9],dtype=getattr(lib,dtype))
            return getattr(lib,name)(3,x,rounding_mode=mode) if reverse else getattr(lib,name)(x,3,rounding_mode=mode)
        compare(run)


@pytest.mark.parametrize('mode', ['floor','trunc'])
def test_rounded_integer_division_never_rounds_through_float(mode):
    data=np.array([2**62+3,-2**62-3,2**63-1,-2**63+1],dtype=np.int64)
    compare(lambda lib:lib.div(lib.tensor(data),3,rounding_mode=mode))
    for lib in (native,torch):
        with pytest.raises(RuntimeError):lib.div(lib.tensor([1,2]),0,rounding_mode=mode)


@pytest.mark.parametrize('name', OPS)
@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_mixed_dtype_broadcast_reverse_and_forward_graphs(name,dtype):
    function=lambda lib,a,b:getattr(lib,name)(a,b)
    raw=np.array([[1.25,2.5,3.75],[2.,3.,4.]],dtype=dtype)
    other=np.array([.5,1.5,2.5],dtype='float64')
    outputs=[]
    for lib in (native,torch):
        x=lib.tensor(raw,requires_grad=True);y=lib.tensor(other,requires_grad=True)
        result=function(lib,x,y);loss=result.square().sum();loss.backward()
        outputs.append((result.detach().numpy(),x.grad.numpy(),y.grad.numpy()))
    for a,b in zip(*outputs):np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)
    runs=[]
    for lib in (native,torch):
        x=lib.tensor(raw,requires_grad=True);v=lib.tensor(np.full_like(raw,.25),requires_grad=True)
        y=lib.tensor(other,requires_grad=True)
        result,tangent=lib.func.jvp(lambda z:function(lib,z,y),(x,),(v,))
        (result.square().sum()+tangent.square().sum()).backward()
        runs.append((result.detach().numpy(),tangent.detach().numpy(),x.grad.numpy(),v.grad.numpy(),y.grad.numpy()))
    for a,b in zip(*runs):np.testing.assert_allclose(a,b,rtol=3e-6,atol=3e-6)


@pytest.mark.parametrize('name', ['add','sub'])
@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_alpha_reverse_over_forward_and_zero_coefficient(name,dtype):
    for alpha in [0.,.125,-2.5]:
        results=[]
        for lib in (native,torch):
            a=lib.tensor([[1.,2.],[3.,4.]],dtype=getattr(lib,dtype),requires_grad=True)
            b=lib.tensor([.5,1.5],dtype=getattr(lib,dtype),requires_grad=True)
            v=lib.ones_like(a,requires_grad=True)
            primal,tangent=lib.func.jvp(lambda x:getattr(lib,name)(x,b,alpha=alpha),(a,),(v,))
            (primal.square().sum()+tangent.square().sum()).backward()
            results.append((primal.detach().numpy(),tangent.detach().numpy(),a.grad.numpy(),b.grad.numpy(),v.grad.numpy()))
        for a,b in zip(*results):np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)


@pytest.mark.parametrize('mode', ['floor','trunc'])
def test_rounded_division_has_native_zero_vjp_and_jvp(mode):
    results=[]
    for lib in (native,torch):
        a=lib.tensor([1.25,2.75],requires_grad=True);b=lib.tensor(1.5,requires_grad=True)
        result=lib.div(a,b,rounding_mode=mode);result.sum().backward()
        p,t=lib.func.jvp(lambda x:lib.div(x,b,rounding_mode=mode),(a,),(lib.ones_like(a),))
        results.append((result.detach().numpy(),a.grad.numpy(),b.grad.numpy(),p.detach().numpy(),t.detach().numpy()))
    for a,b in zip(*results):np.testing.assert_array_equal(a,b)


def test_public_keywords_aliases_and_typed_boundaries():
    for lib in (native,torch):
        a,b=lib.tensor([1.,2.]),lib.tensor([3.,4.])
        assert lib.add(input=a,other=b,alpha=2,out=None).tolist()==[7.,10.]
        assert lib.subtract(a,b).tolist()==a.sub(b).tolist()
        assert lib.multiply(a,b).tolist()==a.mul(b).tolist()
        assert lib.true_divide(a,b).tolist()==a.true_divide(b).tolist()
        with pytest.raises(TypeError):lib.add(a,b,2)
        with pytest.raises(TypeError):lib.mul(a,b,alpha=2)
        with pytest.raises(RuntimeError):lib.div(a,b,rounding_mode='invalid')
        with pytest.raises(TypeError):lib.div(a,b,rounding_mode=1)
    with pytest.raises(UnsupportedOperation):torch.add(torch.ones(2),torch.ones(2),out=torch.zeros(2))


@pytest.mark.parametrize('function', [lambda lib,x:x-True,lambda lib,x:True-x,
                                     lambda lib,x:lib.sub(x,True),lambda lib,x:lib.sub(True,x)])
def test_boolean_subtraction_is_rejected_before_forward_constant_coercion(function):
    for lib in (native,torch):
        x=lib.tensor([1.,2.]);v=lib.ones_like(x)
        with pytest.raises(NotImplementedError):lib.func.jvp(lambda z:function(lib,z),(x,),(v,))


@pytest.mark.parametrize('value', [np.bool_(True),np.complex64(1+.5j),2**63,1e300])
@pytest.mark.parametrize('name', OPS)
def test_forward_number_protocols_match_primal_value_and_tangent(name,value):
    result=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for lib in (native,torch):
            x=lib.tensor([1.,2.]);v=lib.ones_like(x)
            result.append(lib.func.jvp(lambda z:getattr(lib,name)(z,value),(x,),(v,)))
    for actual,expected in zip(result[1],result[0]):
        assert str(actual.dtype)==str(expected.dtype).removeprefix('torch.')
        np.testing.assert_allclose(actual.numpy(),expected.numpy(),equal_nan=True,rtol=3e-6)


@pytest.mark.parametrize('name', OPS)
@pytest.mark.parametrize('value', [[1,2],np.array([1,2]),None,'x'])
def test_functional_arithmetic_requires_tensor_or_number_operands(name,value):
    for lib in (native,torch):
        with pytest.raises(TypeError):getattr(lib,name)(lib.ones(2),value)


@pytest.mark.parametrize('name', OPS)
def test_double_wrapped_scalar_precision_survives_forward_mode(name):
    scalar=1.+2.**-40
    pairs=[]
    for lib in (native,torch):
        x=lib.tensor([1.,2.],dtype=lib.float64);v=lib.ones_like(x)
        pairs.append(lib.func.jvp(lambda z:getattr(lib,name)(z,scalar),(x,),(v,)))
    for a,b in zip(pairs[1],pairs[0]):np.testing.assert_array_equal(a.numpy(),b.numpy())


@pytest.mark.parametrize('function', [lambda x:x+1j,lambda x:x*1j,lambda x:torch.add(x,1j,alpha=2),
                                     lambda x:torch.div(x,1j),lambda x:x.to(dtype=torch.complex64)])
def test_unvalidated_complex_reverse_graphs_fail_explicitly(function):
    x=torch.ones(2,requires_grad=True)
    with pytest.raises(UnsupportedOperation):function(x)
    with torch.no_grad():assert function(x).dtype.kind=='c'
