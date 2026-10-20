"""Compatibility of private square scratch buffers, including IEEE VJP order."""
import numpy as np
import pytest
from torchwasm import torch, _arithmetic


@pytest.fixture(autouse=True)
def exercise_wasm_branch_on_cpu(monkeypatch):
    monkeypatch.setattr(_arithmetic, '_WASM', True)


def data(dtype, shape, layout):
    with np.errstate(all='ignore'):
        value=np.resize(np.array([0.,-0.,1.,-1.,np.inf,-np.inf,np.nan,
            np.finfo(dtype).tiny,np.nextafter(dtype(0),dtype(1)),
            np.finfo(dtype).max,-np.finfo(dtype).max,.375],dtype=dtype),shape)
    if layout=='F':value=np.asfortranarray(value).reshape(shape,order='F')
    if layout=='strided' and shape:
        backing=np.empty(shape+(2,),dtype=dtype);backing[...,0]=value;value=backing[...,0]
    if layout=='readonly':value.flags.writeable=False
    return value


def exact(actual,expected):
    actual,expected=np.asarray(actual),np.asarray(expected)
    assert actual.dtype==expected.dtype and actual.shape==expected.shape
    np.testing.assert_array_equal(actual,expected)
    # Positive/negative zero and infinity signs must survive the same operation order.
    finite_or_inf=~np.isnan(expected)
    np.testing.assert_array_equal(np.signbit(actual[finite_or_inf]),np.signbit(expected[finite_or_inf]))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(),(12,),(3,4)])
@pytest.mark.parametrize('layout',['C','F','strided','readonly'])
def test_primal_values_dtype_strides_and_input_immutability(dtype,shape,layout):
    source=data(dtype,shape,layout)
    before=source.tobytes();strides=source.strides;writable=source.flags.writeable
    with np.errstate(all='ignore'):
        expected=np.multiply(source,source)
        actual=_arithmetic.scalar_power_values(source,2)
    exact(actual,expected)
    assert np.asarray(actual).strides==np.asarray(expected).strides
    assert source.tobytes()==before and source.strides==strides and source.flags.writeable==writable
    assert not np.shares_memory(source,actual)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('seed_dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(),(12,),(3,4)])
@pytest.mark.parametrize('layout',['C','F','strided','readonly','broadcast-one','broadcast-half','broadcast-nan','broadcast-infinity','broadcast-neg-one','broadcast-neg-zero'])
def test_pullback_values_operation_order_and_no_shared_buffer_edits(dtype,seed_dtype,shape,layout):
    source=data(dtype,shape,'C' if layout=='broadcast' else layout)
    gradient=data(seed_dtype,shape,'C' if layout=='broadcast' else layout)
    if layout.startswith('broadcast-'):
        number={'broadcast-one':1.,'broadcast-half':.5,'broadcast-nan':np.nan,'broadcast-infinity':np.inf,'broadcast-neg-one':-1.,'broadcast-neg-zero':-0.}[layout]
        gradient=np.broadcast_to(np.asarray(number,dtype=seed_dtype),shape)
    before=source.tobytes(),gradient.tobytes()
    with np.errstate(all='ignore'):
        expected=gradient*(2*source)
        actual=_arithmetic.square_backward(source,gradient)
    exact(actual,expected)
    assert (source.tobytes(),gradient.tobytes())==before
    assert not np.shares_memory(actual,source) and not np.shares_memory(actual,gradient)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('shape',[(0,),(2,0,3),(0,4)])
def test_empty_layout_keeps_existing_kernel_policy(dtype,shape):
    source=np.empty(shape,dtype=dtype);gradient=np.empty(shape,dtype=dtype)
    expected=_arithmetic.scalar_power_values(source,2)
    assert expected.shape==shape and expected.dtype==source.dtype
    actual=_arithmetic.square_backward(source,gradient)
    assert actual.shape==shape and actual.dtype==source.dtype


@pytest.mark.parametrize('retained',[False,True])
@pytest.mark.parametrize('readonly_seed',[False,True])
def test_tensor_pullback_retains_native_overflow_order_and_shared_seed(retained,readonly_seed):
    x=torch.tensor([1e308,1e-308,-0.,np.inf],dtype=torch.float64,requires_grad=True)
    array=np.array([1e-308,1e308,-0.,0.],dtype=np.float64)
    if readonly_seed:array.flags.writeable=False
    seed=torch.from_numpy(array);before=array.tobytes()
    with np.errstate(all='ignore'):
        expected=array*(2*x.detach().numpy())
        result=x.square()
        first,=torch.autograd.grad(result,x,seed,retain_graph=retained)
        exact(first.numpy(),expected)
        if retained:
            second,=torch.autograd.grad(result,x,seed)
            exact(second.numpy(),expected)
    assert array.tobytes()==before and x.grad is None


def test_tensor_raw_alias_is_read_at_backward_and_preserves_directional_derivatives():
    x=torch.tensor([.5,-.5],dtype=torch.float64,requires_grad=True)
    result=x.square();alias=x.detach().numpy();alias[:]=[.25,-.75];del alias
    result.sum().backward();exact(x.grad.numpy(),[.5,-1.5])
    direction=torch.tensor([2.,3.],dtype=torch.float64)
    _,tangent=torch.func.jvp(lambda a:a.square(),(x,),(direction,))
    exact(tangent.numpy(),[1.,-4.5])
    _,hvp=torch.autograd.functional.hvp(lambda a:a.square().sum(),x,direction)
    exact(hvp.numpy(),[4.,6.])


@pytest.mark.parametrize('seed',[0.,1.,-0.,np.inf,np.nan])
def test_uniform_seed_keeps_floating_error_behavior(seed):
    source=np.array([np.inf,-np.inf,np.nan,-0.,.5],dtype=np.float64)
    gradient=np.broadcast_to(np.asarray(seed,dtype=np.float64),source.shape)
    expected_error=None
    with np.errstate(all='raise'):
        try:expected=gradient*(2*source)
        except FloatingPointError as error:expected_error=type(error)
    with np.errstate(all='raise'):
        if expected_error:
            with pytest.raises(expected_error):_arithmetic.square_backward(source,gradient)
        else:exact(_arithmetic.square_backward(source,gradient),expected)


def test_scalar_loss_unity_seed_retains_nonfinite_and_zero_signs():
    array=np.array([np.inf,-np.inf,np.nan,-0.,.5],dtype=np.float64)
    x=torch.from_numpy(array).requires_grad_(True)
    with np.errstate(all='ignore'):
        x.square().sum().backward()
        expected=2*array
    exact(x.grad.numpy(),expected)
