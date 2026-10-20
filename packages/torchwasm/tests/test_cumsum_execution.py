import contextlib
import numpy as np
import pytest
from torchwasm import torch


@pytest.mark.parametrize('dtype',['float16','float32','float64','complex64','complex128'])
@pytest.mark.parametrize('length',[63,64,65])
@pytest.mark.parametrize('mode',['inactive','no_grad'])
def test_zero_signs_owned_outputs_and_inactive_graphs(dtype,length,mode):
    raw=np.full(length,complex(-0.,-0.) if dtype.startswith('complex') else -0.,dtype=dtype)
    x=torch.from_numpy(raw).requires_grad_(mode=='no_grad' and not dtype.startswith('complex'))
    active=torch.is_grad_enabled()
    with torch.no_grad() if mode=='no_grad' else contextlib.nullcontext():
        result=x.cumsum(0)
    array=result.numpy()
    assert result.shape==(length,) and result.stride()==(1,)
    assert not result.requires_grad and x.grad is None
    assert not np.shares_memory(array,raw)
    np.testing.assert_array_equal(array,np.zeros(length,dtype=dtype))
    assert not np.signbit(array.real).any()
    if array.dtype.kind=='c':assert not np.signbit(array.imag).any()
    assert torch.is_grad_enabled()==active


@pytest.mark.parametrize('dtype',['float16','float32','float64','complex64','complex128'])
@pytest.mark.parametrize('length',[63,64,65])
def test_caller_error_state_is_restored_after_overflow(dtype,length):
    real_dtype=np.empty((),dtype=dtype).real.dtype
    maximum=np.finfo(real_dtype).max
    raw=np.full(length,maximum,dtype=dtype)
    expected=np.full(length,complex(float('inf'),0.) if dtype.startswith('complex') else float('inf'),dtype=dtype)
    expected[0]=maximum
    before=np.geterr()
    with np.errstate(all='raise'):
        result=torch.from_numpy(raw).cumsum(0)
        assert set(np.geterr().values())=={'raise'}
    assert np.geterr()==before
    np.testing.assert_array_equal(result.numpy(),expected)


@pytest.mark.parametrize('source,target,scale',[
    ('float32','float32',1e8),('float64','float32',1e8),
    ('float64','float64',1e8),('float32','float64',1e8),
    ('float16','float16',1e4),('float32','float16',1e4)])
@pytest.mark.parametrize('length',[63,64,65])
@pytest.mark.parametrize('strided',[False,True])
def test_quantization_precedes_scan_and_reverse_input_cast(source,target,scale,length,strided):
    first=scale+1 if (source,target)==('float64','float32') else scale
    raw=np.resize(np.array([first,1.,-scale,0.],dtype=source),length)
    if strided:
        storage=np.empty(length*2,dtype=source);storage[::2]=raw;raw=storage[::2]
    x=torch.from_numpy(raw).requires_grad_()
    y=x.cumsum(0,dtype=getattr(torch,target))
    indices=np.arange(length);cycle=indices//4;position=indices%4
    expected=np.where(position==0,scale+cycle,np.where(position==1,scale+cycle+1,cycle+1)).astype(target)
    np.testing.assert_array_equal(y.numpy(),expected)
    seed=np.resize(np.array([.125,-.25,.5,-.375],dtype=target),length)
    gradient,=torch.autograd.grad(y,x,torch.from_numpy(seed))
    triangular=np.arange(length)[None,:]<=np.arange(length)[:,None]
    expected_gradient=(triangular.T@seed.astype(source).astype('float64')).astype(source)
    np.testing.assert_array_equal(gradient.numpy(),expected_gradient)
    assert x.grad is None


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('length',[63,64,65])
def test_composed_hessian_with_independent_triangular_operator(dtype,length):
    data=np.resize(np.array([.125,-.25,.5,-.375],dtype=dtype),length)
    x=torch.from_numpy(data).requires_grad_();direction=torch.from_numpy(data.copy())
    value,product=torch.autograd.functional.hvp(lambda a:a.cumsum(0).square().sum(),x,direction)
    triangular=(np.arange(length)[None,:]<=np.arange(length)[:,None]).astype('float64')
    prefix=triangular@data.astype('float64')
    expected=2*triangular.T@prefix
    np.testing.assert_array_equal(product.numpy(),expected.astype(dtype))
    np.testing.assert_array_equal(value.numpy(),np.array(prefix@prefix,dtype=dtype))
    assert x.grad is None
