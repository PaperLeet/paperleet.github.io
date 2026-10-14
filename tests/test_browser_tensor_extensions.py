"""Independent checks for the helper APIs used by spectral and recurrent lessons."""
import numpy as np
import pytest
import subprocess
import sys
import torch
import torch.nn.functional as F

from paperleet.browser import numpy_autograd as wasm


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('dim,keepdim',[(None,False),(0,False),(0,True),(1,False),(1,True),(-1,False),(-1,True)])
def test_zero_safe_product_and_maximum_tie_gradients(dtype,dim,keepdim):
    data=np.array([[0.,2.,-1.],[0.,0.,0.],[2.,2.,1.],[1.,3.,-2.]],dtype=dtype).T
    for operation in ('prod','amax'):
        a,b=wasm.Tensor(data.copy(order='K'),requires_grad=True),torch.tensor(data,requires_grad=True)
        if operation=='prod' and dim is None:
            actual,expected=a.prod(),b.prod()
        else:
            actual=getattr(a,operation)(dim=dim,keepdim=keepdim)
            expected=getattr(b,operation)(dim=dim,keepdim=keepdim)
        np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),atol=1e-6)
        weights=np.arange(actual.numel(),dtype=dtype).reshape(actual.shape)*.2+1
        (actual*wasm.Tensor(weights)).sum().backward();(expected*torch.tensor(weights)).sum().backward()
        np.testing.assert_allclose(a.grad.numpy(),b.grad.numpy(),atol=2e-6)


def test_product_empty_scalar_integer_promotion_and_maximum_multi_axes():
    for data in (np.empty((2,0)),np.array(0.),np.array(-3.)):
        a,b=wasm.Tensor(data.copy(),requires_grad=True),torch.tensor(data,requires_grad=True)
        a.prod().backward();b.prod().backward()
        np.testing.assert_allclose(a.grad.numpy(),b.grad.numpy())
    for dtype in (np.int32,np.int64,np.bool_):
        a=wasm.Tensor([[1,1],[1,0]],dtype=dtype)
        assert a.prod(0).dtype==np.dtype('int64')
    data=np.arange(24.).reshape(2,3,4)
    a,b=wasm.Tensor(data.copy(),requires_grad=True),torch.tensor(data,requires_grad=True)
    a.amax((0,2)).sum().backward();b.amax((0,2)).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(),b.grad.numpy())


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_round_is_ties_to_even_with_zero_derivative_and_preserved_dtype(dtype):
    data=np.array([-.5,.5,1.5,2.5,-2.5,3.2],dtype=dtype)
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    actual=a.round();expected=b.round()
    assert actual.dtype==np.dtype(dtype)
    np.testing.assert_array_equal(actual.numpy(),expected.detach().numpy())
    actual.sum().backward();expected.sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(),b.grad.numpy())
    integers=np.array([2**62+1,-2**62-1],dtype=np.int64)
    np.testing.assert_array_equal(wasm.Tensor(integers).round().numpy(),integers)
    with pytest.raises(wasm.BrowserUnsupported): a.round(decimals=2)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('beta,threshold',[(1.,20.),(.5,2.),(3.,10.)])
def test_softplus_values_and_gradients_match_native_stability_branches(dtype,beta,threshold):
    data=np.array([-1000.,-20.,-1.,0.,1.,20.,1000.],dtype=dtype)
    a,b=wasm.Tensor(data,requires_grad=True),torch.tensor(data,requires_grad=True)
    actual,expected=wasm.softplus(a,beta,threshold),F.softplus(b,beta,threshold)
    np.testing.assert_allclose(actual.numpy(),expected.detach().numpy(),rtol=2e-6,atol=2e-6)
    actual.sum().backward();expected.sum().backward()
    np.testing.assert_allclose(a.grad.numpy(),b.grad.numpy(),atol=2e-6)


@pytest.mark.parametrize('size,chunks',[(6,3),(5,4),(2,5),(0,3)])
def test_chunk_count_shapes_strides_and_gradient_accumulation(size,chunks):
    data=np.arange(2*size,dtype=np.float64).reshape(2,size).T
    a,b=wasm.Tensor(data.copy(order='K'),requires_grad=True),torch.tensor(data,requires_grad=True)
    actual,expected=a.chunk(chunks),b.chunk(chunks)
    assert len(actual)==len(expected)
    for x,y in zip(actual,expected): np.testing.assert_array_equal(x.numpy(),y.detach().numpy())
    sum(x.square().sum() for x in actual).backward();sum(x.square().sum() for x in expected).backward()
    np.testing.assert_array_equal(a.grad.numpy(),b.grad.numpy())


def test_new_factories_inherit_dtype_and_readonly_metadata_without_inheriting_grad():
    for dtype in (np.float32,np.float64,np.int64):
        x=wasm.Tensor(np.ones((2,3),dtype=dtype),requires_grad=dtype!=np.int64)
        empty=x.new_empty(2,0,3);full=x.new_full((2,3),-4)
        assert empty.shape==(2,0,3) and empty.dtype==full.dtype==x.dtype
        assert not empty.requires_grad and not full.requires_grad
        assert x.is_floating_point()==(dtype!=np.int64)
        np.testing.assert_array_equal(full.numpy(),np.full((2,3),-4,dtype=dtype))
        expected=torch.tensor(np.ones((2,3),dtype=dtype))
        for name in ('new_zeros','new_ones','new_empty'):
            assert getattr(x,name)((2,3),dtype=None).dtype==np.dtype(dtype)
            assert getattr(expected,name)((2,3),dtype=None).numpy().dtype==np.dtype(dtype)
        assert x.new_full((2,3),-4,dtype=None).dtype==expected.new_full((2,3),-4,dtype=None).numpy().dtype
        strided=np.ones((3,2),dtype=dtype).T
        actual=wasm.full_like(wasm.Tensor(strided),fill_value=2,dtype=None)
        native=torch.full_like(torch.tensor(strided),fill_value=2,dtype=None)
        assert actual.dtype==native.numpy().dtype and actual.is_contiguous()==native.is_contiguous()
        np.testing.assert_array_equal(actual.numpy(),native.numpy())


def test_installed_like_factories_preserve_explicit_none_dtype_and_grad_flags():
    source='''
from paperleet.browser.numpy_autograd import install
torch=install()
for dtype in (torch.float32,torch.float64,torch.int64):
    x=torch.ones(2,3,dtype=dtype,requires_grad=dtype!=torch.int64)
    for factory in (torch.zeros_like,torch.ones_like,torch.empty_like):
        value=factory(x,dtype=None)
        assert value.dtype==dtype and not value.requires_grad
    value=torch.full_like(x,2,dtype=None)
    assert value.dtype==dtype and not value.requires_grad and value.tolist()==[[2,2,2],[2,2,2]]
x=torch.ones(2,3,dtype=torch.float32)
assert torch.full_like(x,2,dtype=torch.float64,requires_grad=True).requires_grad
assert torch.zeros_like(x,dtype=torch.float64).dtype==torch.float64
'''
    result=subprocess.run([sys.executable,'-c',source],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
