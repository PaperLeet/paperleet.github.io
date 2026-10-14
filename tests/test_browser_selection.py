"""Independent native parity for selection, routing masks, and their VJPs."""
import numpy as np
import operator
import pytest
import subprocess
import sys
import torch

from paperleet.browser import numpy_autograd as wasm


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('dim', [0, 1, -1])
@pytest.mark.parametrize('largest', [False, True])
@pytest.mark.parametrize('k', [0, 1, 2])
def test_topk_strided_values_indices_and_analytical_gradients(dtype, dim, largest, k):
    data = np.arange(24, dtype=dtype).reshape(2,3,4).transpose(2,1,0)
    a, b = wasm.Tensor(data.copy(order='K'),requires_grad=True), torch.tensor(data,requires_grad=True)
    actual, expected = a.topk(k,dim,largest), b.topk(k,dim,largest)
    np.testing.assert_array_equal(actual.indices.numpy(), expected.indices.numpy())
    np.testing.assert_array_equal(actual.values.numpy(), expected.values.detach().numpy())
    assert actual.indices.dtype == np.dtype('int64') and not actual.indices.requires_grad
    weights = np.arange(actual.values.numel(),dtype=dtype).reshape(actual.values.shape) * .13
    (actual.values*wasm.Tensor(weights)).sum().backward()
    (expected.values*torch.tensor(weights)).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), atol=2e-6)
    np.testing.assert_array_equal(a.numpy(), data)


@pytest.mark.parametrize('dtype', [np.uint8, np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('largest', [False, True])
def test_topk_unsorted_flag_and_full_axis_preserve_selected_values(dtype, largest):
    data = np.array([[8,3,7,1,9],[2,6,0,5,4]],dtype=dtype)
    for k in (0,3,5):
        actual, expected = wasm.Tensor(data).topk(k,largest=largest,sorted=False), torch.tensor(data).topk(k,largest=largest,sorted=False)
        np.testing.assert_array_equal(np.sort(actual.values.numpy(),axis=-1),np.sort(expected.values.numpy(),axis=-1))
        np.testing.assert_array_equal(actual.values.numpy(),np.take_along_axis(data,actual.indices.numpy(),axis=-1))


@pytest.mark.parametrize('largest', [False, True])
def test_topk_ties_and_nan_use_returned_indices_without_assuming_tie_identity(largest):
    data = np.array([[2.,2.,0.,2.],[np.nan,1.,np.nan,-np.inf]])
    a = wasm.Tensor(data.copy(),requires_grad=True)
    actual, expected = a.topk(3,largest=largest), torch.tensor(data).topk(3,largest=largest)
    np.testing.assert_allclose(actual.values.numpy(),expected.values.numpy(),equal_nan=True)
    np.testing.assert_allclose(actual.values.numpy(),np.take_along_axis(data,actual.indices.numpy(),axis=-1),equal_nan=True)
    assert all(len(set(row))==3 for row in actual.indices.tolist())
    actual.values.sum().backward()
    expected_grad = np.zeros_like(data); np.put_along_axis(expected_grad,actual.indices.numpy(),1.,axis=-1)
    np.testing.assert_array_equal(a.grad.numpy(),expected_grad)


def test_topk_scalar_empty_axes_output_copies_and_validation():
    for k in (0,1):
        a, b = wasm.Tensor(2.,requires_grad=True), torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        actual, expected = a.topk(k), b.topk(k)
        assert actual.values.shape==expected.values.shape==() and actual.indices.item()==0
        actual.values.backward(); expected.values.backward()
        assert a.grad.item()==b.grad.item()==1
    empty = wasm.Tensor(np.empty((2,0)),requires_grad=True)
    empty.topk(0).values.sum().backward()
    assert empty.grad.shape == (2,0)
    source = wasm.Tensor([1.,3.,2.])
    selected = source.topk(2)
    selected.values[...] = -10.
    np.testing.assert_array_equal(source.numpy(),[1.,3.,2.])
    for k in (-1,4):
        with pytest.raises(RuntimeError): source.topk(k)
    for k in (True,1.5):
        with pytest.raises(TypeError): source.topk(k)
    with pytest.raises(IndexError): source.topk(1,dim=2)
    with pytest.raises(wasm.BrowserUnsupported): source.topk(1,out=(source,source))


@pytest.mark.parametrize('dim', [0,1,-1])
@pytest.mark.parametrize('dtype', [np.float32,np.float64])
def test_scatter_tensor_source_partial_rows_strides_and_both_gradients(dim,dtype):
    data = np.arange(12,dtype=dtype).reshape(4,3).T
    indices = np.array([[0,2],[1,0]],dtype=np.int64)
    values = np.array([[.2,.4],[.7,.8]],dtype=dtype).T
    a, s = wasm.Tensor(data.copy(order='K'),requires_grad=True), wasm.Tensor(values.copy(order='K'),requires_grad=True)
    b, t = torch.tensor(data,requires_grad=True), torch.tensor(values,requires_grad=True)
    actual, expected = a.scatter(dim,wasm.Tensor(indices),s), b.scatter(dim,torch.tensor(indices),t)
    np.testing.assert_array_equal(actual.numpy(),expected.detach().numpy())
    weights = np.arange(12,dtype=dtype).reshape(3,4) * .17
    (actual*wasm.Tensor(weights)).sum().backward(); (expected*torch.tensor(weights)).sum().backward()
    for actual,expected in ((a,b),(s,t)):
        np.testing.assert_allclose(actual.grad.numpy(),expected.grad.numpy(),atol=2e-6)
    np.testing.assert_array_equal(a.numpy(),data);np.testing.assert_array_equal(s.numpy(),values)


@pytest.mark.parametrize('dtype', [np.bool_,np.uint8,np.int64,np.float32,np.float64])
def test_scatter_scalar_duplicate_destinations_and_bool_mask(dtype):
    data = np.ones((2,4),dtype=dtype)
    index = np.array([[1,1,3],[0,2,2]],dtype=np.int64)
    value = False if dtype==np.bool_ else 3
    a, b = wasm.Tensor(data.copy()), torch.tensor(data)
    np.testing.assert_array_equal(a.scatter(-1,wasm.Tensor(index),value).numpy(),b.scatter(-1,torch.tensor(index),value).numpy())
    np.testing.assert_array_equal(a.numpy(),data)


def test_scatter_scalar_input_gradients_repeated_inputs_and_empty_indices():
    data = np.arange(8,dtype=np.float64).reshape(2,4)
    a, b = wasm.Tensor(data.copy(),requires_grad=True), torch.tensor(data,requires_grad=True)
    indices = np.array([[1,1,3],[0,2,2]],dtype=np.int64)
    a.scatter(1,wasm.Tensor(indices),.5).square().sum().backward()
    b.scatter(1,torch.tensor(indices),.5).square().sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(),b.grad.numpy())
    # The same leaf can supply both the preserved input and the written values.
    a.grad=None;b.grad=None
    permutation=np.array([[3,1,0,2],[2,0,3,1]],dtype=np.int64)
    a.scatter(1,wasm.Tensor(permutation),a).square().sum().backward()
    b.scatter(1,torch.tensor(permutation),b).square().sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(),b.grad.numpy())
    empty=wasm.Tensor(np.empty((2,0)),requires_grad=True)
    result=a.scatter(1,wasm.Tensor(np.empty((2,0),dtype=np.int64)),empty)
    da,ds=wasm.autograd_grad(result.sum(),(a,empty))
    np.testing.assert_array_equal(da.numpy(),np.ones((2,4)))
    assert ds.shape==(2,0)


def test_scatter_validation_and_modified_selection_indices_fail_explicitly():
    a=wasm.Tensor(np.ones((2,3)),requires_grad=True)
    index=wasm.Tensor([[0,1],[1,2]],dtype=np.int64)
    src=wasm.Tensor(np.zeros((2,2)),requires_grad=True)
    for idx,source in [(wasm.Tensor([[0,0],[1,1]],dtype=np.int64),src),(index,a),(index.float(),src)]:
        with pytest.raises(wasm.BrowserUnsupported): a.scatter(1,idx,source)
    with pytest.raises(IndexError): a.scatter(1,wasm.Tensor([[0,3],[1,2]],dtype=np.int64),src)
    with pytest.raises(wasm.BrowserUnsupported): a.scatter(1,index,src,reduce='add')
    output=a.scatter(1,index,src)
    index[...] = 0
    with pytest.raises(wasm.BrowserUnsupported,match='saved for backward'): output.sum().backward()
    assert a.grad is None and src.grad is None
    chosen=a.topk(2)
    chosen.indices[...] = 0
    with pytest.raises(wasm.BrowserUnsupported,match='saved for backward'): chosen.values.sum().backward()


@pytest.mark.parametrize('dtype', [np.float32,np.float64,np.int64,np.bool_])
def test_count_nonzero_uses_int64_and_disconnects_gradients(dtype):
    data=np.array([[0,1,0],[2,0,3]],dtype=dtype).T
    for dim in (None,0,-1,(0,1)):
        a=wasm.Tensor(data,requires_grad=dtype in (np.float32,np.float64))
        actual,expected=wasm.count_nonzero(a,dim),torch.count_nonzero(torch.tensor(data),dim=dim)
        np.testing.assert_array_equal(actual.numpy(),expected.numpy())
        assert actual.dtype==np.dtype('int64') and not actual.requires_grad
    for values in (np.empty((2,0),dtype=dtype),np.array(0,dtype=dtype)):
        assert wasm.count_nonzero(wasm.Tensor(values)).item()==0


def test_integer_index_protocol_matches_native_for_range_and_scalar_indices():
    for data in (np.array(3),np.array([3]),np.array([[3]]),np.array(True)):
        assert operator.index(wasm.Tensor(data))==operator.index(torch.tensor(data))
    assert list(range(wasm.Tensor(3,dtype=np.int64)))==[0,1,2]
    for data in (np.array(3.),np.array([1,2]),np.array([],dtype=np.int64)):
        with pytest.raises(TypeError): operator.index(wasm.Tensor(data))
        with pytest.raises(TypeError): operator.index(torch.tensor(data))


def test_installed_functional_selection_bindings_in_an_isolated_runtime():
    script = '''
from paperleet.browser.numpy_autograd import install
torch=install()
x=torch.tensor([[0.,2.,0.],[3.,0.,4.]],requires_grad=True)
assert torch.count_nonzero(x,(0,1)).dtype==torch.int64
assert x.count_nonzero(-1).tolist()==[1,2]
selected=torch.topk(x,2,dim=-1)
mask=torch.scatter(torch.zeros_like(x,dtype=torch.bool),-1,selected.indices,True)
assert mask.dtype==torch.bool and mask.sum().item()==4
selected.values.sum().backward()
assert torch.equal(x.grad,mask.float())
assert list(range(torch.tensor(3)))==[0,1,2]
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
