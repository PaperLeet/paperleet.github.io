"""Independent values, index/gradient routing, module and lifetime contracts."""
import gc
import pickle
import weakref
import numpy as np
import pytest
from stridetorch import UnsupportedOperation, as_torch, torch


@pytest.fixture(params=['scan', 'window', 'block', 'auto'], autouse=True)
def pooling_algorithm(request, monkeypatch):
    from stridetorch import _pooling
    monkeypatch.setattr(_pooling, '_ALGORITHM', request.param)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.float16, np.int8, np.int16, np.int32, np.int64, np.uint8])
def test_exact_values_indices_and_aliases(dtype):
    x = torch.tensor([[[1, 2, 3], [4, 9, 6], [7, 8, 5]]], dtype=np.dtype(dtype))
    value, index = torch.nn.functional.max_pool2d(x, 2, 1, return_indices=True)
    np.testing.assert_array_equal(value.numpy(), np.full((1, 2, 2), 9, dtype=dtype))
    np.testing.assert_array_equal(index.numpy(), np.full((1, 2, 2), 4, dtype=np.int64))
    assert value.dtype == np.dtype(dtype) and index.dtype == torch.int64
    np.testing.assert_array_equal(torch.max_pool2d(x, 2, 1).numpy(), value.numpy())
    y, i = torch.nn.functional.max_pool2d_with_indices(x, 2, 1, return_indices=False)
    np.testing.assert_array_equal(y.numpy(), value.numpy());np.testing.assert_array_equal(i.numpy(), index.numpy())


@pytest.mark.parametrize('values,expected_index', [
    ([2., 2., 2., 2.], [[0, 0, 1], [0, 0, 1], [2, 2, 3]]),
    ([-np.inf] * 4, [[0, 0, 1], [0, 0, 1], [2, 2, 3]]),
    ([np.nan, 1., np.nan, 0.], [[0, 0, 1], [2, 2, 1], [2, 2, 3]]),
    ([-0., 0., 0., -0.], [[0, 0, 1], [0, 0, 1], [2, 2, 3]]),
])
def test_first_finite_tie_last_nan_and_padding(values, expected_index):
    x = torch.tensor(values, dtype=torch.float64).reshape(1, 1, 2, 2).requires_grad_()
    y, i = torch.nn.functional.max_pool2d(x, 2, 1, 1, return_indices=True)
    expected = np.asarray(expected_index, dtype=np.int64)
    np.testing.assert_array_equal(i.numpy()[0, 0], expected)
    selected = np.asarray(values)[expected]
    np.testing.assert_allclose(y.numpy()[0, 0], selected, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(np.signbit(y.numpy()[0, 0]), np.signbit(selected))
    y.backward(torch.ones_like(y))
    np.testing.assert_array_equal(x.grad.numpy().ravel(), np.bincount(expected.ravel(), minlength=4))


def test_overlapping_routes_retained_backwards_and_nonfinite_seeds():
    x = torch.tensor([[[[1., 2., 3.], [4., 9., 6.], [7., 8., 5.]]]], requires_grad=True)
    y = torch.nn.functional.max_pool2d(x, 2, 1)
    seed = torch.tensor([[[[1., 2.], [3., 4.]]]])
    y.backward(seed, retain_graph=True);y.backward(seed)
    np.testing.assert_array_equal(x.grad.numpy(), [[[[0., 0., 0.], [0., 20., 0.], [0., 0., 0.]]]])
    with pytest.raises(RuntimeError):y.backward(seed)
    x = torch.tensor([[[[1., 2.], [3., 4.]]]], requires_grad=True)
    torch.nn.functional.max_pool2d(x, 2).backward(torch.tensor([[[[-0.]]]]))
    assert not np.signbit(x.grad.numpy()).any()


@pytest.mark.parametrize('values,selected', [([2.,2.,2.,2.],0),([-np.inf]*4,0),([np.nan,1.,np.nan,0.],2),([-0.,0.,0.,-0.],0)])
def test_nonoverlap_block_preserves_ieee_values_and_selected_indices(values, selected):
    x=torch.tensor(values,dtype=torch.float64).reshape(1,1,2,2).requires_grad_()
    y,i=torch.nn.functional.max_pool2d(x,2,return_indices=True)
    np.testing.assert_array_equal(i.numpy(),selected)
    np.testing.assert_allclose(y.numpy(),values[selected],rtol=0,atol=0,equal_nan=True)
    np.testing.assert_array_equal(np.signbit(y.numpy()),np.signbit(values[selected]))
    np.testing.assert_allclose(torch.nn.functional.max_pool2d(x.detach(),2).numpy(),y.numpy(),rtol=0,atol=0,equal_nan=True)
    y.sum().backward();expected=np.zeros(4);expected[selected]=1
    np.testing.assert_array_equal(x.grad.numpy().ravel(),expected)


def test_ceil_tail_dilation_and_empty_batch():
    x = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4)
    y, i = torch.nn.functional.max_pool2d(x, (2, 3), (2, 2), (1, 1), ceil_mode=True, return_indices=True)
    np.testing.assert_array_equal(y.numpy(), [[[[1, 3, 3], [9, 11, 11]]]])
    np.testing.assert_array_equal(i.numpy(), [[[[1, 3, 3], [9, 11, 11]]]])
    x = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
    np.testing.assert_array_equal(torch.nn.functional.max_pool2d(x, 2, 1, dilation=2).numpy(), [[[[10, 11], [14, 15]]]])
    x = torch.empty(0, 3, 5, 7, dtype=torch.float64, requires_grad=True)
    y, i = torch.nn.functional.max_pool2d(x, 2, return_indices=True)
    assert y.shape == i.shape == (0, 3, 2, 3)
    y.sum().backward();assert x.grad.shape == x.shape


@pytest.mark.parametrize('target', ['input', 'indices', 'output'])
def test_saved_input_and_indices_versions_but_not_output(target):
    x = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3).requires_grad_()
    y, i = torch.nn.functional.max_pool2d(x, 2, 1, return_indices=True)
    with torch.no_grad():{'input':x, 'indices':i, 'output':y}[target].detach().add_(1)
    if target == 'output':y.sum().backward();assert x.grad.sum().item() == 4
    else:
        with pytest.raises(RuntimeError):y.sum().backward()


@pytest.mark.parametrize('target', ['none', 'input', 'tangent', 'indices', 'primal_output', 'tangent_output'])
def test_forward_ad_reverse_dependencies_and_saved_versions(target):
    x = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3).requires_grad_()
    v = torch.ones_like(x, requires_grad=True)
    (p, i), (t, ti) = torch.func.jvp(lambda z:torch.nn.functional.max_pool2d(z, 2, 1, return_indices=True), (x,), (v,))
    np.testing.assert_array_equal(t.numpy(), 1);np.testing.assert_array_equal(ti.numpy(), 0)
    assert ti.dtype == torch.int64
    with torch.no_grad():
        if target != 'none':{'input':x, 'tangent':v, 'indices':i, 'primal_output':p, 'tangent_output':t}[target].detach().add_(1)
    if target in ('tangent', 'indices'):
        with pytest.raises(RuntimeError):t.sum().backward()
    else:
        t.sum().backward();assert x.grad is None
        np.testing.assert_array_equal(v.grad.numpy(), [[[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]])


def test_module_imports_state_hooks_and_lifetimes():
    with as_torch():
        from torch.nn.modules.pooling import MaxPool2d
        from torch.nn.functional import max_pool2d
        model = MaxPool2d(2, return_indices=True)
        assert isinstance(model, torch.nn.Module) and not model.state_dict()
        assert repr(model) == 'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)'
        assert model.eval() is model and not model.training
        x = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
        np.testing.assert_array_equal(model(x)[0].numpy(), max_pool2d(x, 2).numpy())
        restored=pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored(x)[0].numpy(),model(x)[0].numpy())
        calls=[];handle=model.register_forward_hook(lambda module,args,output:calls.append(output[0].shape))
        model(x);handle.remove();model(x);assert calls==[(1,1,2,2)]
    def build():
        x = torch.ones(1, 1, 4, 4, requires_grad=True);reference = weakref.ref(x)
        return torch.nn.MaxPool2d(2)(x), reference
    y, reference = build();gc.collect();assert reference() is not None
    y.sum().backward();del y;gc.collect();assert reference() is None


def test_workspace_fallback_and_inference_only_dilated_holes(monkeypatch):
    from stridetorch import _pooling
    monkeypatch.setattr(_pooling, '_WINDOW_WORKSPACE_BYTES', 0)
    x = torch.arange(16).reshape(1, 1, 4, 4)
    np.testing.assert_array_equal(torch.nn.MaxPool2d(2)(x).numpy(), [[[[5, 7], [13, 15]]]])
    x = torch.ones(1, 1, 1, 1)
    y, i = torch.nn.functional.max_pool2d(x, 2, dilation=2, padding=1, return_indices=True)
    np.testing.assert_array_equal(y.numpy(), -np.inf);np.testing.assert_array_equal(i.numpy(), 2)
    with pytest.raises(UnsupportedOperation):
        torch.nn.functional.max_pool2d(x.requires_grad_(), 2, dilation=2, padding=1)


@pytest.mark.parametrize('kw,error', [
    ({'kernel_size':True}, TypeError), ({'kernel_size':(2, 2, 2)}, RuntimeError),
    ({'kernel_size':0}, RuntimeError), ({'kernel_size':2, 'stride':0}, RuntimeError),
    ({'kernel_size':2, 'padding':2}, RuntimeError), ({'kernel_size':2, 'dilation':0}, RuntimeError),
    ({'kernel_size':2, 'ceil_mode':1}, TypeError), ({'kernel_size':2, 'padding':True}, TypeError),
])
def test_invalid_arguments(kw, error):
    with pytest.raises(error):torch.nn.functional.max_pool2d(torch.ones(1, 1, 4, 4), **kw)
