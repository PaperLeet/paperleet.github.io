"""Independent native checks for quantization, normalization and gradcheck APIs."""
import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


def pair(data, dtype=np.float64):
    array = np.asarray(data, dtype=dtype)
    return wasm.Tensor(array.copy(), requires_grad=True), torch.tensor(array.copy(), requires_grad=True)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(2, 3, 4), (2, 0, 4), (3, 2, 4)])
@pytest.mark.parametrize('weighted', [False, True])
def test_rmsnorm_independent_values_and_input_scale_vjps(dtype, shape, weighted):
    rng = np.random.RandomState(32)
    data = rng.randn(*shape).astype(dtype)
    if data.size: data[0] = 0
    x, native = pair(data, dtype)
    weight, native_weight = pair(rng.randn(shape[-1]), dtype)
    actual = wasm.rms_norm_oracle(x, (shape[-1],), weight if weighted else None, eps=1e-4)
    expected = torch.nn.functional.rms_norm(native, (shape[-1],), native_weight if weighted else None, eps=1e-4)
    upstream = rng.randn(*shape).astype(dtype)
    (actual * wasm.Tensor(upstream)).sum().backward(); (expected * torch.tensor(upstream)).sum().backward()
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    for a, b in [(actual, expected), (x.grad, native.grad)]:
        np.testing.assert_allclose(a.numpy(), b.detach().numpy(), rtol=tolerance, atol=tolerance)
    if weighted: np.testing.assert_allclose(weight.grad.numpy(), native_weight.grad.numpy(), rtol=tolerance, atol=tolerance)


def test_new_tensor_inherits_precision_copies_and_detaches():
    template = wasm.Tensor([0.], dtype=np.float64)
    value = 1 + 2**-40
    created = template.new_tensor([value])
    assert created.item() == torch.zeros(1, dtype=torch.float64).new_tensor([value]).item() == value
    source, _ = pair([2., 3.])
    detached = template.new_tensor(source)
    assert not detached.requires_grad and not np.shares_memory(source.numpy(), detached.numpy())
    with wasm.no_grad(): leaf = template.new_tensor(source, requires_grad=True)
    leaf.sum().backward()
    assert source.grad is None
    np.testing.assert_array_equal(leaf.grad.numpy(), [1., 1.])
    codes = template.new_tensor([0, 7, 15], dtype=np.uint8)
    assert codes.dtype == np.dtype('uint8') and codes.tolist() == [0, 7, 15]


@pytest.mark.parametrize('dim', [None, 0, -1])
@pytest.mark.parametrize('keepdim', [False, True])
def test_arg_extrema_first_ties_nan_indices_and_int64(dim, keepdim):
    data = np.asarray([[2., 2., 0.], [np.nan, 3., np.nan]])
    x, native = wasm.Tensor(data), torch.tensor(data)
    for name in ('argmin', 'argmax'):
        actual, expected = getattr(x, name)(dim, keepdim), getattr(native, name)(dim, keepdim)
        assert actual.dtype == np.dtype('int64')
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize('name,args', [('flip', ((0, -1),)), ('flip', ((),)), ('roll', ((1, -2), (0, -1))), ('roll', (3,)), ('triu', (1,)), ('tril', (-1,))])
def test_index_transforms_values_gradients_and_copy_semantics(name, args):
    x, native = pair(np.arange(24.).reshape(2, 3, 4))
    value, expected = getattr(x, name)(*args), getattr(native, name)(*args)
    upstream = np.arange(24.).reshape(value.shape) * .1
    (value * wasm.Tensor(upstream)).sum().backward(); (expected * torch.tensor(upstream)).sum().backward()
    np.testing.assert_array_equal(value.numpy(), expected.detach().numpy())
    np.testing.assert_allclose(x.grad.numpy(), native.grad.numpy())
    assert not np.shares_memory(value.numpy(), x.numpy()), 'These transforms copy rather than alias the original storage'


@pytest.mark.parametrize('shape', [(3, 7), (1, 7), (7,)])
@pytest.mark.parametrize('correction', [0, 1, 2])
def test_covariance_independent_diagnostics_and_vjps(shape, correction):
    data = np.random.RandomState(51).randn(*shape)
    x, native = pair(data)
    value, expected = wasm.cov(x, correction=correction), torch.cov(native, correction=correction)
    upstream = np.random.RandomState(13).randn(*value.shape) if value.shape else np.asarray(.7)
    (value * wasm.Tensor(upstream)).sum().backward(); (expected * torch.tensor(upstream)).sum().backward()
    np.testing.assert_allclose(value.numpy(), expected.detach().numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x.grad.numpy(), native.grad.numpy(), rtol=1e-12, atol=1e-12)


def test_full_numerical_jacobian_check_agrees_with_native_and_preserves_inputs():
    x, native_x = pair([-.7, .2, 1.3]); y, native_y = pair([.3, -.8, 1.])
    x.grad = wasm.Tensor([9., 9., 9.]); before = x.numpy().copy()
    function = lambda a,b: ((a.sin() * b).sum(), (a * b).square())
    assert wasm.gradcheck(function, (x, y))
    assert torch.autograd.gradcheck(function, (native_x, native_y))
    np.testing.assert_array_equal(x.numpy(), before)
    np.testing.assert_array_equal(x.grad.numpy(), [9., 9., 9.])
    assert y.grad is None


def test_gradcheck_rejects_wrong_derivatives_and_detached_value_dependence():
    x, native = pair([-.7, .2, 1.3])
    wrong = lambda a: wasm.result(a.numpy() ** 2, (a, lambda g: 2 * g))
    assert not wasm.gradcheck(wrong, x, raise_exception=False)
    with pytest.raises(RuntimeError, match='Jacobian mismatch'): wasm.gradcheck(wrong, x)
    assert not wasm.gradcheck(lambda a: a.detach().square(), x, raise_exception=False)
    class Wrong(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a): return a.square()
        @staticmethod
        def backward(ctx, g): return 2 * g
    assert not torch.autograd.gradcheck(Wrong.apply, native, raise_exception=False)


def test_gradcheck_and_inplace_unsupported_paths_fail_explicitly():
    x, _ = pair([.2, .8])
    for inputs, options in [((x,), {'fast_mode': True}), ((x.float(),), {}), ((x, x), {}), ((x.expand(3,2),), {}), ((wasm.Tensor(np.zeros(257), requires_grad=True),), {})]:
        with pytest.raises(wasm.BrowserUnsupported): wasm.gradcheck(lambda *a: a[0].square(), inputs, **options)
    for name in ('__iadd__', '__isub__', '__imul__', '__itruediv__'):
        with wasm.no_grad(), pytest.raises(wasm.BrowserUnsupported): getattr(x, name)(1.)
    constant = wasm.Tensor([1., 2.]); alias = constant.detach()
    constant += 3
    np.testing.assert_array_equal(alias.numpy(), [4., 5.])
    with pytest.raises(RuntimeError): wasm.Tensor([1,2]).__itruediv__(2)
