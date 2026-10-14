"""Independent native parity for APIs used by the remaining CPU/WASM lessons."""
import operator

import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('divisor', [3, -3, 2.5])
def test_floor_division_signed_broadcast_and_reverse_match_native(dtype, divisor):
    data = np.array([[-7, -1, 0], [1, 7, 9]], dtype=dtype).T
    a, b = wasm.Tensor(data), torch.tensor(data)
    for actual, expected in [(a // divisor, b // divisor), (divisor // (a + 10), divisor // (b + 10)),
                             (a // wasm.Tensor(np.array([2, -2], dtype=dtype)), b // torch.tensor([2, -2], dtype=b.dtype))]:
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        assert actual.dtype == expected.numpy().dtype


def test_floor_division_zero_overflow_and_unsupported_backward():
    with pytest.raises(RuntimeError): wasm.Tensor(np.array([2], dtype=np.int64)) // 0
    with pytest.raises(wasm.BrowserUnsupported): wasm.Tensor(np.array([np.iinfo(np.int64).min])) // -1
    a, b = wasm.Tensor(np.array([-1., 0., 1.])), torch.tensor([-1., 0., 1.], dtype=torch.float64)
    np.testing.assert_equal((a // 0).numpy(), (b // 0).numpy())
    with pytest.raises(wasm.BrowserUnsupported): wasm.Tensor(np.array([2.]), requires_grad=True) // 2
    with wasm.no_grad():
        assert not (wasm.Tensor(np.array([2.]), requires_grad=True) // 2).requires_grad


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32])
@pytest.mark.parametrize('shape', [(3, 5), (5, 3), (0, 4), (4, 0)])
def test_trace_strided_rectangular_empty_values_and_vjp(dtype, shape):
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape).T
    floating = np.dtype(dtype).kind == 'f'
    a, b = wasm.Tensor(data, requires_grad=floating), torch.tensor(data, requires_grad=floating)
    actual, expected = a.trace(), b.trace()
    np.testing.assert_equal(actual.numpy(), expected.detach().numpy())
    assert actual.dtype == expected.detach().numpy().dtype
    if floating:
        (actual * 1.7).backward(); (expected * 1.7).backward()
        np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy())
    with pytest.raises(RuntimeError): wasm.Tensor(np.ones(3)).trace()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_relu_module_values_and_analytical_vjp_match_native(dtype):
    data = np.array([-np.inf, -2., -0., 0., .4, np.inf, np.nan], dtype=dtype)
    a, b = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    actual, expected = wasm.ReLU()(a), torch.nn.ReLU()(b)
    actual.sum().backward(); expected.sum().backward()
    np.testing.assert_equal(actual.numpy(), expected.detach().numpy())
    np.testing.assert_equal(a.grad.numpy(), b.grad.numpy())
    assert list(wasm.ReLU().parameters()) == []
    with pytest.raises(wasm.BrowserUnsupported): wasm.ReLU(inplace=True)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape,dim,order', [((), None, None), ((3,), None, 2), ((2, 3), None, None),
    ((2, 3), None, 'fro'), ((2, 3, 4), None, None), ((2, 3, 4), -1, 2),
    ((2, 3, 4), (0, 2), 'fro'), ((2, 3, 4), (0, 2), None), ((2, 0, 4), 1, None)])
@pytest.mark.parametrize('keepdim', [False, True])
def test_linalg_euclidean_norm_values_zero_inputs_and_vjp_match_native(dtype, shape, dim, order, keepdim):
    for data in [np.random.RandomState(97).randn(*shape).astype(dtype) if shape else np.array(-.7, dtype=dtype), np.zeros(shape, dtype=dtype)]:
        a, b = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
        actual = wasm.linalg_norm(a, ord=order, dim=dim, keepdim=keepdim)
        expected = torch.linalg.norm(b, ord=order, dim=dim, keepdim=keepdim)
        actual.sum().backward(); expected.sum().backward()
        for got, want in [(actual, expected.detach()), (a.grad, b.grad)]:
            np.testing.assert_allclose(got.numpy(), want.numpy(), atol=2e-6 if dtype == np.float32 else 1e-12, rtol=2e-6)


def test_linalg_norm_orders_do_not_confuse_spectral_and_frobenius_norms():
    x = wasm.Tensor(np.array([[3., 0.], [0., 4.]]))
    assert float(wasm.linalg_norm(x)) == 5
    with pytest.raises(wasm.BrowserUnsupported): wasm.linalg_norm(x, ord=2)
    with pytest.raises(wasm.BrowserUnsupported): wasm.linalg_norm(x, ord=1)
    with pytest.raises(wasm.BrowserUnsupported): wasm.linalg_norm(x, dtype=np.float32)
    for dim in [(), (0, 0), (0, 1, 2)]:
        with pytest.raises((RuntimeError, IndexError, wasm.BrowserUnsupported)): wasm.linalg_norm(x, dim=dim)


@pytest.mark.parametrize('dtype', [np.int64, np.int32, np.uint8, np.float32, np.float64])
def test_randint_overloads_and_generator_stream_match_native(dtype):
    a, b = wasm.Generator().manual_seed(103), torch.Generator().manual_seed(103)
    native_dtype = {np.int64: torch.int64, np.int32: torch.int32, np.uint8: torch.uint8, np.float32: torch.float32, np.float64: torch.float64}[dtype]
    for args, kwargs in [((7, (3, 4)), {}), ((2, 7, (3, 4)), {}), ((), {'high': 7, 'size': ()}),
                         ((7,), {'size': (0, 2)}), ((), {'low': 1, 'high': 7, 'size': [2, 3]})]:
        actual = wasm.randint(*args, **kwargs, generator=a, dtype=dtype)
        expected = torch.randint(*args, **kwargs, generator=b, dtype=native_dtype)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        assert actual.dtype == expected.numpy().dtype and not actual.requires_grad
    with pytest.raises(RuntimeError): wasm.randint(-1, 300, (2,), dtype=np.uint8)
    with pytest.raises(wasm.BrowserUnsupported): wasm.randint(3, (50000, 50000))


@pytest.mark.parametrize('name,op', [('eq', operator.eq), ('ne', operator.ne), ('lt', operator.lt), ('le', operator.le), ('gt', operator.gt), ('ge', operator.ge)])
def test_named_comparisons_keep_precision_broadcasting_and_detach(name, op):
    data = np.array([[1., 1 + 2**-40], [np.nan, np.inf]])
    a, b = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    actual = getattr(a, name)(1 + 2**-40)
    np.testing.assert_array_equal(actual.numpy(), op(b, 1 + 2**-40).numpy())
    assert actual.dtype == np.dtype(bool) and not actual.requires_grad
