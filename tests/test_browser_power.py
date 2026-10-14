"""Native differential checks for RoPE's scalar-base tensor powers."""
import subprocess
import sys

import numpy as np
import pytest
import torch
from paperleet.browser import numpy_autograd as wasm


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('base', [.1, .5, 1., 1 + 2**-40, 17, 10000.])
@pytest.mark.parametrize('shape', [(), (2, 3), (0, 3)])
def test_scalar_base_values_and_exponent_vjp_match_native(dtype, base, shape):
    data = np.linspace(-1.5, 1.5, int(np.prod(shape)) if shape else 1, dtype=dtype).reshape(shape)
    a, b = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    actual, expected = base ** a, base ** b
    assert actual.dtype == expected.detach().numpy().dtype
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=tolerance, rtol=tolerance)
    upstream = np.arange(data.size, dtype=dtype).reshape(shape) + .3
    (actual * wasm.Tensor(upstream)).sum().backward(); (expected * torch.tensor(upstream)).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), atol=0, rtol=tolerance)
    np.testing.assert_equal(wasm.power(base, a.detach()).numpy(), actual.numpy())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_scalar_base_nonfinite_exponents_and_no_grad_match_native(dtype):
    data = np.array([-np.inf, -1., 0., 1., np.inf, np.nan], dtype=dtype)
    for base in (.5, 1., 1 + 2**-40, 2.):
        a, b = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
        actual, expected = base ** a, base ** b
        actual.sum().backward(); expected.sum().backward()
        np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), equal_nan=True)
        np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), equal_nan=True)
        with wasm.no_grad(): assert not (base ** a).requires_grad


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('base', [.5, 1., 1 + 2**-40, 17.])
def test_scalar_power_jvp_and_reverse_over_forward_match_native(dtype, base):
    data, tangent = np.array([-.3, .2, .7], dtype=dtype), np.array([.4, -.7, .9], dtype=dtype)
    a, da = wasm.Tensor(data, requires_grad=True), wasm.Tensor(tangent, requires_grad=True)
    b, db = torch.tensor(data, requires_grad=True), torch.tensor(tangent, requires_grad=True)
    actual, expected = wasm.jvp(lambda x:base ** x, (a,), (da,)), torch.func.jvp(lambda x:base ** x, (b,), (db,))
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    for got, want in zip(actual, expected): np.testing.assert_allclose(got.numpy(), want.detach().numpy(), atol=0, rtol=tolerance)
    (actual[0].sum() + actual[1].square().sum()).backward(); (expected[0].sum() + expected[1].square().sum()).backward()
    for got, want in [(a.grad, b.grad), (da.grad, db.grad)]:
        np.testing.assert_allclose(got.numpy(), want.numpy(), atol=0, rtol=tolerance)


def test_power_api_fallbacks_output_buffers_and_mutation_guard():
    a = wasm.Tensor(np.array([.2, .7]), requires_grad=True)
    for base in (0, -1, True, complex(1, 1), float('inf'), float('nan')):
        with pytest.raises(wasm.BrowserUnsupported): base ** a
    for value in (wasm.Tensor(np.array([1, 2])), wasm.Tensor(np.array([1j]))):
        with pytest.raises(wasm.BrowserUnsupported): 2. ** value
    for base in (1e-300, 1e300):
        with pytest.raises(wasm.BrowserUnsupported): base ** wasm.Tensor(np.array([1.], dtype=np.float32))
    with pytest.raises(wasm.BrowserUnsupported): wasm.power(2., a, out=a)
    with pytest.raises(wasm.BrowserUnsupported): wasm.power(a, a)
    result = 2. ** a
    with wasm.no_grad(): a[0] = 3
    with pytest.raises(wasm.BrowserUnsupported, match='modified'): result.sum().backward()


def test_public_torch_power_api_preserves_forward_ad_and_existing_scalar_exponents():
    script = '''from paperleet.browser import numpy_autograd as w
t=w.install();x=t.tensor([.3,.7],dtype=t.float64,requires_grad=True);v=t.ones_like(x)
p,d=t.func.jvp(lambda z:t.pow(17.,z),(x,),(v,));assert t.allclose(d,p*__import__('math').log(17.))
assert t.allclose(t.pow(x,2),x.square())
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
