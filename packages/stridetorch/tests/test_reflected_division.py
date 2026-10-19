"""Closed-form curvature for captured Tensor numerators in norm HVPs."""
import numpy as np
import pytest

from stridetorch import torch


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
@pytest.mark.parametrize('coefficient_dtype', ['float32', 'float64'])
@pytest.mark.parametrize('active', [False, True])
@pytest.mark.parametrize('scalar', [False, True])
@pytest.mark.parametrize('use_norm', [False, True], ids=['sum', 'norm'])
def test_captured_tensor_numerator_keeps_history_and_caller_state(
        dtype_name, coefficient_dtype, active, scalar, use_norm):
    raw = np.array([1., 2., 4.], dtype=dtype_name)
    coefficient = np.array(2. if scalar else [2., 3., 5.], dtype=coefficient_dtype)
    direction = np.array([.25, -.5, .75], dtype=dtype_name)
    x = torch.tensor(raw, requires_grad=True)
    x.grad = torch.full_like(x, 7.)
    numerator = torch.tensor(coefficient, dtype=getattr(torch, coefficient_dtype), requires_grad=active)
    if active:
        numerator.grad = torch.full_like(numerator, 9.)
    original_class = type(numerator)
    calls = []
    def function(value):
        calls.append(1)
        quotient = numerator / value
        return torch.linalg.vector_norm(quotient) if use_norm else quotient.sum()
    value, product = torch.autograd.functional.hvp(function, x, torch.tensor(direction))
    # For f(x)=sqrt(sum(c_i^2/x_i^2)), H=diag(3*c^2/x^4/f)
    # - outer(c^2/x^3,c^2/x^3)/f^3. Differentiate only x; c is captured.
    reference = np.broadcast_to(coefficient.astype('float64'), raw.shape)
    point, vector = raw.astype('float64'), direction.astype('float64')
    if use_norm:
        expected_value = np.sqrt(np.sum((reference / point) ** 2))
        slope = reference ** 2 / point ** 3
        expected = 3 * reference ** 2 / point ** 4 / expected_value * vector - slope * np.dot(slope, vector) / expected_value ** 3
    else:
        expected_value = np.sum(reference / point)
        expected = 2 * reference / point ** 3 * vector
    tolerance = 5e-6 if dtype_name == 'float32' else 2e-12
    output_dtype = np.dtype(dtype_name) if scalar else np.promote_types(dtype_name, coefficient_dtype)
    assert value.numpy().dtype == output_dtype
    assert product.numpy().dtype == np.dtype(dtype_name)
    np.testing.assert_allclose(value.numpy(), expected_value, rtol=tolerance)
    np.testing.assert_allclose(product.numpy(), expected, rtol=tolerance, atol=tolerance * 1e-2)
    assert calls == [1] and type(numerator) is original_class
    assert numerator.requires_grad == active
    np.testing.assert_array_equal(x.detach().numpy(), raw)
    np.testing.assert_array_equal(x.grad.numpy(), np.full(raw.shape, 7.))
    np.testing.assert_array_equal(numerator.detach().numpy(), coefficient)
    if active:
        np.testing.assert_array_equal(numerator.grad.numpy(), np.full(coefficient.shape, 9.))
    else:
        assert numerator.grad is None
