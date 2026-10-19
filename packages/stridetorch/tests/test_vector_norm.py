"""Known geometry, finite differences, AD composition and storage contracts."""
import gc
import weakref

import numpy as np
import pytest

from stridetorch import torch, UnsupportedOperation


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('p,expected', [(0, 3.), (1, 6.), (2, 14. ** .5),
                                      (3, 36. ** (1 / 3)), (-1, 6 / 11),
                                      (float('inf'), 3.), (-float('inf'), 1.)])
def test_scalar_known_values(dtype, p, expected):
    result = torch.linalg.vector_norm(torch.tensor([-1., 2., -3.], dtype=getattr(torch, dtype)), ord=p)
    assert result.shape == ()
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_allclose(result.numpy(), expected, rtol=3e-6)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('keepdim', [False, True])
def test_permuted_strided_axes_and_empty_axis_list(dtype, keepdim):
    source = np.zeros((2, 4, 6), dtype=dtype)
    source[0, :, ::2] = [3., 4., 0.]
    source[1, :, ::2] = [0., 0., -12.]
    x = torch.from_numpy(source[:, :, ::2]).transpose(0, 1)
    value = torch.linalg.vector_norm(x, dim=(2, 1), keepdim=keepdim)
    expected = np.full((4, 1, 1) if keepdim else (4,), 13., dtype=dtype)
    np.testing.assert_array_equal(value.numpy(), expected)
    total = torch.linalg.vector_norm(x, dim=[])
    np.testing.assert_array_equal(total.numpy(), np.asarray(26, dtype=dtype))


@pytest.mark.parametrize('p,expected', [
    (1, [1., -1., 0.]), (2, [.6, -.8, 0.]),
    (float('inf'), [0., -1., 0.]), (-float('inf'), [0., 0., 0.]),
])
def test_known_first_derivative_and_dynamic_seed(p, expected):
    x = torch.tensor([3., -4., 0.], dtype=torch.float64, requires_grad=True)
    value = torch.linalg.vector_norm(x, ord=p)
    for seed in (.25, -2.):
        gradient, = torch.autograd.grad(value, x, torch.tensor(seed, dtype=torch.float64), retain_graph=True)
        np.testing.assert_allclose(gradient.numpy(), np.asarray(expected) * seed, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('p', [float('inf'), -float('inf')])
def test_equal_extrema_share_the_gradient(p):
    x = torch.tensor([2., -2., 2.], dtype=torch.float64, requires_grad=True)
    torch.linalg.vector_norm(x, ord=p).backward()
    np.testing.assert_array_equal(x.grad.numpy(), [1 / 3, -1 / 3, 1 / 3])


@pytest.mark.parametrize('p', [-3., -.5, .5, 1.5, 3.5])
@pytest.mark.parametrize('dim', [None, 0, 1])
def test_arbitrary_order_reverse_and_jvp_by_finite_difference(p, dim):
    raw = np.array([[.75, -1.25, 2.], [-2.5, 1.75, -.5]])
    tangent = np.array([[.3, -.1, .2], [-.2, .4, .1]])
    x = torch.tensor(raw, requires_grad=True)
    function = lambda a: torch.linalg.vector_norm(a, ord=p, dim=dim)
    value = function(x)
    seed = np.arange(value.numel(), dtype='float64').reshape(value.shape) / 5 + .7
    gradient, = torch.autograd.grad(value, x, torch.tensor(seed))
    epsilon = 1e-5
    expected = np.empty_like(raw)
    for index in np.ndindex(raw.shape):
        positive, negative = raw.copy(), raw.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        expected[index] = ((function(torch.tensor(positive)).numpy() - function(torch.tensor(negative)).numpy()) * seed).sum() / (2 * epsilon)
    np.testing.assert_allclose(gradient.numpy(), expected, rtol=2e-7, atol=2e-9)
    _, actual = torch.func.jvp(function, (x,), (torch.tensor(tangent),))
    difference = (function(torch.tensor(raw + epsilon * tangent)).numpy() - function(torch.tensor(raw - epsilon * tangent)).numpy()) / (2 * epsilon)
    np.testing.assert_allclose(actual.numpy(), difference, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize('api', ['direct', 'functional', 'hvp'])
def test_euclidean_second_derivative_geometry(api):
    x = torch.tensor([3., 4.], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([2., -1.], dtype=torch.float64, requires_grad=True)
    function = lambda a: torch.linalg.vector_norm(a)
    unit = np.array([.6, .8])
    expected = (np.eye(2) - np.outer(unit, unit)) @ np.array([2., -1.]) / 5
    if api == 'hvp':
        _, result = torch.autograd.functional.hvp(function, x, v)
    elif api == 'direct':
        _, tangent = torch.func.jvp(function, (x,), (v,))
        result, dv = torch.autograd.grad(tangent, (x, v))
        np.testing.assert_allclose(dv.numpy(), unit, rtol=1e-13, atol=1e-13)
    else:
        _, tangent = torch.autograd.functional.jvp(function, x, v, create_graph=True)
        result, = torch.autograd.grad(tangent, x)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('p', [.5, 1., 1.5, 2., 3.5, float('inf'), -float('inf')])
def test_zero_vector_and_zero_jvp(p):
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    function = lambda a: torch.linalg.vector_norm(a, ord=p)
    value = function(x)
    value.backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.zeros(3))
    _, tangent = torch.func.jvp(function, (x,), (torch.ones(3, dtype=torch.float64),))
    np.testing.assert_array_equal(tangent.numpy(), 0.)


def test_zero_order_has_no_reverse_edge_but_zero_forward_direction():
    x = torch.tensor([0., 2., -3.], dtype=torch.float64, requires_grad=True)
    value = torch.linalg.vector_norm(x, ord=0)
    value.backward()
    assert x.grad is None
    _, tangent = torch.func.jvp(lambda a: torch.linalg.vector_norm(a, ord=0), (x,), (torch.ones(3, dtype=torch.float64),))
    assert not tangent.requires_grad
    np.testing.assert_array_equal(tangent.numpy(), 0.)


@pytest.mark.parametrize('p', [1.5, 3.5])
@pytest.mark.parametrize('empty', [False, True])
def test_reverse_based_jvp_keeps_native_zero_denominator_ieee_behavior(p, empty):
    x = torch.zeros(0 if empty else 3, dtype=torch.float64)
    vector = torch.ones_like(x)
    _, direct = torch.func.jvp(lambda a: torch.linalg.vector_norm(a, ord=p), (x,), (vector,))
    _, functional = torch.autograd.functional.jvp(lambda a: torch.linalg.vector_norm(a, ord=p), x, vector)
    np.testing.assert_array_equal(direct.numpy(), 0.)
    assert np.isnan(functional.item())


@pytest.mark.parametrize('p', [2., 3.5, -1., -2.])
def test_functional_hvp_zero_subgradient_does_not_change_direct_reverse(p):
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    vector = torch.ones_like(x)
    _, hessian = torch.autograd.functional.hvp(lambda a: torch.linalg.vector_norm(a, ord=p), x, vector)
    np.testing.assert_array_equal(hessian.numpy(), np.zeros(3))
    if p == 2:
        _, tangent = torch.func.jvp(lambda a: torch.linalg.vector_norm(a), (x,), (vector,))
        derivative, = torch.autograd.grad(tangent, x)
        assert np.isnan(derivative.numpy()).all()


def test_fractional_hvp_masks_zero_coordinate_and_matches_nonzero_finite_difference():
    x = torch.tensor([0., 1., 4.], dtype=torch.float64)
    vector = torch.tensor([.2, -.3, .1], dtype=torch.float64)
    function = lambda a: torch.linalg.vector_norm(a, ord=.5)
    _, hessian = torch.autograd.functional.hvp(function, x, vector)
    assert hessian.numpy()[0] == 0
    epsilon = 1e-5
    # Keep the nonsmooth zero coordinate fixed for the independent difference.
    direction = np.array([0., -.3, .1])
    results = []
    for sign in [-1, 1]:
        point = torch.tensor(x.numpy() + sign * epsilon * direction, requires_grad=True)
        gradient, = torch.autograd.grad(function(point), point)
        results.append(gradient.numpy())
    np.testing.assert_allclose(hessian.numpy()[1:], ((results[1] - results[0]) / (2 * epsilon))[1:], rtol=2e-7, atol=1e-10)


@pytest.mark.parametrize('values', [[0., 0., 0.], [float('nan'), 1., 0.]])
def test_fractional_hvp_preserves_nan_at_singular_norm(values):
    x = torch.tensor(values, dtype=torch.float64)
    _, hessian = torch.autograd.functional.hvp(lambda a: torch.linalg.vector_norm(a, ord=.5), x, torch.ones_like(x))
    assert np.isnan(hessian.numpy()).all()


@pytest.mark.parametrize('p', [0, 1, 2, .5, 3.5])
def test_empty_reduction_identity_and_empty_batch(p):
    x = torch.empty(2, 0, 3, dtype=torch.float64, requires_grad=True)
    value = torch.linalg.vector_norm(x, ord=p, dim=1)
    assert value.shape == (2, 3)
    np.testing.assert_array_equal(value.numpy(), np.zeros((2, 3)))
    if p:
        value.sum().backward()
        assert x.grad.shape == x.shape
    other = torch.linalg.vector_norm(torch.empty(0, 3, dtype=torch.float64), ord=-1, dim=1)
    assert other.shape == (0,)


@pytest.mark.parametrize('dtype,target', [('float16', 'float32'), ('float32', 'float64'), ('complex64', 'complex128')])
def test_dtype_widening_before_accumulation(dtype, target):
    raw = np.asarray([3., 4.], dtype=dtype)
    if raw.dtype.kind == 'c':
        raw = np.asarray([3j, -4.], dtype=dtype)
    value = torch.linalg.vector_norm(torch.from_numpy(raw), dtype=getattr(torch, target))
    assert value.dtype == np.empty((), dtype=target).real.dtype
    np.testing.assert_array_equal(value.numpy(), 5.)


def test_saved_operand_and_output_mutation_guards_and_final_release():
    leaf = torch.tensor([1., 2., 3.], dtype=torch.float64, requires_grad=True)
    intermediate = leaf * 2
    saved = weakref.ref(intermediate._data)
    wrapper = weakref.ref(intermediate)
    value = torch.linalg.vector_norm(intermediate)
    del intermediate
    gc.collect()
    assert wrapper() is None and saved() is not None
    for seed in (1., -2.):
        value.backward(torch.tensor(seed, dtype=torch.float64), retain_graph=True)
    value.backward()
    gc.collect()
    assert saved() is None
    fresh = torch.linalg.vector_norm(leaf)
    with torch.no_grad():
        leaf.add_(1.)
    with pytest.raises(UnsupportedOperation, match='modified in-place'):
        fresh.backward()


@pytest.mark.parametrize('call', [
    lambda: torch.linalg.vector_norm(torch.ones(3), ord='fro'),
    lambda: torch.linalg.vector_norm(torch.ones(3), dim=(0, -1)),
    lambda: torch.linalg.vector_norm(torch.ones(3), dim=3),
    lambda: torch.linalg.vector_norm(torch.ones(3), keepdim=1),
    lambda: torch.linalg.vector_norm(torch.ones(3, dtype=torch.int64)),
    lambda: torch.linalg.vector_norm(torch.ones(3, dtype=torch.float64), dtype=torch.float32),
    lambda: torch.linalg.vector_norm(torch.ones(3), dtype=torch.complex64),
    lambda: torch.linalg.vector_norm(torch.empty(0), ord=-1),
    lambda: torch.linalg.vector_norm(torch.empty(0), ord=float('inf')),
    lambda: torch.linalg.vector_norm(torch.ones(3), out=torch.empty(())),
])
def test_invalid_or_unsupported_boundaries(call):
    with pytest.raises((TypeError, RuntimeError, IndexError, UnsupportedOperation)):
        call()
