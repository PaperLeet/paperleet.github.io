"""Independent native and polynomial oracles for symmetric spectral gradients."""
import numpy as np
import pytest
import torch

from paperleet.browser import numpy_autograd as wasm


def close(actual, expected, dtype=np.float64):
    np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=3e-5 if dtype == np.float32 else 2e-11, rtol=3e-5 if dtype == np.float32 else 2e-11)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(3, 3), (2, 4, 4), (2, 1, 3, 3), (0, 3, 3), (2, 0, 0)])
@pytest.mark.parametrize('triangle', ['L', 'U'])
def test_eigh_both_output_vjps_match_native_on_strided_triangle_inputs(dtype, shape, triangle):
    rng = np.random.RandomState(109)
    data = rng.randn(*shape).astype(dtype).swapaxes(-1, -2)
    a, b = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    values, vectors = wasm.eigh(a, triangle)
    native_values, native_vectors = torch.linalg.eigh(b, UPLO=triangle)
    close(values, native_values, dtype)
    signs = np.sign((vectors.numpy() * native_vectors.detach().numpy()).sum(axis=-2))
    gl, gq = rng.randn(*values.shape).astype(dtype), rng.randn(*vectors.shape).astype(dtype)
    ((values * wasm.Tensor(gl)).sum() + (vectors * wasm.Tensor(gq)).sum()).backward()
    ((native_values * torch.tensor(gl)).sum() + (native_vectors * torch.tensor(gq * signs[..., None, :])).sum()).backward()
    close(a.grad, b.grad, dtype)
    np.testing.assert_array_equal(a.numpy(), data)


def test_symmetric_objective_vjp_matches_independent_finite_differences():
    rng = np.random.RandomState(113)
    data = rng.randn(3, 3); data = data + data.T + 5 * np.eye(3)
    weights = rng.randn(3, 3)
    a = wasm.Tensor(data.copy(), requires_grad=True)
    values, vectors = wasm.eigh(a)
    reconstruction = (vectors * values.sqrt()[None, :]) @ vectors.T
    (reconstruction * wasm.Tensor(weights)).sum().backward()
    def objective(raw):
        l, q = np.linalg.eigh(raw)
        return ((q * np.sqrt(l)[None, :]) @ q.T * weights).sum()
    for i in range(3):
        for j in range(i, 3):
            direction = np.zeros((3, 3)); direction[i, j] = direction[j, i] = 1
            finite = (objective(data + 1e-6 * direction) - objective(data - 1e-6 * direction)) / 2e-6
            np.testing.assert_allclose((a.grad.numpy() * direction).sum(), finite, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize('triangle', ['L', 'U'])
@pytest.mark.parametrize('power', [1, 2, 3])
def test_repeated_eigenvalues_retain_finite_polynomial_gradients(triangle, power):
    rotation, _ = np.linalg.qr(np.random.RandomState(127).randn(4, 4))
    data = rotation @ np.diag([0., 2., 2., 5.]) @ rotation.T
    a = wasm.Tensor(data, requires_grad=True)
    (wasm.eigvalsh(a, triangle) ** power).sum().backward()
    expected = power * np.linalg.matrix_power(data, power - 1)
    np.testing.assert_allclose(a.grad.numpy(), expected, atol=2e-12, rtol=2e-12)


def test_unresolved_vector_and_basis_dependent_value_gradients_require_server():
    for data in [np.eye(3), np.diag([1., 1 + 1e-16, 2.])]:
        a = wasm.Tensor(data, requires_grad=True)
        with pytest.raises(wasm.BrowserUnsupported, match='eigenvector'): wasm.eigh(a).eigenvectors.sum().backward()
        assert a.grad is None
        with pytest.raises(wasm.BrowserUnsupported, match='repeated-eigenvalue'):
            (wasm.eigvalsh(a) * wasm.Tensor(np.array([1., 2., 3.]))).sum().backward()
        assert a.grad is None
    with wasm.no_grad():
        assert all(not t.requires_grad for t in wasm.eigh(wasm.Tensor(np.eye(3), requires_grad=True)))


def test_eigh_saved_input_values_and_vectors_preserve_version_guards():
    for changed in ['input', 'values', 'vectors']:
        a = wasm.Tensor(np.array([[3., .4], [.4, 1.]]), requires_grad=True)
        values, vectors = wasm.eigh(a)
        loss = values.sum() + vectors.sum()
        with wasm.no_grad(): {'input': a, 'values': values, 'vectors': vectors}[changed].detach()[...] = 0.
        with pytest.raises(wasm.BrowserUnsupported, match='saved for backward'): loss.backward()
        assert a.grad is None


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(1, 1), (3, 3), (2, 4, 4), (0, 3, 3), (2, 0, 0)])
def test_determinants_and_cofactor_vjps_match_native(dtype, shape):
    data = np.random.RandomState(131).randn(*shape).astype(dtype).swapaxes(-1, -2)
    a, b = wasm.Tensor(data, requires_grad=True), torch.tensor(data, requires_grad=True)
    actual, expected = wasm.det(a), torch.linalg.det(b)
    (actual * 1.7).sum().backward(); (expected * 1.7).sum().backward()
    close(actual, expected, dtype); close(a.grad, b.grad, dtype)


@pytest.mark.parametrize('data', [np.diag([2., 3., 0.]), np.diag([2., 0., 0.]), np.zeros((3, 3)),
    np.array([[0., 2.], [0., 0.]]), np.zeros((1, 1)), np.empty((0, 0))])
def test_singular_determinant_gradients_match_polynomial_cofactors(data):
    a = wasm.Tensor(data, requires_grad=True)
    a.det().backward()
    expected = np.empty_like(data)
    for i, j in np.ndindex(data.shape):
        expected[i, j] = (-1.) ** (i + j) * np.linalg.det(np.delete(np.delete(data, i, axis=0), j, axis=1))
    np.testing.assert_allclose(a.grad.numpy(), expected, atol=1e-12)


@pytest.mark.parametrize('shape,offset,dim1,dim2', [((3, 5), 0, 0, 1), ((3, 5), -1, 0, 1),
    ((3, 5), 2, 1, 0), ((2, 3, 4), 1, -1, -2), ((2, 3, 4), -2, 0, 2), ((0, 3, 4), 10, 1, 2)])
def test_diagonal_views_values_vjp_and_alias_versions_match_native(shape, offset, dim1, dim2):
    data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    a, b = wasm.Tensor(data.copy(), requires_grad=True), torch.tensor(data, requires_grad=True)
    actual, expected = a.diagonal(offset, dim1, dim2), b.diagonal(offset, dim1, dim2)
    actual.square().sum().backward(); expected.square().sum().backward()
    close(actual, expected); close(a.grad, b.grad)
    pending = a.square().sum()
    with wasm.no_grad(), torch.no_grad(): actual.detach()[...] = -3; expected.detach()[...] = -3
    close(a, b)
    with pytest.raises(wasm.BrowserUnsupported, match='saved for backward'): pending.backward()


def test_invalid_and_unsupported_spectral_arguments_remain_explicit():
    for function in [wasm.eigh, wasm.eigvalsh, wasm.det]:
        for data in [np.ones((2, 3)), np.ones(3), np.array([[np.nan]])]:
            with pytest.raises(RuntimeError): function(wasm.Tensor(data))
        with pytest.raises(wasm.BrowserUnsupported): function(wasm.Tensor(np.eye(2, dtype=np.complex128)))
        with pytest.raises(wasm.BrowserUnsupported): function(wasm.Tensor(np.eye(2)), out=wasm.Tensor(np.eye(2)))
    with pytest.raises(RuntimeError): wasm.eigh(wasm.Tensor(np.eye(2)), 'X')
