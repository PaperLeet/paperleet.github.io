"""Independent central differences for first-order reverse Jacobians."""
import numpy as np
import pytest
from stridetorch import torch

def functions(case, arrays):
    if case == 'softmax':
        def numerical(a):
            ex = np.exp(a - a.max(-1, keepdims=True))
            return ex / ex.sum(-1, keepdims=True)
        return lambda a: a.softmax(-1), numerical
    if case == 'linear':
        return lambda a, w, b: torch.nn.functional.linear(a, w, b).tanh(), lambda a, w, b: np.tanh(a @ w.T + b)
    if case == 'groupnorm':
        def numerical(a, w, b):
            grouped = a.reshape(1, 2, -1)
            centered = grouped - grouped.mean(-1, keepdims=True)
            normalized = centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + 1e-5)
            return normalized.reshape(a.shape) * w.reshape(1, 4, 1) + b.reshape(1, 4, 1)
        return lambda a, w, b: torch.nn.functional.group_norm(a, 2, w, b), numerical
    if case == 'convolution':
        def numerical(a, w):
            return np.array([[[[(a[0, 0, i:i+2, j:j+2] * w[0, 0]).sum()
                                  for j in range(2)] for i in range(2)]]], dtype=a.dtype)
        return lambda a, w: torch.nn.functional.conv2d(a, w), numerical
    if case == 'inverse':
        return lambda a: torch.linalg.inv(a), lambda a: np.linalg.inv(a)
    if case == 'attention':
        def numerical(q, k, v):
            logits = q @ k.T / np.sqrt(q.shape[-1])
            ex = np.exp(logits - logits.max(-1, keepdims=True))
            return ex / ex.sum(-1, keepdims=True) @ v
        return lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v), numerical
    raise AssertionError(case)

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('case', ['softmax', 'linear', 'groupnorm', 'convolution', 'inverse', 'attention'])
def test_jacobian_matches_independent_numpy_finite_differences(case, dtype):
    rng = np.random.default_rng(7014)
    shapes = dict(softmax=[(2, 3)], linear=[(2, 3), (2, 3), (2,)],
                  groupnorm=[(1, 4, 2), (4,), (4,)], convolution=[(1, 1, 3, 3), (1, 1, 2, 2)],
                  inverse=[(2, 2)], attention=[(2, 2), (2, 2), (2, 2)])[case]
    arrays = [(rng.normal(size=shape) * .3).astype(dtype) for shape in shapes]
    if case == 'inverse': arrays[0] += np.eye(2, dtype=dtype) * 2
    function, numerical = functions(case, arrays)
    values = tuple(torch.tensor(a, requires_grad=bool(i % 2)) for i, a in enumerate(arrays))
    buffers = []
    for value in values:
        value.grad = torch.ones_like(value)
        buffers.append(value.grad)
    calls = []
    def callback(*args):
        calls.append(1)
        return function(*args)
    with torch.no_grad():
        actual = torch.autograd.functional.jacobian(callback, values)
        assert not torch.is_grad_enabled()
    assert len(calls) == 1
    output = numerical(*arrays)
    epsilon = 1e-3 if dtype == np.float32 else 1e-5
    for i, (a, result) in enumerate(zip(arrays, actual)):
        expected = np.empty(output.shape + a.shape, dtype=np.float64)
        for index in np.ndindex(a.shape):
            plus = [v.copy() for v in arrays]; minus = [v.copy() for v in arrays]
            plus[i][index] += epsilon; minus[i][index] -= epsilon
            expected[(...,) + index] = (numerical(*plus) - numerical(*minus)) / (2 * epsilon)
        assert result.shape == output.shape + a.shape and result.dtype == np.dtype(dtype)
        assert not result.requires_grad
        np.testing.assert_allclose(result.numpy(), expected, rtol=5e-3 if dtype == np.float32 else 1e-7,
                                   atol=1e-4 if dtype == np.float32 else 2e-9)
        assert values[i].grad is buffers[i] and values[i].requires_grad is bool(i % 2)
        np.testing.assert_array_equal(values[i].detach().numpy(), a)

@pytest.mark.parametrize('kwargs', [dict(create_graph=True), dict(vectorize=True), dict(strategy='forward-mode')])
def test_unsupported_jacobian_modes_reject_before_callback(kwargs):
    calls = []
    with pytest.raises(RuntimeError):
        torch.autograd.functional.jacobian(lambda a: calls.append(a), torch.ones(2), **kwargs)
    assert not calls
