"""Logical cotangents: IEEE distinction, hooks, pruning and tape ownership."""
import gc
import weakref

import numpy as np
import pytest

from stridetorch import torch, UnsupportedOperation, _autograd


def constant_sign_tangent(source):
    primal, tangent = torch.func.jvp(lambda a: torch.linalg.vector_norm(a, ord=1),
                                    (source,), (torch.ones_like(source),))
    del primal
    return tangent


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_logical_zero_skips_nonfinite_upstream_pullback(dtype):
    x = torch.tensor([float('nan'), float('inf'), -float('inf'), 2.], dtype=dtype, requires_grad=True)
    tangent = constant_sign_tangent(x.square())
    gradient, = torch.autograd.grad(tangent, x)
    np.testing.assert_array_equal(gradient.numpy(), np.zeros(4))
    assert gradient.dtype == dtype


def test_numeric_zero_keeps_ieee_multiplication():
    x = torch.tensor([float('inf'), -float('inf')], dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad((x.square() * 0.).sum(), x)
    assert np.isnan(gradient.numpy()).all()


@pytest.mark.parametrize('reverse_order', [False, True])
def test_logical_zero_and_real_contribution_accumulate_in_either_order(reverse_order):
    x = torch.tensor([float('inf'), -float('inf')], dtype=torch.float64, requires_grad=True)
    zero = constant_sign_tangent(x.square())
    real = (x * 3).sum()
    result = zero + real if reverse_order else real + zero
    gradient, = torch.autograd.grad(result, x)
    np.testing.assert_array_equal(gradient.numpy(), [3., 3.])


def test_logical_zero_reaches_joint_parents_without_numerical_kernel():
    a = torch.full((2, 2), float('inf'), dtype=torch.float64, requires_grad=True)
    b = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    def forbidden(*args):
        raise AssertionError('A logical-zero cotangent ran the numerical joint kernel')
    source = _autograd.joint_result(np.full((2, 2), float('inf')), (a, b), forbidden, zero_passthrough=True)
    tangent = constant_sign_tangent(source)
    gradients = torch.autograd.grad(tangent, (a, b))
    for gradient, input in zip(gradients, (a, b)):
        np.testing.assert_array_equal(gradient.numpy(), np.zeros(input.shape))


def test_undefined_upstream_pullback_stays_unused():
    x = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    source = torch.linalg.vector_norm(x.square(), ord=0)
    gradient, = torch.autograd.grad(constant_sign_tangent(source), x, allow_unused=True)
    assert gradient is None


def test_undeclared_materializing_rule_keeps_numeric_zero_behavior():
    x = torch.tensor([float('inf'), -float('inf')], dtype=torch.float64, requires_grad=True)
    source = x.square()
    opaque = _autograd.result(source.numpy().copy(), (source, lambda g: np.zeros_like(g)))
    gradient, = torch.autograd.grad(constant_sign_tangent(opaque), x)
    assert np.isnan(gradient.numpy()).all()


def test_saved_versions_are_checked_for_logical_zero():
    x = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    tangent = constant_sign_tangent(x.square())
    with torch.no_grad():
        x.add_(1.)
    with pytest.raises(UnsupportedOperation, match='modified in-place'):
        tangent.backward()


@pytest.mark.parametrize('hook_kind', ['none', 'identity', 'replace'])
def test_intermediate_hooks_observe_zero_and_can_replace_it(hook_kind):
    x = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    source = x.square()
    source.retain_grad()
    seen = []
    def hook(gradient):
        seen.append(gradient.numpy().copy())
        return None if hook_kind == 'none' else gradient if hook_kind == 'identity' else torch.ones_like(gradient)
    source.register_hook(hook)
    gradient, = torch.autograd.grad(constant_sign_tangent(source), x)
    np.testing.assert_array_equal(seen, [[0., 0.]])
    np.testing.assert_array_equal(source.grad.numpy(), [1., 1.] if hook_kind == 'replace' else [0., 0.])
    np.testing.assert_array_equal(gradient.numpy(), [4., 6.] if hook_kind == 'replace' else [0., 0.])


def test_identity_hook_preserves_zero_through_nonfinite_source():
    x = torch.tensor([float('inf'), float('nan')], dtype=torch.float64, requires_grad=True)
    source = x.square()
    source.register_hook(lambda g: g)
    gradient, = torch.autograd.grad(constant_sign_tangent(source), x)
    np.testing.assert_array_equal(gradient.numpy(), [0., 0.])


def test_leaf_hook_can_override_materialized_zero():
    x = torch.tensor([float('inf'), 2.], dtype=torch.float64, requires_grad=True)
    x.register_hook(lambda g: g + 1)
    gradient, = torch.autograd.grad(constant_sign_tangent(x.square()), x)
    np.testing.assert_array_equal(gradient.numpy(), [1., 1.])


def test_logical_zero_graph_pruning_repeat_and_release():
    x = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    other = torch.tensor([4., 5.], dtype=torch.float64, requires_grad=True)
    source = x.square() + other.square()
    data = weakref.ref(source._data)
    wrapper = weakref.ref(source)
    tangent = constant_sign_tangent(source)
    del source
    gc.collect()
    assert wrapper() is None
    gradient, = torch.autograd.grad(tangent, x, retain_graph=True)
    np.testing.assert_array_equal(gradient.numpy(), [0., 0.])
    gradients = torch.autograd.grad(tangent, (x, other))
    for gradient in gradients:
        np.testing.assert_array_equal(gradient.numpy(), [0., 0.])
    gc.collect()
    assert data() is None
    with pytest.raises(RuntimeError, match='second time'):
        tangent.backward()
