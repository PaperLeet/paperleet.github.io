"""Public HVP behavior across scoped recording, ownership and old fallback."""
import gc
import weakref
import numpy as np
import pytest
from torchwasm import torch, _state


def norm_hessian_product(values, direction):
    length = np.linalg.norm(values)
    unit = values / length
    return (direction - unit * np.sum(unit * direction)) / length


def test_hvp_preserves_caller_owned_coefficient_and_calls_function_once():
    x = torch.tensor([1., 2., 3.], dtype=torch.float64)
    coefficient = torch.tensor([.5, -.25, 1.], dtype=torch.float64, requires_grad=True)
    original_type = type(coefficient)
    vector = torch.tensor([.2, -.3, .5], dtype=torch.float64)
    calls = []
    def function(a):
        calls.append(1)
        return torch.linalg.vector_norm(a + coefficient, ord=2)
    output, product = torch.autograd.functional.hvp(function, x, vector)
    assert calls == [1]
    assert type(coefficient) is original_type
    assert coefficient.grad is None
    assert not output.requires_grad and not product.requires_grad
    np.testing.assert_allclose(product.numpy(), norm_hessian_product(x.numpy() + coefficient.detach().numpy(), vector.numpy()), rtol=2e-14, atol=2e-14)
    assert _state._tensor_reverse_context is None


def test_unrecorded_matrix_path_preserves_existing_hvp_and_one_callback():
    x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    vector = torch.tensor([[.2, -.3], [.4, .1]], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    calls = []
    def function(a):
        calls.append(1)
        return torch.linalg.vector_norm(a @ identity, ord=2)
    _, product = torch.autograd.functional.hvp(function, x, vector)
    assert calls == [1]
    np.testing.assert_allclose(product.numpy(), norm_hessian_product(x.numpy(), vector.numpy()), rtol=2e-14, atol=2e-14)
    assert _state._tensor_reverse_context is None


def test_hvp_failure_restores_all_transform_and_recording_flags():
    names = ['_tensor_reverse_context', '_hvp_active', '_functional_ad_active',
             '_full_forward_graph_active', '_jvp_active', '_grad_enabled']
    original = {name: getattr(_state, name) for name in names}
    def function(a):
        torch.linalg.vector_norm(a, ord=2)
        raise ValueError('caller failure')
    with pytest.raises(ValueError, match='caller failure'):
        torch.autograd.functional.hvp(function, torch.tensor([1., 2.]), torch.ones(2))
    assert {name: getattr(_state, name) for name in names} == original
    torch.tensor(2., requires_grad=True).square().backward()


def test_saved_alias_mutation_is_rejected_and_context_is_released():
    def function(a):
        value = torch.linalg.vector_norm(a, ord=2)
        a.detach().fill_(3.)
        return value
    with pytest.raises(RuntimeError, match='mutation|saved|modified'):
        torch.autograd.functional.hvp(function, torch.tensor([1., 2.]), torch.ones(2))
    assert _state._tensor_reverse_context is None


def test_tensor_pullback_rejects_mutation_of_its_saved_alias():
    from torchwasm import _higher_order
    context = _higher_order.Context()
    previous = _state._tensor_reverse_context
    assert previous is None
    _state._tensor_reverse_context = context
    try:
        x = context.wrap(torch.tensor([1., 2.], requires_grad=True))
        output = torch.linalg.vector_norm(x, ord=2)
        x.detach().fill_(3.)
        with pytest.raises(RuntimeError, match='saved|modified'):
            context.grad((output,), (x,))
    finally:
        _state._tensor_reverse_context = previous
        context.clear()
    assert _state._tensor_reverse_context is None


def test_repeated_hvp_does_not_retain_internal_input_tensors_without_gc():
    references = []
    x = torch.tensor([1., 2., 3.], dtype=torch.float64)
    direction = torch.tensor([.1, -.2, .3], dtype=torch.float64)
    def function(a):
        references.append(weakref.ref(a._primal))
        return torch.linalg.vector_norm(a.exp(), ord=2)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(100):
            torch.autograd.functional.hvp(function, x, direction)
        assert all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()
    assert _state._tensor_reverse_context is None


@pytest.mark.parametrize('order', [0., 1., 2.])
def test_zero_or_constant_curvature_has_finite_zero_hvp(order):
    x = torch.zeros(3, dtype=torch.float64)
    _, result = torch.autograd.functional.hvp(lambda a: torch.linalg.vector_norm(a, ord=order), x, torch.ones_like(x))
    np.testing.assert_array_equal(result.numpy(), np.zeros(3))
    assert _state._tensor_reverse_context is None
