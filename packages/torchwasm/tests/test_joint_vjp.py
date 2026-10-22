"""Core graph contracts for a shared multi-input pullback, without native torch."""
import numpy as np
import pytest

from torchwasm import UnsupportedOperation, torch
from torchwasm._autograd import joint_result


def test_joint_pullback_runs_once_per_traversal_and_accumulates_shared_inputs():
    x = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    frozen = torch.tensor([5., 7.], dtype=torch.float64)
    calls = []

    def backward(seed, needs):
        calls.append((seed.copy(), needs))
        return seed * x.numpy(), seed * x.numpy(), None

    value = joint_result(x.numpy() ** 2 + frozen.numpy(), [x, x, frozen], backward)
    combined = value + value
    for seed in (np.array([1., -.5]), np.array([-2., 3.])):
        x.grad = None
        combined.backward(torch.tensor(seed), retain_graph=True)
        np.testing.assert_array_equal(x.grad.numpy(), 4 * seed * x.numpy())
    assert len(calls) == 2
    assert all(needs == (True, True, False) for _, needs in calls)
    assert frozen.grad is None


def test_joint_pullback_guards_frozen_and_explicit_saved_operands_before_callback():
    x = torch.tensor([2.], requires_grad=True)
    frozen = torch.tensor([3.])
    auxiliary = torch.tensor([4.])
    calls = []

    def backward(seed, needs):
        calls.append(needs)
        return seed, None

    for mutated in (frozen, auxiliary):
        value = joint_result(x.numpy().copy(), [x, frozen], backward, saved=[auxiliary])
        with torch.no_grad():
            mutated.fill_(6.)
        with pytest.raises(UnsupportedOperation, match='modified in-place'):
            value.backward()
        assert calls == [] and x.grad is None


def test_joint_pullback_is_disabled_by_no_grad():
    x = torch.tensor([2.], requires_grad=True)
    with torch.no_grad():
        value = joint_result(x.numpy().copy(), [x], lambda *_: pytest.fail('unexpected pullback'))
    assert not value.requires_grad
    assert value._parents == () and value._joint_vjp is None


@pytest.mark.parametrize('derivatives,error', [
    ([], 'one gradient per input'), ([None], 'required input gradient'),
    ([np.ones(2)], 'shape must match'), (np.ones(1), 'one gradient per input'),
])
def test_joint_pullback_rejects_malformed_derivatives_without_accumulating(derivatives, error):
    x = torch.tensor([2.], requires_grad=True)
    value = joint_result(x.numpy().copy(), [x], lambda *_: derivatives)
    with pytest.raises(RuntimeError, match=error):
        value.backward()
    assert x.grad is None


def test_joint_pullback_rejects_unvalidated_forward_mode():
    x = torch.tensor([2.])
    with pytest.raises(UnsupportedOperation, match='no validated forward-mode'):
        torch.func.jvp(lambda dual: joint_result(np.array([2.]), [dual], lambda *_: [np.ones(1)]),
                       (x,), (torch.ones(1),))
