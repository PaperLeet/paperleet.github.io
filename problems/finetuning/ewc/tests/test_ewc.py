import math
import pytest
import torch

from paperleet.common.asserts import assert_close_diagnostic as close


@pytest.mark.t0
def test_scalar_shape_dtype_and_tensor_contract(impl):
    """Equation (3) sums parameters into one scalar, including a scalar parameter."""
    for dtype in (torch.float32, torch.float64):
        theta = torch.tensor(3., dtype=dtype)
        result = impl.ewc_penalty(theta, torch.tensor(1., dtype=dtype), torch.tensor(.5, dtype=dtype), 2.)
        assert isinstance(result, torch.Tensor) and result.shape == ()
        assert result.dtype == dtype and result.device == theta.device
        close(result, torch.tensor(2., dtype=dtype))


@pytest.mark.t1
def test_independent_scalar_sum_oracle(impl):
    """A Python scalar sum checks unequal weights, signs and the factor one-half."""
    values = [[-2., .5, 3.], [4., -1., 0.]]
    anchors = [[1., -.5, 3.], [-2., 2., .5]]
    weights = [[.25, 4., 0.], [2., .5, 3.]]
    for dtype in (torch.float32, torch.float64):
        for strength in (.25, 3.):
            expected = math.fsum(strength * f * (x - a) ** 2 / 2
                                 for xs, aa, ff in zip(values, anchors, weights)
                                 for x, a, f in zip(xs, aa, ff))
            got = impl.ewc_penalty(torch.tensor(values, dtype=dtype), torch.tensor(anchors, dtype=dtype),
                                   torch.tensor(weights, dtype=dtype), strength)
            close(got, torch.tensor(expected, dtype=dtype))


@pytest.mark.t2
def test_analytic_gradient_and_frozen_snapshots(impl):
    """dP/dtheta=lambda*F*(theta-anchor); neither stored snapshot is optimized."""
    theta = torch.tensor([[2., -1.], [3., 4.]], dtype=torch.float64, requires_grad=True)
    anchor = torch.tensor([[1., 1.], [3., 0.]], dtype=torch.float64, requires_grad=True)
    fisher = torch.tensor([[2., .5], [0., 3.]], dtype=torch.float64, requires_grad=True)
    penalty = impl.ewc_penalty(theta, anchor, fisher, 1.5)
    close(penalty, torch.tensor(39., dtype=torch.float64))
    penalty.backward()
    close(theta.grad, torch.tensor([[3., -1.5], [0., 18.]], dtype=torch.float64))
    assert anchor.grad is None, "The old optimum must be frozen, not trained toward theta"
    assert fisher.grad is None, "Given importance weights must remain frozen"


@pytest.mark.t2
def test_zero_strength_zero_fisher_and_empty_are_differentiable(impl):
    """Disabled, unimportant and absent parameters have exactly zero contribution and gradient."""
    cases = [([1e30, -1e30], [-1e30, 1e30], [2., 3.], 0.),
             ([1e30, -1e30], [-1e30, 1e30], [0., 0.], 2.),
             ([1e30, 3.], [-1e30, 1.], [0., 2.], 1.)]
    for xs, aa, ff, strength in cases:
        theta = torch.tensor(xs, dtype=torch.float32, requires_grad=True)
        result = impl.ewc_penalty(theta, torch.tensor(aa), torch.tensor(ff), strength)
        expected = 4. if ff == [0., 2.] else 0.
        close(result, torch.tensor(expected))
        result.backward()
        close(theta.grad, torch.tensor([0., 4.] if expected else [0., 0.]))
    for shape in ((0,), (2, 0, 3)):
        theta = torch.empty(shape, dtype=torch.float64, requires_grad=True)
        result = impl.ewc_penalty(theta, torch.empty(shape, dtype=torch.float64), torch.empty(shape, dtype=torch.float64), 1.)
        close(result, torch.tensor(0., dtype=torch.float64))
        result.backward()
        assert theta.grad is not None and theta.grad.shape == theta.shape


@pytest.mark.t2
def test_additivity_translation_and_quadratic_scaling(impl):
    """Independent parameter blocks add; common shifts cancel and displacements scale quadratically."""
    theta = torch.tensor([1., 3., -2.], dtype=torch.float64)
    anchor = torch.tensor([-.5, 1., 4.], dtype=torch.float64)
    fisher = torch.tensor([2., .5, 0.], dtype=torch.float64)
    base = impl.ewc_penalty(theta, anchor, fisher, .75)
    split = impl.ewc_penalty(theta[:1], anchor[:1], fisher[:1], .75) + impl.ewc_penalty(theta[1:], anchor[1:], fisher[1:], .75)
    close(base, split)
    close(impl.ewc_penalty(theta + 8, anchor + 8, fisher, .75), base)
    close(impl.ewc_penalty(anchor + 2 * (theta - anchor), anchor, fisher, .75), 4 * base)
    close(impl.ewc_penalty(theta, anchor, fisher, 2.25), 3 * base)
    close(impl.ewc_penalty(anchor, anchor, fisher, .75), torch.tensor(0., dtype=torch.float64))


@pytest.mark.t2
def test_noncontiguous_layout_permutation_and_input_preservation(impl):
    """Reordering matched parameters cannot change the sum or mutate any snapshot."""
    theta = torch.arange(12, dtype=torch.float64).reshape(3, 4)[:, ::2]
    anchor = torch.ones(3, 4, dtype=torch.float64)[:, ::2]
    fisher = (torch.arange(12, dtype=torch.float64).reshape(3, 4) + 1)[:, ::2]
    before = [value.clone() for value in (theta, anchor, fisher)]
    value = impl.ewc_penalty(theta, anchor, fisher, .2)
    close(impl.ewc_penalty(theta.T, anchor.T, fisher.T, .2), value)
    expected = .1 * math.fsum(f * (x - a) ** 2 for xs, aa, ff in zip(theta.tolist(), anchor.tolist(), fisher.tolist()) for x, a, f in zip(xs, aa, ff))
    close(value, torch.tensor(expected, dtype=torch.float64))
    for current, original in zip((theta, anchor, fisher), before):
        close(current, original)


@pytest.mark.t3
def test_sequential_training_matches_quadratic_optimum_and_tradeoff(impl, artifact):
    """Actual task-B updates approach the independently solved regularized quadratic optimum."""
    from problems.finetuning.ewc.experiment import run
    result = run(impl.ewc_penalty)
    artifact('toy_experiment', result)
    metrics = result['metrics']
    assert metrics['task_a_before_b'] < 1e-12
    # Task-B Hessian is .5*I; supplied F=[2.5,0], strength=1.
    optimum = torch.tensor([(.5 * -2 + 2.5 * result['anchor'][0]) / 3., 2.], dtype=torch.float64)
    close(torch.tensor(result['runs']['ewc']['theta'], dtype=torch.float64), optimum, rtol=1e-3, atol=1e-3)
    close(torch.tensor(result['runs']['unregularized']['theta'], dtype=torch.float64), torch.tensor([-2., 2.], dtype=torch.float64), rtol=1e-3, atol=1e-3)
    assert metrics['unregularized_old_mse'] > 30
    assert .5 < metrics['ewc_old_mse'] < 2
    assert metrics['ewc_new_mse'] < metrics['initial_new_mse']
    assert metrics['ewc_new_mse'] > metrics['unregularized_new_mse']
