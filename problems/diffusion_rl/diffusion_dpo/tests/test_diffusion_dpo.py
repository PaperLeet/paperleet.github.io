import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.diffusion_dpo.experiment import run


@pytest.mark.t0
def test_contract_and_no_mutation(impl):
    """Noising preserves arbitrary feature shapes; loss is a scalar of the original dtype."""
    for dtype in (torch.float32, torch.float64):
        values = [torch.randn(3, 2, 4, dtype=dtype) for _ in range(5)]
        before = [v.clone() for v in values]
        w, l = impl.paired_noising(values[0], values[1], values[4], torch.tensor([0., .4, 1.], dtype=dtype))
        assert w.shape == l.shape == values[0].shape and w.dtype == l.dtype == dtype
        loss = impl.diffusion_dpo_loss(*values, 1.2)
        assert loss.shape == () and loss.dtype == dtype and loss.device == values[0].device
        for value, saved in zip(values, before): close(value, saved)


@pytest.mark.t0
def test_invalid_contracts(impl):
    """Reject mismatched pairs, non-finite inputs and invalid schedule or beta values."""
    x = torch.ones(2, 3); alpha = torch.ones(2)
    for wrong in (torch.ones(2, 1), torch.tensor([-1., 1.]), torch.tensor([0., float('nan')])):
        with pytest.raises(ValueError): impl.paired_noising(x, x, x, wrong)
    for beta in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError): impl.diffusion_dpo_loss(x, x, x, x, x, beta)
    with pytest.raises(ValueError): impl.diffusion_dpo_loss(x, x[:, :1], x, x, x, 1.)
    with pytest.raises(ValueError): impl.paired_noising(x, x, x * float('inf'), alpha)


@pytest.mark.t1
def test_shared_noise_and_schedule(impl):
    """Scalar noising uses square-root cumulative alpha and identical noise for each pair."""
    w = torch.tensor([[2., -1.], [3., .5], [.4, -.8]], dtype=torch.float64)
    l = -w; noise = w.new_tensor([[.2, .8], [-.6, .3], [1., -2.]])
    alpha = w.new_tensor([0., .36, 1.])
    actual_w, actual_l = impl.paired_noising(w, l, noise, alpha)
    expected_w, expected_l = [], []
    for a, row_w, row_l, row_n in zip(alpha.tolist(), w.tolist(), l.tolist(), noise.tolist()):
        expected_w.append([math.sqrt(a) * x + math.sqrt(1 - a) * n for x, n in zip(row_w, row_n)])
        expected_l.append([math.sqrt(a) * x + math.sqrt(1 - a) * n for x, n in zip(row_l, row_n)])
    close(actual_w, w.new_tensor(expected_w)); close(actual_l, w.new_tensor(expected_l))
    close(actual_w - actual_l, alpha.sqrt()[:, None] * (w - l))


@pytest.mark.t1
def test_independent_pairwise_objective(impl):
    """Scalar arithmetic checks the winner/loser signs, reference subtraction and half-beta convention."""
    values = [torch.tensor(v, dtype=torch.float64) for v in (
        [[1., 2.], [-1., .5]], [[-.5, .2], [2., 1.]],
        [[.4, 1.], [0., .8]], [[.2, -.5], [1., .3]], [[.1, .3], [-.2, .4]])]
    beta = 1.7; expected = 0.
    for w, l, rw, rl, noise in zip(*(v.tolist() for v in values)):
        errors = [sum((p - n) ** 2 for p, n in zip(row, noise)) / 2 for row in (w, l, rw, rl)]
        margin = -.5 * beta * (errors[0] - errors[1] - errors[2] + errors[3])
        expected += max(-margin, 0) + math.log1p(math.exp(-abs(margin)))
    close(impl.diffusion_dpo_loss(*values, beta), values[0].new_tensor(expected / 2), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_reference_baseline_and_stability(impl):
    """The reference policy yields log(2); very unfavorable margins retain finite loss and gradient."""
    w, l, noise = [torch.randn(3, 4, dtype=torch.float64) for _ in range(3)]
    close(impl.diffusion_dpo_loss(w, l, w, l, noise, 9.), w.new_tensor(math.log(2)))
    huge = torch.full((2, 3), 100., dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(huge)
    loss = impl.diffusion_dpo_loss(huge, zero, zero, zero, zero, 2.)
    gradient, = torch.autograd.grad(loss, huge)
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    close(loss, huge.new_tensor(10000.))


@pytest.mark.t2
def test_analytic_gradient_and_frozen_context(impl):
    """Only current predictions receive the logistic margin gradient; reference and target remain frozen."""
    values = [torch.randn(3, 2, dtype=torch.float64, requires_grad=True) for _ in range(5)]
    w, l, rw, rl, noise = values; beta = .8
    loss = impl.diffusion_dpo_loss(*values, beta)
    gradients = torch.autograd.grad(loss, values, allow_unused=True)
    with torch.no_grad():
        gap = ((w - noise).square().mean(1) - (l - noise).square().mean(1)
               - (rw - noise).square().mean(1) + (rl - noise).square().mean(1))
        factor = torch.sigmoid(.5 * beta * gap)[:, None] * beta / 6
        close(gradients[0], factor * (w - noise), atol=1e-12, rtol=1e-12)
        close(gradients[1], -factor * (l - noise), atol=1e-12, rtol=1e-12)
    assert all(g is None or not g.any() for g in gradients[2:])


@pytest.mark.t2
def test_feature_and_batch_duplication(impl):
    """Feature-mean errors and pair-mean loss are invariant to duplication along either axis."""
    values = [torch.randn(3, 4, dtype=torch.float64) for _ in range(5)]
    expected = impl.diffusion_dpo_loss(*values, .3)
    close(expected, impl.diffusion_dpo_loss(*(v.repeat(1, 3) for v in values), .3))
    close(expected, impl.diffusion_dpo_loss(*(v.repeat(2, 1) for v in values), .3))


@pytest.mark.t2
def test_noising_is_frozen_training_data(impl):
    """The paired forward-process helper does not create gradients through latents, noise or schedule."""
    w, l, noise = [torch.randn(2, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    alpha = torch.tensor([.2, .8], dtype=torch.float64, requires_grad=True)
    outputs = impl.paired_noising(w, l, noise, alpha)
    assert all(not value.requires_grad for value in outputs)


@pytest.mark.t3
def test_tiny_denoiser_preference_training(impl, artifact):
    """The submitted noising and objective train a two-parameter denoiser on fixed preference pairs."""
    result = run(impl.paired_noising, impl.diffusion_dpo_loss)
    artifact('diffusion_dpo_training', result)
    metrics = result['metrics']
    assert metrics['final_loss'] < .15 * metrics['initial_loss']
    assert metrics['minimum_margin'] > 2.
    assert metrics['shared_noise_error'] < 1e-12
