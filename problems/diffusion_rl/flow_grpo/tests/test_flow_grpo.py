import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion_rl.flow_grpo.experiment import run


@pytest.mark.t0
def test_shapes_dtype_and_no_mutation(impl):
    """Transition statistics and a multi-prompt trajectory loss preserve dtype and inputs."""
    for dtype in (torch.float32, torch.float64):
        x, v = torch.randn(3, 2, 4, dtype=dtype), torch.randn(3, 2, 4, dtype=dtype)
        before = x.clone()
        mean, std = impl.sde_transition(x, v, .7, .5, .6)
        assert mean.shape == x.shape and mean.dtype == std.dtype == dtype and std.shape == ()
        close(x, before)
        current = torch.randn(2, 3, 2, 4, dtype=dtype)
        saved = current.clone()
        loss = impl.flow_grpo_loss(current, current, current, current, current.new_tensor([.5, .7]), current.new_tensor([[1., 2., 3.], [3., 2., 1.]]), .2, .1)
        assert loss.shape == () and loss.dtype == dtype
        close(current, saved)


@pytest.mark.t0
def test_reject_singular_or_incompatible_inputs(impl):
    """Reject endpoint singularities, forward-time steps and zero-variance policy densities."""
    x = torch.ones(2, 3)
    for time, next_time, noise in ((1., .8, .7), (0., 0., .7), (.5, .6, .7), (.5, -.1, .7), (.5, .3, -.1), (.5, .3, float('nan'))):
        with pytest.raises(ValueError): impl.sde_transition(x, x, time, next_time, noise)
    c = torch.ones(1, 2, 2, 3); rewards = c.new_tensor([[1., 2.]])
    with pytest.raises(ValueError): impl.flow_grpo_loss(c, c, c, c, c.new_tensor([0., 1.]), rewards, .2, 0.)
    with pytest.raises(ValueError): impl.flow_grpo_loss(c, c, c, c, c.new_tensor([1., 1.]), rewards[:, :1], .2, 0.)
    with pytest.raises(ValueError): impl.flow_grpo_loss(c, c, c, c, c.new_tensor([1., 1.]), rewards, 1., 0.)


@pytest.mark.t1
def test_reverse_time_transition_scalar_oracle(impl):
    """Negative dt, score drift correction and sqrt(-dt) noise scale match scalar SDE arithmetic."""
    x = torch.tensor([[2., -1.], [.4, .8]], dtype=torch.float64)
    v = x.new_tensor([[-.5, 1.], [2., -.2]])
    t, s, a = .8, .55, .6
    mean, std = impl.sde_transition(x, v, t, s, a)
    sigma2 = a * a * t / (1 - t)
    expected = [[xx + (s - t) * (vv + sigma2 / (2 * t) * (xx + (1 - t) * vv)) for xx, vv in zip(rx, rv)] for rx, rv in zip(x.tolist(), v.tolist())]
    close(mean, x.new_tensor(expected), atol=1e-12, rtol=1e-12)
    close(std, x.new_tensor(math.sqrt(sigma2 * (t - s))), atol=1e-12, rtol=1e-12)
    ode, zero = impl.sde_transition(x, v, t, s, 0.)
    close(ode, x + (s - t) * v); close(zero, x.new_tensor(0.))


@pytest.mark.t1
def test_full_scalar_gaussian_objective(impl):
    """A loop oracle covers joint-coordinate log ratios, group centering, both clipping signs and exact KL."""
    c = torch.tensor([[[[.4, -.2], [.8, .1]], [[-.4, .6], [.1, -.3]], [[.2, .3], [-.7, .5]]],
                      [[[.7, -.8], [.3, .2]], [[-.6, .1], [.4, .7]], [[.2, -.5], [.9, -.4]]]], dtype=torch.float64)
    old = c * .2; ref = c * -.3; actions = c * 1.3 + .1
    std = c.new_tensor([.7, .4]); rewards = c.new_tensor([[1., -2., .5], [9., 12., 10.]])
    total = 0.; epsilon, beta, eps = .2, .07, 1e-8
    for p in range(2):
        r = rewards[p].tolist(); avg = sum(r) / 3
        scale = math.sqrt(sum((v - avg) ** 2 for v in r) / 3) + eps
        for g in range(3):
            advantage = (r[g] - avg) / scale
            for t in range(2):
                log_ratio, kl = 0., 0.
                for d in range(2):
                    cur, previous, frozen, action = [v[p, g, t, d].item() for v in (c, old, ref, actions)]
                    variance = std[t].item() ** 2
                    log_ratio += ((action - previous) ** 2 - (action - cur) ** 2) / (2 * variance)
                    kl += (cur - frozen) ** 2 / (2 * variance)
                ratio = math.exp(log_ratio)
                total += -min(ratio * advantage, min(1 + epsilon, max(1 - epsilon, ratio)) * advantage) + beta * kl
    close(impl.flow_grpo_loss(c, old, ref, actions, std, rewards, epsilon, beta), c.new_tensor(total / 12), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_transition_jacobian_and_frozen_state(impl):
    """The score correction contributes to the velocity Jacobian; rollout states remain frozen."""
    x, v = [torch.randn(2, 3, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    mean, std = impl.sde_transition(x, v, .6, .4, .7)
    dx, dv = torch.autograd.grad(mean.sum(), (x, v), allow_unused=True)
    assert dx is None or not dx.any()
    close(dv, v.new_full(v.shape, -.2 * (1 + .7 ** 2 / 2)), atol=1e-12, rtol=1e-12)
    assert not std.requires_grad


@pytest.mark.t2
def test_policy_gradient_and_frozen_rollout(impl):
    """At the old policy, the exact Gaussian score and KL gradients reach only current means."""
    current = torch.randn(2, 3, 2, 2, dtype=torch.float64, requires_grad=True)
    old = current.detach().clone().requires_grad_()
    reference, actions = [torch.randn_like(current, requires_grad=True) for _ in range(2)]
    std = current.new_tensor([.7, 1.2], requires_grad=True)
    rewards = current.new_tensor([[0., 1., 3.], [9., 10., 12.]], requires_grad=True)
    beta = .13
    grads = torch.autograd.grad(impl.flow_grpo_loss(current, old, reference, actions, std, rewards, .2, beta), (current, old, reference, actions, std, rewards), allow_unused=True)
    r = rewards.detach(); centered = r - r.mean(1, keepdim=True)
    adv = centered / (centered.square().mean(1, keepdim=True).sqrt() + 1e-8)
    expected = (-adv[:, :, None, None] * (actions.detach() - current.detach()) + beta * (current.detach() - reference.detach())) / std.detach()[None, None, :, None].square() / 12
    close(grads[0], expected, atol=1e-12, rtol=1e-12)
    assert all(g is None or not g.any() for g in grads[1:])


@pytest.mark.t2
def test_group_isolation_constant_rewards_and_time_reduction(impl):
    """Prompt reward offsets cancel independently, tied groups are finite, and duplicated time steps preserve loss."""
    values = [torch.randn(2, 3, 2, 2, dtype=torch.float64) * .2 for _ in range(4)]
    c = values[0]; std = c.new_tensor([.8, 1.1]); rewards = c.new_tensor([[1., 2., 4.], [-2., 0., 3.]])
    baseline = impl.flow_grpo_loss(*values, std, rewards, .2, .03)
    close(baseline, impl.flow_grpo_loss(*values, std, rewards + c.new_tensor([[40.], [-20.]]), .2, .03))
    close(baseline, impl.flow_grpo_loss(*(v.repeat(1, 1, 2, 1) for v in values), std.repeat(2), rewards, .2, .03))
    tied = impl.flow_grpo_loss(c, c, c, c, std, torch.ones_like(rewards), .2, .03)
    close(tied, c.new_tensor(0.))


@pytest.mark.t2
def test_clipped_gradient_for_both_advantage_signs(impl):
    """An improved positive sample and suppressed negative sample stop receiving surrogate gradients past clipping."""
    current = torch.tensor([[[[1.]], [[-1.]]]], dtype=torch.float64, requires_grad=True)
    old = torch.zeros_like(current); actions = torch.ones_like(current)
    loss = impl.flow_grpo_loss(current, old, old, actions, current.new_tensor([1.]), current.new_tensor([[1., -1.]]), .2, 0.)
    gradient, = torch.autograd.grad(loss, current)
    close(gradient, torch.zeros_like(current), atol=1e-12, rtol=1e-12)


@pytest.mark.t3
def test_online_stochastic_flow_training(impl, artifact):
    """Fresh Gaussian rollouts and terminal group rewards train a tiny conditional velocity model."""
    result = run(impl.sde_transition, impl.flow_grpo_loss)
    artifact('flow_grpo_online_training', result)
    metrics = result['metrics']
    assert metrics['final_expected_error'] < .15 * metrics['initial_expected_error']
    assert metrics['rollout_std'] > 0
    assert metrics['updates'] == 80


@pytest.mark.t2
def test_large_clipped_ratios_and_tied_rewards_are_finite(impl):
    """Zero advantages and saturated positive samples do not exponentiate an overflowing ratio."""
    c = torch.zeros(1, 2, 1, 1, dtype=torch.float64, requires_grad=True)
    old = torch.full_like(c, 50.); reference = torch.ones_like(c)
    loss = impl.flow_grpo_loss(c, old, reference, torch.zeros_like(c), c.new_tensor([1.]), c.new_tensor([[3., 3.]]), .2, .1)
    gradient, = torch.autograd.grad(loss, c)
    close(loss, c.new_tensor(.05)); close(gradient, torch.full_like(c, -.05))
    current = torch.tensor([[[[0.]], [[50.]]]], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([[[[50.]], [[0.]]]], dtype=torch.float64)
    loss = impl.flow_grpo_loss(current, old, current.detach(), torch.zeros_like(current), current.new_tensor([1.]), current.new_tensor([[1., -1.]]), .2, 0.)
    gradient, = torch.autograd.grad(loss, current)
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    close(loss, current.new_tensor(-.2 / (1 + 1e-8)))
    close(gradient, torch.zeros_like(current))
