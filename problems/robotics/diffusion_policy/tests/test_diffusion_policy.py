import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.robotics.diffusion_policy.experiment import run


@pytest.mark.t0
def test_contract_and_immutability(impl):
    """Action batches preserve their shape, dtype, device and original values."""
    for dtype in (torch.float32, torch.float64):
        x = torch.arange(24, dtype=dtype).reshape(2, 4, 3)
        n = torch.ones_like(x); a = x.new_tensor([.2, .8]); original = x.clone()
        noisy = impl.add_action_noise(x, n, a)
        assert noisy.shape == x.shape and noisy.dtype == dtype and noisy.device == x.device
        loss = impl.action_denoising_loss(noisy, n)
        assert loss.shape == () and loss.dtype == dtype
        chunk = impl.action_chunk(x, 2, 3)
        assert chunk.shape == (2, 3, 3) and chunk.dtype == dtype
        close(x, original); close(n, torch.ones_like(n)); close(a, x.new_tensor([.2, .8]))


@pytest.mark.t0
def test_invalid_inputs(impl):
    """Reject invalid horizons, noise schedules and accidental broadcasting."""
    x = torch.ones(2, 4, 3); a = torch.tensor([.2, .8])
    for alpha in (a[:1], torch.tensor([0., .8]), torch.tensor([.2, 1.]), a * float('nan')):
        with pytest.raises(ValueError): impl.add_action_noise(x, x, alpha)
    with pytest.raises(ValueError): impl.action_denoising_loss(x, x[:1])
    with pytest.raises(ValueError): impl.action_chunk(x[:, :0], 1, 1)
    for to, ta in ((0, 2), (True, 2), (2, 0), (2, 4), (1.5, 1)):
        with pytest.raises(ValueError): impl.action_chunk(x, to, ta)


@pytest.mark.t1
def test_forward_noise_scalar_oracle(impl):
    """Each example uses its own cumulative alpha on every time/action coordinate."""
    x = torch.tensor([[[1., -2.], [3., .5]], [[-4., 2.], [0., 7.]]], dtype=torch.float64)
    n = torch.tensor([[[.2, 1.], [-2., .5]], [[3., -1.], [2., 0.]]], dtype=x.dtype)
    a = x.new_tensor([.36, .81])
    expected = [[[math.sqrt(a[b].item()) * x[b,t,d].item() + math.sqrt(1-a[b].item()) * n[b,t,d].item() for d in range(2)] for t in range(2)] for b in range(2)]
    close(impl.add_action_noise(x, n, a), x.new_tensor(expected))


@pytest.mark.t1
def test_mse_and_observation_alignment(impl):
    """Noise error averages every coordinate; the first executed action is To-1."""
    pred = torch.tensor([[[1., 3.], [5., 7.]]], dtype=torch.float64)
    target = torch.ones_like(pred)
    close(impl.action_denoising_loss(pred, target), pred.new_tensor(14.))
    actions = torch.arange(36, dtype=pred.dtype).reshape(2, 6, 3)
    close(impl.action_chunk(actions, 3, 2), actions[:, [2, 3], :])
    close(impl.action_chunk(actions, 1, 6), actions)


@pytest.mark.t2
def test_forward_path_gradients(impl):
    """Forward noising preserves the action, sampled-noise and schedule Jacobians."""
    x = torch.tensor([[[2., -1.]], [[.5, 3.]]], dtype=torch.float64, requires_grad=True)
    n = torch.tensor([[[.2, .8]], [[-.3, 1.]]], dtype=x.dtype, requires_grad=True)
    a = torch.tensor([.25, .64], dtype=x.dtype, requires_grad=True)
    grad = torch.autograd.grad(impl.add_action_noise(x, n, a).sum(), (x,n,a))
    close(grad[0], a.detach().sqrt().reshape(2,1,1).expand_as(x))
    close(grad[1], (1-a.detach()).sqrt().reshape(2,1,1).expand_as(n))
    expected = (x.detach()/(2*a.detach().sqrt().reshape(2,1,1))-n.detach()/(2*(1-a.detach()).sqrt().reshape(2,1,1))).sum((1,2))
    close(grad[2], expected)


@pytest.mark.t2
def test_loss_gradient_and_frozen_noise(impl):
    """Only predicted epsilon receives gradients from the training objective."""
    p = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    n = torch.randn_like(p).requires_grad_()
    loss = impl.action_denoising_loss(p, n)
    gp, gn = torch.autograd.grad(loss, (p,n), allow_unused=True)
    close(gp, 2*(p.detach()-n.detach())/p.numel())
    assert gn is None or torch.count_nonzero(gn) == 0


@pytest.mark.t2
def test_chunk_gradient_and_prefix_exclusion(impl):
    """Execution excludes past actions and leaves later predictions for replanning."""
    x = torch.randn(2, 6, 2, dtype=torch.float64, requires_grad=True)
    chunk = impl.action_chunk(x, 3, 2)
    grad, = torch.autograd.grad(chunk.sum(), x)
    expected = torch.zeros_like(x); expected[:, 2:4] = 1
    close(grad, expected)
    changed = x.detach().clone(); changed[:, :2] = 100; changed[:, 4:] = -100
    close(impl.action_chunk(changed, 3, 2), chunk.detach())


@pytest.mark.t2
def test_reduction_and_permutation_invariance(impl):
    """Duplicating coordinates or permuting examples cannot change mean noise error."""
    p = torch.randn(3, 4, 2, dtype=torch.float64); n = torch.randn_like(p)
    loss = impl.action_denoising_loss(p,n)
    close(impl.action_denoising_loss(p.repeat(2,1,3),n.repeat(2,1,3)),loss)
    close(impl.action_denoising_loss(p.flip(0),n.flip(0)),loss)
    a = p.new_tensor([.2,.5,.9])
    close(impl.add_action_noise(p.flip(0),n.flip(0),a.flip(0)),impl.add_action_noise(p,n,a).flip(0))


@pytest.mark.t3
def test_conditioned_action_denoising_experiment(impl, artifact):
    """The submitted noising, loss and chunk jointly train a goal-conditioned action denoiser."""
    result = run(impl.add_action_noise, impl.action_denoising_loss, impl.action_chunk)
    artifact('diffusion_policy_action_training', result)
    m = result['metrics']
    assert m['final_loss'] < 1e-4 * m['initial_loss']
    assert m['executed_chunk_mse'] < 1e-5
    assert m['without_conditioning_mse'] > 100 * m['executed_chunk_mse']
